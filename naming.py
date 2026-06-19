import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover - optional dependency for noisy LLM outputs
    repair_json = None


NAMING_PROMPT_TEMPLATE = """You are an expert in event ontology construction.

Name a newly discovered event type from its representative conceptual abstractions and its predicted parent path.

Known event types, used only as naming-style reference:
<KNOWN_TYPES>

Predicted parent path:
<PARENT_PATH>

Representative conceptual abstractions from the cluster:
<CONCEPTS>

Return JSON only:
{
  "event_type_name": "A concise ontology-style event type name"
}

Rules:
1. The name should be short, type-level, and reusable across contexts.
2. The name should be consistent with the parent path and known ontology style.
3. Do not return a sentence or explanation.
"""


def read_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    for candidate in (cleaned, _first_json_object(cleaned)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            if repair_json is not None:
                try:
                    return json.loads(repair_json(candidate))
                except Exception:
                    pass

    raise ValueError(f"Model response is not valid JSON: {text[:500]}")


def _first_json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else ""


def sentence_from_record(record: Dict[str, Any]) -> str:
    if record.get("sentence"):
        return str(record["sentence"])
    if record.get("words"):
        return " ".join(map(str, record["words"]))
    return ""


def flatten_event_mentions(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    instances = []
    for record_id, record in enumerate(records):
        sentence = sentence_from_record(record)
        for mention_id, mention in enumerate(record.get("golden-event-mentions", [])):
            trigger = mention.get("trigger", {})
            concept_name = (
                trigger.get("trigger_concept_name")
                or trigger.get("concept_name")
                or mention.get("trigger_concept_name")
            )
            concept_description = (
                trigger.get("trigger_concept_description")
                or trigger.get("concept_description")
                or mention.get("trigger_concept_description")
            )
            instances.append(
                {
                    "global_id": len(instances),
                    "record_id": record_id,
                    "mention_id": mention_id,
                    "sentence": sentence,
                    "trigger": trigger.get("text", ""),
                    "concept_name": concept_name or "",
                    "concept_description": concept_description or "",
                }
            )
    return instances


def normalize_linking_results(raw: Any, path_order: str) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        items = []
        for cluster_id, value in raw.items():
            if not isinstance(value, dict):
                continue
            item = dict(value)
            item.setdefault("cluster_id", cluster_id)
            items.append(item)
    elif isinstance(raw, list):
        items = [dict(item) for item in raw if isinstance(item, dict)]
    else:
        raise ValueError("Linking result must be a JSON object or a JSON list.")

    normalized = []
    for offset, item in enumerate(items):
        cluster_id = item.get("cluster_id", item.get("name", item.get("id", offset)))
        instance_ids = item.get("instance", item.get("instances", item.get("ids", [])))
        parent_path = item.get("parent_path", item.get("path", item.get("fathers", [])))
        path_nodes = normalize_path(parent_path, path_order)
        normalized.append(
            {
                "cluster_id": cluster_id,
                "instance_ids": [int(x) for x in instance_ids],
                "parent_path": path_nodes,
            }
        )
    return normalized


def normalize_path(path: Any, path_order: str) -> List[str]:
    if path is None:
        return []
    if isinstance(path, str):
        nodes = [node.strip() for node in re.split(r"[:>/]", path) if node.strip()]
    else:
        nodes = [str(node).strip() for node in path if str(node).strip()]

    if path_order == "bottom_up":
        nodes = list(reversed(nodes))
    return nodes


def load_known_types(path: Optional[Path]) -> str:
    if path is None:
        return "Not provided."
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return "Not provided."

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return "\n".join(map(str, data))
        if isinstance(data, dict):
            return "\n".join(map(str, data.keys()))
    except json.JSONDecodeError:
        pass

    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def aggregate_cluster_concepts(
    cluster: Dict[str, Any],
    instances: List[Dict[str, Any]],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Counter]:
    name_counter = Counter()
    descriptions = defaultdict(Counter)
    trigger_counter = Counter()

    for instance_id in cluster["instance_ids"]:
        if instance_id < 0 or instance_id >= len(instances):
            continue
        instance = instances[instance_id]
        concept_name = instance["concept_name"].strip()
        concept_description = instance["concept_description"].strip()
        trigger = str(instance["trigger"]).strip()
        if concept_name:
            name_counter[concept_name] += 1
            if concept_description:
                descriptions[concept_name][concept_description] += 1
        if trigger:
            trigger_counter[trigger] += 1

    top_concepts = []
    for concept_name, count in name_counter.most_common(top_k):
        description = descriptions[concept_name].most_common(1)[0][0] if descriptions[concept_name] else ""
        top_concepts.append({"name": concept_name, "count": count, "description": description})

    return top_concepts, trigger_counter


def format_concepts_for_prompt(top_concepts: List[Dict[str, Any]]) -> str:
    if not top_concepts:
        return "Not available."
    lines = []
    for concept in top_concepts:
        description = concept.get("description") or "No description."
        lines.append(f"- {concept['name']} (count={concept['count']}): {description}")
    return "\n".join(lines)


def build_openai_client(api_key: str, base_url: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Please install the OpenAI Python package: pip install openai") from exc
    return OpenAI(api_key=api_key, base_url=base_url)


def call_llm(client: Any, prompt: str, args: argparse.Namespace) -> str:
    request = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.use_vllm_options:
        request["extra_body"] = {
            "top_k": args.top_k,
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        }

    response = client.chat.completions.create(**request)
    return response.choices[0].message.content or ""


def build_prompt(parent_path: List[str], top_concepts: List[Dict[str, Any]], known_types: str) -> str:
    parent = ":".join(parent_path) if parent_path else "event_type"
    return (
        NAMING_PROMPT_TEMPLATE.replace("<KNOWN_TYPES>", known_types)
        .replace("<PARENT_PATH>", parent)
        .replace("<CONCEPTS>", format_concepts_for_prompt(top_concepts))
    )


def name_with_llm(
    client: Any,
    parent_path: List[str],
    top_concepts: List[Dict[str, Any]],
    known_types: str,
    args: argparse.Namespace,
) -> str:
    prompt = build_prompt(parent_path, top_concepts, known_types)
    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response_text = call_llm(client, prompt, args)
            parsed = parse_json_response(response_text)
            raw_name = (
                parsed.get("event_type_name")
                or parsed.get("type_name")
                or parsed.get("name")
                or parsed.get("full_path", "").split(":")[-1]
            )
            name = clean_type_name(raw_name)
            if name:
                return name
            raise ValueError("The model did not return a valid event type name.")
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(args.retry_delay)

    raise RuntimeError(f"Failed to name cluster after {args.max_retries} retries: {last_error}")


def clean_type_name(name: Any) -> str:
    text = str(name or "").strip().strip("\"'")
    text = text.split(":")[-1].strip()
    text = re.sub(r"\s+", "-", text)
    return text


def fallback_name(top_concepts: List[Dict[str, Any]], trigger_counter: Counter) -> str:
    if top_concepts:
        return clean_type_name(top_concepts[0]["name"])
    if trigger_counter:
        return clean_type_name(trigger_counter.most_common(1)[0][0])
    return "Unknown-Event"


def full_path(parent_path: List[str], event_type_name: str) -> str:
    return ":".join(parent_path + [event_type_name]) if parent_path else event_type_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Name discovered event clusters after hierarchy linking.",
    )
    parser.add_argument("--concept_file", type=Path, required=True, help="JSON/JSONL file produced by get_concept.py.")
    parser.add_argument("--linking_file", type=Path, required=True, help="JSON file with linking results.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file for named clusters.")
    parser.add_argument("--known_types", type=Path, default=None, help="Optional text/JSON file of known event types.")
    parser.add_argument("--method", choices=["llm", "top_concept"], default="llm")
    parser.add_argument(
        "--path_order",
        choices=["bottom_up", "top_down"],
        default="bottom_up",
        help="Order of paths in linking_file. Existing ConceptE linking output uses bottom_up fathers.",
    )
    parser.add_argument("--top_k_concepts", type=int, default=5)
    parser.add_argument(
        "--openai_api_base",
        type=str,
        default=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "http://localhost:8000/v1",
    )
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", type=str, default="Qwen3-32B")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_delay", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--use_vllm_options", action="store_true", help="Pass top_k and enable_thinking to vLLM.")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--enable_thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    concept_records = read_json_or_jsonl(args.concept_file)
    if isinstance(concept_records, dict):
        concept_records = [concept_records]
    if not isinstance(concept_records, list):
        raise ValueError("--concept_file must contain a JSON list or JSONL records.")

    instances = flatten_event_mentions(concept_records)
    linking_raw = read_json_or_jsonl(args.linking_file)
    clusters = normalize_linking_results(linking_raw, args.path_order)
    known_types = load_known_types(args.known_types)
    client = None
    if args.method == "llm":
        client = build_openai_client(args.openai_api_key, args.openai_api_base)

    outputs = []
    for cluster in tqdm(clusters, desc="naming clusters", ncols=90):
        top_concepts, trigger_counter = aggregate_cluster_concepts(cluster, instances, args.top_k_concepts)
        if args.method == "llm":
            event_type_name = name_with_llm(client, cluster["parent_path"], top_concepts, known_types, args)
        else:
            event_type_name = fallback_name(top_concepts, trigger_counter)

        outputs.append(
            {
                "cluster_id": cluster["cluster_id"],
                "parent_path": cluster["parent_path"],
                "event_type_name": event_type_name,
                "full_path": full_path(cluster["parent_path"], event_type_name),
                "num_instances": len(cluster["instance_ids"]),
                "instance_ids": cluster["instance_ids"],
                "top_concepts": top_concepts,
            }
        )

    write_json(outputs, args.output)
    print(f"Saved {len(outputs)} named clusters to {args.output}")


if __name__ == "__main__":
    main()
