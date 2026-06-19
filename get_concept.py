import json
import os
import argparse
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import deque
import re
from json_repair import repair_json
from openai import OpenAI

def get_length(file):
    f_all_data = open(file, 'r').readlines()
    return len(f_all_data)




concept_prompt = """
You are an expert in information extraction. Your task is to **Conceptualize** the already-identified events.

## Goal
Given a JSON input that contains the original sentence and the identified events (each with a trigger, each only having "text"), you must return the **same structure** but add, for every trigger:
- "concept_name": a short abstract semantic category (reusable across contexts; not the surface word)
- "concept_description": 1 concise sentence describing the semantic meaning/role of the span in context

## Hard Rules (must follow)
1) **Preserve** all input events and triggers **exactly** (order and text). Do NOT add or remove spans, do not change any "text".
2) Only **add** fields: "concept_name" and "concept_description" to each trigger.
3) Use concise, schema-level concept names (e.g., "Attack", "Transport", "Meet", "End-Position", "Transfer-Money").
4) Output **JSON only**, no extra text.

### OUTPUT JSON SCHEMA
{{
  "events": [
    {{
      "trigger": {{
        "text": "...",
        "concept_name": "...",
        "concept_description": "..."
      }}
    }}
  ]
}}

### EXAMPLE
Input
{{
  "sentence": "Dan Snyder of Baden, Pennsylvania writes, \"Bush should torture the al Qaeda chief operations officer."
  "events": [
    {{
      "trigger": {{
        "text": "torture"
      }}
    }}
  ]
}}

Output:
{{
  "events": [
    {{
      "trigger": {{
        "text": "torture",
        "concept_name": "Attack",
        "concept_description": "An Attack Event is defined as a violent physical act causing harm or damage."
      }}
    }}
  ]
}}

# Your turn
{{
  "sentence": "{sentence}"
  "events": {event}
}}
"""




def chat_openai(query_prompt, query_model="Qwen3-32B"):
    chat_response = client.chat.completions.create(
        model=query_model,
        messages=[
            {"role": "user", "content": query_prompt},
        ],
        max_tokens=8192,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return chat_response.choices[0].message.content


max_retries = 20
retry_delay = 5
# sentence = "Malaysia ' s prison department has agreed to allow jailed politician Anwar Ibrahim to attend his daughter ' s wedding ceremony Friday after his bail hearing , his lawyer said Thursday ."
# print(prompt_identify.format(sentence=sentence))

parser = argparse.ArgumentParser()
parser.add_argument('--set_type', type=str, default="test_dev")
parser.add_argument('--openai_api_base', type=str, default="http://xxxx")
parser.add_argument('--openai_api_key', type=str, default="EMPTY")
parser.add_argument('--model', type=str, default="Qwen3-32B")
parser.add_argument('--start', type=int, default="0")
parser.add_argument('--end', type=int, default=None)

args = parser.parse_args()

set_type = args.set_type
model_name = args.model
start = args.start
end = args.end


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = args.openai_api_key
openai_api_base = args.openai_api_base

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
# if not os.path.exists(dataset):
#     os.makedirs(dataset, exist_ok=True)
source_file = f"{set_type}.json"
target_file = f"{set_type}_concept_{model_name}.json"
0
with open(source_file, "r") as f, open(target_file, "a") as f_pred:
    total_lines = get_length(source_file)
    progress = tqdm(total=min(total_lines - start + 1, end - start + 1) if end else total_lines - start + 1,
                    ncols=75, desc='processing')
    current_line = 0
    for raw_line in f:
        current_line += 1
        # 跳过start之前的行
        if current_line < start:
            continue
        # 如果指定了end且当前行超过end，则结束处理
        if end and current_line > end:
            break
        line = json.loads(raw_line)
        progress.update(1)
        events = []
        if line["golden-event-mentions"]:
            for evt_gold in line["golden-event-mentions"]:
                evt = {"trigger": {"text": evt_gold["trigger"]["text"]}}
                events.append(evt)
            events_string = json.dumps(events)
            for attempt in range(max_retries):
                try:
                    sentence = line["sentence"]
                    query = concept_prompt.format(sentence=sentence, event=events_string)
                    # print(query)
                    result = chat_openai(query_prompt=query, query_model=model_name)
                    predict_concept = json.loads(repair_json(result))
                    for gold_evt, evt in zip(line["golden-event-mentions"], predict_concept["events"]):
                        if gold_evt["trigger"]["text"] != evt["trigger"]["text"]:
                            result = "format error" + result
                            break
                        gold_evt["trigger"]["trigger_concept_name"] = evt["trigger"]["concept_name"]
                        gold_evt["trigger"]["trigger_concept_description"] = evt["trigger"]["concept_description"]
                    if "Error: " in result:
                        print(f"Server Error, Retrying... ({attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    elif "format error" in result:
                        print(f"format Error, Retrying... ({attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                    else:
                        break  # 如果成功，则退出循环
                except json.decoder.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}. Retrying... ({attempt + 1}/{max_retries})")
                except Exception as e:
                    print(f"Unexpected error: {e}. Retrying... ({attempt + 1}/{max_retries})")

                # 如果出现异常，等待一段时间再重试
                time.sleep(retry_delay)
            else:
                print("API call failed after maximum retries.")
        f_pred.write(json.dumps(line, ensure_ascii=False) + "\n")
