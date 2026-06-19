from torch.utils.data import Dataset
from utils import clean_text
import torch
import json
import random
import numpy as np
import torch
import networkx as nx
from transformers import BertTokenizer, BertModel
import consts  # 导入 consts 文件


class InputExample(object):
    def __init__(
            self,
            unique_id,
            text,
            pos_span,
            label,
            arg_spans=None,
            trg_concept=None,
            arg_concepts=None,
    ):
        """
        arg_spans: List[[start, end], ...]  # 基于 example.text 的 word 索引
        trg_concept: dict 或 None，例如 {"name": str, "description": str}
        arg_concepts: List[dict]，每个 dict 例如 {"role": str, "name": str, "description": str}
        """
        self.unique_id = unique_id
        self.text = text  # list of tokens (words)
        self.pos_span = pos_span
        self.label = label
        self.pseudo = -1

        self.arg_spans = arg_spans if arg_spans is not None else []
        self.trg_concept = trg_concept  # 可以是 None
        self.arg_concepts = arg_concepts if arg_concepts is not None else []


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            unique_id,
            tokens,
            input_ids,
            input_mask,
            pos_span,
            valid_mask,
            mask_span,
            arg_spans=None,
    ):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.pos_span = pos_span
        self.valid_mask = valid_mask
        self.mask_span = mask_span

        # 仍然以 word 索引保存 argument span，后面在 model 里用 valid_mask 映射到 BPE
        self.arg_spans = arg_spans if arg_spans is not None else []


class Labeled_Dataset(Dataset):
    def __init__(self, args, examples, tokenizer):
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.examples = examples
        print(len(self.examples))
        actual_max_len = self.get_max_seq_length(self.examples, self.tokenizer)
        print(len(self.examples))
        self.features = self.convert_examples_to_features(
            args=args,
            examples=self.examples,
            seq_length=min(2 + actual_max_len, self.max_len) + 30,
            tokenizer=self.tokenizer,
        )

    def __getitem__(self, index):
        feat = self.features[index]
        ex = self.examples[index]

        input_ids = torch.tensor(feat.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feat.input_mask, dtype=torch.long)
        valid_mask = torch.tensor(feat.valid_mask, dtype=torch.long)
        label = ex.label
        pos_span = feat.pos_span
        mask_span = feat.mask_span

        # 新增：arg_spans 从 features 拿（word 索引）
        arg_spans = feat.arg_spans
        # 概念信息从 example 拿（原始结构 / 文本，后面在线编码）
        trg_concept = ex.trg_concept
        arg_concepts = ex.arg_concepts

        return (
            input_ids,
            input_mask,
            valid_mask,
            label,
            pos_span,
            mask_span,
            arg_spans,
            trg_concept,
            arg_concepts,
            index,
        )

    def __len__(self):
        return len(self.examples)

    def get_neighbors(self, neighbor_list):
        # 直接用 __getitem__ + collate_fn，保证返回结构一致
        batch = [self.__getitem__(i) for i in neighbor_list]
        return Labeled_Dataset.collate_fn(batch)

    def get_pos(self, pos_list):
        batch = []
        for i in pos_list:
            while True:
                x = np.random.randint(0, len(self.examples))
                sample = self.__getitem__(x)
                t_label = sample[3]  # 第 4 个是 label
                if t_label == i:
                    batch.append(sample)
                    break
        return Labeled_Dataset.collate_fn(batch)

    def get_neg(self, neg_list):
        batch = []
        for i in neg_list:
            while True:
                x = np.random.randint(0, len(self.examples))
                sample = self.__getitem__(x)
                t_label = sample[3]
                if t_label != i:
                    batch.append(sample)
                    break
        return Labeled_Dataset.collate_fn(batch)

    def collate_fn(data):
        # data: list of tuples from __getitem__
        data = list(zip(*data))
        (
            input_ids,
            input_mask,
            valid_mask,
            label,
            pos_span,
            mask_span,
            arg_spans,
            trg_concept,
            arg_concepts,
            index,
        ) = data

        input_ids = torch.stack(input_ids, dim=0)  # (B, L)
        input_mask = torch.stack(input_mask, dim=0)
        valid_mask = torch.stack(valid_mask, dim=0)
        label = torch.LongTensor(label)  # (B,)
        pos_span = torch.LongTensor(pos_span)  # (B, 2)
        mask_span = torch.LongTensor(mask_span)  # (B,)

        index = torch.LongTensor(index)

        # ---------- argument spans: pad 成 (B, max_arg, 2) ----------
        max_arg_num = max(len(x) for x in arg_spans) if len(arg_spans) > 0 else 0
        if max_arg_num == 0:
            # 没有任何 argument，则返回一个占位 tensor
            arg_spans_tensor = torch.full(
                (len(arg_spans), 1, 2), -1, dtype=torch.long
            )
        else:
            arg_spans_tensor = torch.full(
                (len(arg_spans), max_arg_num, 2), -1, dtype=torch.long
            )
            for i, spans in enumerate(arg_spans):
                for j, (s, e) in enumerate(spans):
                    arg_spans_tensor[i, j, 0] = s
                    arg_spans_tensor[i, j, 1] = e

        # 概念信息保持原样（list of dict / None），交给 model 端在线编码
        # trg_concept: 长度为 B 的 list
        # arg_concepts: 长度为 B 的 list，每个元素是 List[dict]

        return (
            input_ids,
            input_mask,
            valid_mask,
            label,
            pos_span,
            mask_span,
            arg_spans_tensor,
            trg_concept,
            arg_concepts,
            index,
        )

    # -------------------------- preprocess -------------------------- #

    def preprocess(path, dicts, event_dict):
        """
        ACE 风格：golden-event-mentions + arguments
        """
        datas = []
        unique_id = 0
        with open(path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                for event_mention in item["golden-event-mentions"]:
                    if event_mention["event_type"] in dicts:
                        text = item["words"]
                        pos_span = [
                            event_mention["trigger"]["start"],
                            event_mention["trigger"]["end"],
                        ]
                        event_type = event_mention["event_type"]
                        trigger = event_mention["trigger"]
                        trg_concept = {"trigger_concept_name": trigger["trigger_concept_name"],
                                       "trigger_concept_description": trigger["trigger_concept_description"]}
                        arg_concepts = []
                        # arguments 的 word 级 span
                        arg_spans = []
                        for arg in event_mention.get("arguments", []):
                            arg_spans.append(
                                [arg["start"], arg["end"]]
                            )
                            arg_concepts.append({"argument_concept_name": arg["argument_concept_name"],
                                                 "argument_concept_description": arg["argument_concept_description"]})

                        datas.append(
                            InputExample(
                                unique_id,
                                text,
                                pos_span,
                                event_dict[event_type],
                                arg_spans=arg_spans,
                                trg_concept=trg_concept,
                                arg_concepts=arg_concepts,
                            )
                        )
                        unique_id += 1
        return datas

    def preprocess_ere(path, dicts, event_dict):
        datas = []
        unique_id = 0
        with open(path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                for event_mention in item["event_mentions"]:
                    if event_mention["event_type"] in dicts:
                        text = item["tokens"]
                        pos_span = [
                            event_mention["trigger"]["start"],
                            event_mention["trigger"]["end"],
                        ]
                        event_type = event_mention["event_type"]
                        trigger = event_mention["trigger"]
                        trg_concept = {"trigger_concept_name": trigger["trigger_concept_name"],
                                       "trigger_concept_description": trigger["trigger_concept_description"]}
                        arg_concepts = []
                        # arguments 的 word 级 span
                        arg_spans = []
                        for arg in event_mention.get("arguments", []):
                            arg_text = arg["text"].split()
                            if arg_text[0] in text:
                                arg_start = text.index(arg_text[0])
                            else:
                                arg_start = 0
                            if arg_text[-1] in text:
                                arg_end = text.index(arg_text[-1])+1
                            else:
                                arg_end = arg_start + 1
                            if arg_end < arg_start:
                                arg_end = arg_start + 1
                            arg_spans.append(
                                [arg_start, arg_end]
                            )
                            arg_concepts.append({"argument_concept_name": arg["argument_concept_name"],
                                                 "argument_concept_description": arg["argument_concept_description"]})
                        datas.append(
                            InputExample(
                                unique_id,
                                text,
                                pos_span,
                                event_dict[event_type],
                                arg_spans=arg_spans,
                                trg_concept=trg_concept,
                                arg_concepts=arg_concepts,
                            )
                        )
                        unique_id += 1
        return datas

    def preprocess_maven(path, dict, event_dict):
        datas = []
        unique_id = 0
        num_samples = [0] * len(event_dict)
        with open(path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                for event_mention in item["events"]:
                    if event_mention["type"] in dict:
                        for i in range(len(event_mention["mention"])):
                            text = item["content"][
                                int(event_mention["mention"][i]["sent_id"])
                            ]["tokens"]
                            pos_span = [
                                event_mention["mention"][i]["offset"][0],
                                event_mention["mention"][i]["offset"][1],
                            ]
                            event_type = event_mention["type"]
                            trg_concept = {"trigger_concept_name": event_mention["mention"][i]["concept_name"],
                                           "trigger_concept_description": event_mention["mention"][i]["concept_description"]}
                            arg_spans = []
                            arg_concepts = []
                            num_samples[event_dict[event_type]] += 1
                            if num_samples[event_dict[event_type]] > 300:
                                continue
                            else:
                                datas.append(
                                    InputExample(
                                        unique_id,
                                        text,
                                        pos_span,
                                        event_dict[event_type],
                                        arg_spans=arg_spans,
                                        trg_concept=trg_concept,
                                        arg_concepts=arg_concepts,
                                    )
                                )
                                unique_id += 1
        return datas

    # -------------------------- common utils -------------------------- #

    def get_max_seq_length(self, examples, tokenizer):
        max_seq_len = -1
        remove_cnt = 0
        new_examples_list = []
        for example in examples:
            all_string = " ".join(example.text)
            all_string += str(example.trg_concept)
            all_string += str(example.arg_concepts)
            bert_tokens = tokenizer.tokenize(all_string)
            cur_len = len(bert_tokens)
            if cur_len <= self.max_len - 2:
                new_examples_list.append(example)
            else:
                remove_cnt += 1
                continue
            if cur_len > max_seq_len:
                max_seq_len = cur_len
        print("removed sentence number:{}".format(remove_cnt))
        self.examples = new_examples_list
        return max_seq_len

    def convert_examples_to_features(
            self, args, examples, seq_length, tokenizer, prompt_type=3
    ):
        """
        将句子 + prompt 编码为 BERT 输入。
        prompt_type=3 时，使用改造后的 prompt：
          - 保留原本 "the type of the <trigger> is a [MASK] event"
          - 追加 trigger / arguments / 概念 信息
        """

        def build_extra_prompt_tokens(example):
            """
            根据 InputExample 构造附加的 prompt token 列表：
            [TRIGGER] t ... [ARGUMENTS] a1 , a2 ...
            [TRIGGER_CONCEPT_NAME] ...
            [TRIGGER_CONCEPT_DESCRIPTION] ...
            [ARG_CONCEPT] [ARG_CONCEPT_NAME] ... [ARG_CONCEPT_DESCRIPTION] ...
            """
            extra = []

            # trigger span tokens
            trg_tokens = example.text[example.pos_span[0]: example.pos_span[1]]

            # ARG spans tokens
            arg_tokens_list = []
            for (s, e) in example.arg_spans:
                arg_tokens_list.append(example.text[s: e])

            # 1) 显式标注 trigger / arguments
            extra += ["We", "know", "this", "because"]
            extra += ["[TRIGGER]"] + trg_tokens
            if args.use_arg_emb:
                if len(arg_tokens_list) > 0:
                    extra += ["[ARGUMENTS]"]
                    for k, toks in enumerate(arg_tokens_list):
                        extra += toks
                        if k != len(arg_tokens_list) - 1:
                            extra += [","]

            # 2) trigger 概念
            if example.trg_concept is not None and args.use_trg_concept:
                name = example.trg_concept.get("trigger_concept_name", "") or ""
                desc = example.trg_concept.get("trigger_concept_description", "") or ""
                # name = clean_text(name)
                # desc = clean_text(desc)
                if name.strip():
                    extra += ["[TRIGGER_CONCEPT_NAME]"] + name.split()
                if desc.strip():
                    extra += ["[TRIGGER_CONCEPT_DESCRIPTION]"] + desc.split()

            # 3) argument 概念
            if getattr(example, "arg_concepts", None) and args.use_arg_concept:
                for c in example.arg_concepts:
                    if c is None:
                        continue
                    name = c.get("argument_concept_name", "") or ""
                    desc = c.get("argument_concept_description", "") or ""
                    # name = clean_text(name)
                    # desc = clean_text(desc)
                    if not (name.strip() or desc.strip()):
                        continue
                    extra += ["[ARG_CONCEPT]"]
                    if name.strip():
                        extra += ["[ARG_CONCEPT_NAME]"] + name.split()
                    if desc.strip():
                        extra += ["[ARG_CONCEPT_DESCRIPTION]"] + desc.split()

            return extra

        features = []
        for example in examples:
            tokens = []
            valid_mask = [0]
            tokens.append("[CLS]")

            trigger_tokens = example.text[example.pos_span[0]: example.pos_span[1]]

            
            if prompt_type == 1:
                prompt = trigger_tokens + ["is", "a", tokenizer.mask_token, "event"]
                mask_word_prefix = example.text + trigger_tokens + ["is", "a"]
            elif prompt_type == 2:
                prompt = [
                             "According",
                             "to",
                             "this",
                             ",",
                             "the",
                             "trigger",
                             "word",
                             "of",
                             "this",
                             tokenizer.mask_token,
                             "is",
                         ] + trigger_tokens + ["."]
                mask_word_prefix = example.text + [
                    "According",
                    "to",
                    "this",
                    ",",
                    "the",
                    "trigger",
                    "word",
                    "of",
                    "this",
                ]
            elif prompt_type == 3:
                base_prompt = (
                        ["In", "this", "sentence", ",", "the", "event", "type", "of"]
                        + trigger_tokens
                        + ["is", tokenizer.mask_token, "."]
                )
                # 在原有 prompt 基础上追加“结构 + 概念”信息
                extra_prompt = build_extra_prompt_tokens(example)
                prompt = base_prompt + extra_prompt

                mask_word_prefix = (
                        example.text
                        + ["the", "type", "of", "the"]
                        + trigger_tokens
                        + ["is", "a"]
                )
            else:
                # 兜底：不加 prompt
                prompt = []
                mask_word_prefix = example.text

            # ====== 句子 + prompt 一起 token 化 ======
            all_tokens = example.text + prompt

            for word in all_tokens:
                word_pieces = tokenizer.tokenize(word)
                tokens.extend(word_pieces)
                valid_mask.extend([1] + [0] * (len(word_pieces) - 1))

            if len(tokens) > seq_length - 1:
                tokens = tokens[0: (seq_length - 1)]
            valid_mask.extend([0])
            tokens.append("[SEP]")

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            token_ids = tokenizer.encode(
                " ".join(all_tokens), return_tensors="pt"
            ).squeeze(0)
            assert token_ids.size(0) == len(input_ids)
            prefix_bpe = tokenizer.encode(" ".join(mask_word_prefix))
            mask_span = len(prefix_bpe) - 1

            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
            while len(valid_mask) < seq_length:
                valid_mask.append(0)
            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length

            features.append(
                InputFeatures(
                    unique_id=example.unique_id,
                    tokens=tokens,  # bert_token
                    input_ids=input_ids,
                    input_mask=input_mask,
                    pos_span=example.pos_span,
                    valid_mask=valid_mask,
                    mask_span=mask_span,
                    arg_spans=example.arg_spans,
                )
            )
        return features


class unLabeled_Dataset(Dataset):
    def __init__(self, args, examples, tokenizer):
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.examples = examples
        print(len(self.examples))
        actual_max_len = self.get_max_seq_length(self.examples, self.tokenizer)
        print(len(self.examples))
        self.features = self.convert_examples_to_features(
            args=args,
            examples=self.examples,
            seq_length=min(2 + actual_max_len, self.max_len) + 30,
            tokenizer=self.tokenizer,
        )

    def __getitem__(self, index):
        feat = self.features[index]
        ex = self.examples[index]

        input_ids = torch.tensor(feat.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feat.input_mask, dtype=torch.long)
        valid_mask = torch.tensor(feat.valid_mask, dtype=torch.long)
        label = ex.label
        pos_span = feat.pos_span
        mask_span = feat.mask_span
        pseudo = ex.pseudo

        arg_spans = feat.arg_spans
        trg_concept = ex.trg_concept
        arg_concepts = ex.arg_concepts

        return (
            input_ids,
            input_mask,
            valid_mask,
            label,
            pos_span,
            mask_span,
            arg_spans,
            trg_concept,
            arg_concepts,
            index,
            pseudo,
        )

    def __len__(self):
        return len(self.examples)

    def get_neighbors(self, neighbor_list):
        batch = [self.__getitem__(i) for i in neighbor_list]
        return unLabeled_Dataset.collate_fn(batch)

    def collate_fn(data):
        data = list(zip(*data))
        (
            input_ids,
            input_mask,
            valid_mask,
            label,
            pos_span,
            mask_span,
            arg_spans,
            trg_concept,
            arg_concepts,
            index,
            pseudo,
        ) = data

        input_ids = torch.stack(input_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        valid_mask = torch.stack(valid_mask, dim=0)
        label = torch.LongTensor(label)
        pos_span = torch.LongTensor(pos_span)
        mask_span = torch.LongTensor(mask_span)
        index = torch.LongTensor(index)
        pseudo = torch.LongTensor(pseudo)

        max_arg_num = max(len(x) for x in arg_spans) if len(arg_spans) > 0 else 0
        if max_arg_num == 0:
            arg_spans_tensor = torch.full(
                (len(arg_spans), 1, 2), -1, dtype=torch.long
            )
        else:
            arg_spans_tensor = torch.full(
                (len(arg_spans), max_arg_num, 2), -1, dtype=torch.long
            )
            for i, spans in enumerate(arg_spans):
                for j, (s, e) in enumerate(spans):
                    arg_spans_tensor[i, j, 0] = s
                    arg_spans_tensor[i, j, 1] = e

        return (
            input_ids,
            input_mask,
            valid_mask,
            label,
            pos_span,
            mask_span,
            arg_spans_tensor,
            trg_concept,
            arg_concepts,
            index,
            pseudo,
        )

    # preprocess_xxx 和 Labeled_Dataset 相同，只是 label / pseudo 处理方式一样复用上面的逻辑

    def preprocess(path, dicts, event_dict):
        """
        ACE 风格：golden-event-mentions + arguments
        """
        datas = []
        unique_id = 0
        with open(path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                for event_mention in item["golden-event-mentions"]:
                    if event_mention["event_type"] in dicts:
                        text = item["words"]
                        pos_span = [
                            event_mention["trigger"]["start"],
                            event_mention["trigger"]["end"],
                        ]
                        event_type = event_mention["event_type"]
                        trigger = event_mention["trigger"]
                        trg_concept = {"trigger_concept_name": trigger["trigger_concept_name"],
                                       "trigger_concept_description": trigger["trigger_concept_description"]}
                        arg_concepts = []
                        # arguments 的 word 级 span
                        arg_spans = []
                        for arg in event_mention.get("arguments", []):
                            arg_spans.append(
                                [arg["start"], arg["end"]]
                            )
                            arg_concepts.append({"argument_concept_name": arg["argument_concept_name"],
                                                 "argument_concept_description": arg[
                                                     "argument_concept_description"]})

                        datas.append(
                            InputExample(
                                unique_id,
                                text,
                                pos_span,
                                event_dict[event_type],
                                arg_spans=arg_spans,
                                trg_concept=trg_concept,
                                arg_concepts=arg_concepts,
                            )
                        )
                        unique_id += 1
        return datas

    def preprocess_ere(path, dicts, event_dict):
        datas = []
        unique_id = 0
        with open(path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                for event_mention in item["event_mentions"]:
                    if event_mention["event_type"] in dicts:
                        text = item["tokens"]
                        pos_span = [
                            event_mention["trigger"]["start"],
                            event_mention["trigger"]["end"],
                        ]
                        event_type = event_mention["event_type"]
                        trigger = event_mention["trigger"]
                        trg_concept = {"trigger_concept_name": trigger["trigger_concept_name"],
                                       "trigger_concept_description": trigger["trigger_concept_description"]}
                        arg_concepts = []
                        # arguments 的 word 级 span
                        arg_spans = []
                        for arg in event_mention.get("arguments", []):
                            arg_text = arg["text"].split()
                            if arg_text[0] in text:
                                arg_start = text.index(arg_text[0])
                            else:
                                arg_start = 0
                            if arg_text[-1] in text:
                                arg_end = text.index(arg_text[-1])+1
                            else:
                                arg_end = arg_start + 1
                            if arg_end < arg_start:
                                arg_end = arg_start + 1
                            arg_spans.append(
                                [arg_start, arg_end]
                            )
                            arg_concepts.append({"argument_concept_name": arg["argument_concept_name"],
                                                 "argument_concept_description": arg["argument_concept_description"]})
                        datas.append(
                            InputExample(
                                unique_id,
                                text,
                                pos_span,
                                event_dict[event_type],
                                arg_spans=arg_spans,
                                trg_concept=trg_concept,
                                arg_concepts=arg_concepts,
                            )
                        )
                        unique_id += 1
        return datas

    def preprocess_maven(path, dict, event_dict):
        datas = []
        unique_id = 0
        num_samples = [0] * len(event_dict)
        with open(path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                for event_mention in item["events"]:
                    if event_mention["type"] in dict:
                        for i in range(len(event_mention["mention"])):
                            text = item["content"][
                                int(event_mention["mention"][i]["sent_id"])
                            ]["tokens"]
                            pos_span = [
                                event_mention["mention"][i]["offset"][0],
                                event_mention["mention"][i]["offset"][1],
                            ]
                            event_type = event_mention["type"]
                            trg_concept = {"trigger_concept_name": event_mention["mention"][i]["concept_name"],
                                           "trigger_concept_description": event_mention["mention"][i][
                                               "concept_description"]}
                            arg_spans = []
                            arg_concepts = []
                            num_samples[event_dict[event_type]] += 1
                            if num_samples[event_dict[event_type]] > 300:
                                continue
                            else:
                                datas.append(
                                    InputExample(
                                        unique_id,
                                        text,
                                        pos_span,
                                        event_dict[event_type],
                                        arg_spans=arg_spans,
                                        trg_concept=trg_concept,
                                        arg_concepts=arg_concepts,
                                    )
                                )
                                unique_id += 1
        return datas

    def get_max_seq_length(self, examples, tokenizer):
        max_seq_len = -1
        remove_cnt = 0
        new_examples_list = []
        for example in examples:
            all_string = " ".join(example.text)
            all_string += str(example.trg_concept)
            all_string += str(example.arg_concepts)
            bert_tokens = tokenizer.tokenize(all_string)
            cur_len = len(bert_tokens)
            if cur_len <= self.max_len - 2:
                new_examples_list.append(example)
            else:
                remove_cnt += 1
                continue
            if cur_len > max_seq_len:
                max_seq_len = cur_len
        print("removed sentence number:{}".format(remove_cnt))
        self.examples = new_examples_list
        return max_seq_len

    def convert_examples_to_features(
            self, args, examples, seq_length, tokenizer, prompt_type=3
    ):
        """
        将句子 + prompt 编码为 BERT 输入。
        prompt_type=3 时，使用改造后的 prompt：
          - 保留原本 "the type of the <trigger> is a [MASK] event"
          - 追加 trigger / arguments / 概念 信息
        """

        def build_extra_prompt_tokens(example):
            """
            根据 InputExample 构造附加的 prompt token 列表：
            [TRIGGER] t ... [ARGUMENTS] a1 , a2 ...
            [TRIGGER_CONCEPT_NAME] ...
            [TRIGGER_CONCEPT_DESCRIPTION] ...
            [ARG_CONCEPT] [ARG_CONCEPT_NAME] ... [ARG_CONCEPT_DESCRIPTION] ...
            """
            extra = []

            # trigger span tokens
            trg_tokens = example.text[example.pos_span[0]: example.pos_span[1]]

            # ARG spans tokens
            arg_tokens_list = []
            for (s, e) in example.arg_spans:
                arg_tokens_list.append(example.text[s: e])

            # 1) 显式标注 trigger / arguments
            extra += ["We", "know", "this", "because"]
            extra += ["[TRIGGER]"] + trg_tokens
            if args.use_arg_emb:
                if len(arg_tokens_list) > 0:
                    extra += ["[ARGUMENTS]"]
                    for k, toks in enumerate(arg_tokens_list):
                        extra += toks
                        if k != len(arg_tokens_list) - 1:
                            extra += [","]

            # 2) trigger 概念
            if example.trg_concept is not None and args.use_trg_concept:
                name = example.trg_concept.get("trigger_concept_name", "") or ""
                desc = example.trg_concept.get("trigger_concept_description", "") or ""
                # name = clean_text(name)
                # desc = clean_text(desc)
                if name.strip():
                    extra += ["[TRIGGER_CONCEPT_NAME]"] + name.split()
                if desc.strip():
                    extra += ["[TRIGGER_CONCEPT_DESCRIPTION]"] + desc.split()

            # 3) argument 概念
            if getattr(example, "arg_concepts", None) and args.use_arg_concept:
                for c in example.arg_concepts:
                    if c is None:
                        continue
                    name = c.get("argument_concept_name", "") or ""
                    desc = c.get("argument_concept_description", "") or ""
                    # name = clean_text(name)
                    # desc = clean_text(desc)
                    if not (name.strip() or desc.strip()):
                        continue
                    extra += ["[ARG_CONCEPT]"]
                    if name.strip():
                        extra += ["[ARG_CONCEPT_NAME]"] + name.split()
                    if desc.strip():
                        extra += ["[ARG_CONCEPT_DESCRIPTION]"] + desc.split()

            return extra

        features = []
        for example in examples:
            tokens = []
            valid_mask = [0]
            tokens.append("[CLS]")

            trigger_tokens = example.text[example.pos_span[0]: example.pos_span[1]]

            
            if prompt_type == 1:
                prompt = trigger_tokens + ["is", "a", tokenizer.mask_token, "event"]
                mask_word_prefix = example.text + trigger_tokens + ["is", "a"]
            elif prompt_type == 2:
                prompt = [
                             "According",
                             "to",
                             "this",
                             ",",
                             "the",
                             "trigger",
                             "word",
                             "of",
                             "this",
                             tokenizer.mask_token,
                             "is",
                         ] + trigger_tokens + ["."]
                mask_word_prefix = example.text + [
                    "According",
                    "to",
                    "this",
                    ",",
                    "the",
                    "trigger",
                    "word",
                    "of",
                    "this",
                ]
            elif prompt_type == 3:
                base_prompt = (
                        ["In", "this", "sentence", ",", "the", "event", "type", "of"]
                        + trigger_tokens
                        + ["is", tokenizer.mask_token, "."]
                )
                # 在原有 prompt 基础上追加“结构 + 概念”信息
                extra_prompt = build_extra_prompt_tokens(example)
                prompt = base_prompt + extra_prompt

                mask_word_prefix = (
                        example.text
                        + ["the", "type", "of", "the"]
                        + trigger_tokens
                        + ["is", "a"]
                )
            else:
                # 兜底：不加 prompt
                prompt = []
                mask_word_prefix = example.text

            # ====== 句子 + prompt 一起 token 化 ======
            all_tokens = example.text + prompt

            for word in all_tokens:
                word_pieces = tokenizer.tokenize(word)
                tokens.extend(word_pieces)
                valid_mask.extend([1] + [0] * (len(word_pieces) - 1))

            if len(tokens) > seq_length - 1:
                tokens = tokens[0: (seq_length - 1)]
            valid_mask.extend([0])
            tokens.append("[SEP]")

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            token_ids = tokenizer.encode(
                " ".join(all_tokens), return_tensors="pt"
            ).squeeze(0)
            assert token_ids.size(0) == len(input_ids)
            prefix_bpe = tokenizer.encode(" ".join(mask_word_prefix))
            mask_span = len(prefix_bpe) - 1

            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
            while len(valid_mask) < seq_length:
                valid_mask.append(0)
            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length

            features.append(
                InputFeatures(
                    unique_id=example.unique_id,
                    tokens=tokens,  # bert_token
                    input_ids=input_ids,
                    input_mask=input_mask,
                    pos_span=example.pos_span,
                    valid_mask=valid_mask,
                    mask_span=mask_span,
                    arg_spans=example.arg_spans,
                )
            )
        return features





def build_hierarchy(args, l_trigger2idx, tokenizer, bert_model):
    """
    基于 consts.py 构建层级结构和初始化向量
    """
    # 1. 根据数据集选择对应的结构列表
    if args.dataset == 'ace':
        train_struct = consts.ace_structure
        test_struct = consts.ace_structure_test
    elif args.dataset == 'ere':
        train_struct = consts.ere_structure
        test_struct = consts.ere_structure_test
    elif args.dataset == 'maven':
        train_struct = consts.maven_structure
        test_struct = consts.maven_structure_test
    else:
        raise ValueError("Unknown dataset")
    device = torch.device("cuda" if args.cuda else "cpu")
    bert_model = bert_model.to(device)

    # 2. 使用 NetworkX 构建图 (合并 Train 和 Test 结构以获得完整节点列表)
    # consts.py 格式: [[Parent, None], [Child, None]]
    G = nx.DiGraph()

    # 解析函数
    def parse_structure(struct_list):
        for item in struct_list:
            parent_name = item[0][0]
            child_name = item[1][0]
            G.add_edge(parent_name, child_name)

    parse_structure(train_struct)
    parse_structure(test_struct)

    # 3. ID 分配 (Node to ID)
    node2taxoid = {}
    taxoid2node = {}

    # A. 优先分配 Known Classes (必须与 l_trigger2idx 一致)
    # l_trigger2idx 包含了训练集中出现的叶子节点
    for label, idx in sorted(l_trigger2idx.items(), key=lambda x: x[1]):
        if label in G.nodes:
            node2taxoid[label] = idx
            taxoid2node[idx] = label
        else:
            print(f"Warning: Known label '{label}' not found in structure consts!")

    max_known_id = max(l_trigger2idx.values()) if l_trigger2idx else -1
    current_id = max_known_id + 1

    # B. 分配剩余节点 (父节点, Root, 以及 Test集中出现的 Unknown Leaves)
    sorted_nodes = sorted(list(G.nodes))
    for node in sorted_nodes:
        if node not in node2taxoid:
            node2taxoid[node] = current_id
            taxoid2node[current_id] = node
            current_id += 1

    num_taxo_nodes = current_id
    print(f"Total Taxonomy Nodes Constructed: {num_taxo_nodes}")

    # 4. 构建 label2taxoid (Known Label ID -> Taxo ID)
    # 因为我们特意让它们相等，所以这里是 Identity 映射，但为了严谨还是写出来
    label2taxoid = {}
    for label, idx in l_trigger2idx.items():
        if label in node2taxoid:
            label2taxoid[idx] = node2taxoid[label]

    # 5. 构建 hierarchy_relations {Parent_ID: [Child_IDs]} 用于训练
    # 策略：即使 Test 中的父子关系我们知道，但 Test 的 Child 没有实例数据，
    # 无法进行 Bottom-up 聚合训练。
    # 所以，我们只在 relations 中包含那些“子节点都在 node2taxoid 中”的关系吗？
    # 不，Taxonomy Memory 需要学习所有父节点。
    # 这里的 relations 是 topology。ConceptE 训练时，如果没有某个 child 的 instance，
    # aggregator 就只聚合它有的 children。

    hierarchy_relations = {}
    for node in G.nodes:
        children = list(G.successors(node))
        if len(children) > 0:
            p_id = node2taxoid[node]
            c_ids = [node2taxoid[c] for c in children]
            hierarchy_relations[p_id] = c_ids

    # 6. 类型编码初始化 (Semantic Initialization) - 代价最小方案
    # 使用 BERT 对节点名称进行编码
    print("Initializing taxonomy embeddings with BERT...")
    initial_taxo_embeddings = torch.zeros(num_taxo_nodes, bert_model.config.hidden_size).to(device)

    bert_model.eval()
    with torch.no_grad():
        for tid in range(num_taxo_nodes):
            node_name = taxoid2node[tid]

            # 清洗文本: "Life:Die" -> "Life Die"
            # 特殊处理: "event_type" -> "event" (或者是 "root event")
            if node_name == "event_type":
                clean_text = "event"
            else:
                clean_text = node_name.replace(":", " ").replace("_", " ")

            inputs = tokenizer(clean_text, return_tensors="pt", padding=True, truncation=True).to(device)
            # 取 [CLS] 向量
            embedding = bert_model(**inputs).last_hidden_state[:, 0, :]
            initial_taxo_embeddings[tid] = embedding

    return num_taxo_nodes, label2taxoid, hierarchy_relations, initial_taxo_embeddings, node2taxoid, taxoid2node