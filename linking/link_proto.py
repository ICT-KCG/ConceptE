import torch
import torch.nn as nn
from numpy import *
import networkx as nx
import matplotlib.pyplot as plt
from evaluation_link import HierarchyClusterEvaluation
from utils import build_vocab
import math
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn.functional as F
import requests
import json
import os


class node:
    def __init__(self, name, emb=0):
        self.child = []
        self.parent = None
        self.emb = emb
        self.name = name
        self.ancestors = []
        self.height = 1
        self.name_vec = None
        self.rep = None


from nltk.corpus import wordnet as wn


def chat_with_model(prompt, model="deepseek-v3:671b"):
    """
    Function to interact with the language model API.

    Args:
        prompt (str): The user's input message
        model (str): The model to use for generation

    Returns:
        str: The model's response
    """
    url = "https://uni-api.cstcloud.cn/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'EMPTY')}"
    }

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "you have the strong ability to name the event"},
            {
                "role": "user",
                "content": prompt
            },
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for HTTP errors

        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            return f"Error: Unexpected response format: {result}"

    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"


def calcu_path_sim(word1, word2):
    w1 = wn.synsets(word1)
    w2 = wn.synsets(word2)
    score = 0
    if len(w1) == 0 or len(w2) == 0:
        return 0
    for i in w1:
        for j in w2:
            score += i.path_similarity(j)
    return score / len(w1) / len(w2)


def Consinsimilarity(tensor1, tensor2):
    if tensor1.norm(dim=-1, keepdim=True) < 1e-3 or tensor2.norm(dim=-1, keepdim=True) < 1e-3:
        return 0
    normal_t1 = tensor1 / tensor1.norm(dim=-1, keepdim=True)
    normal_t2 = tensor2 / tensor2.norm(dim=-1, keepdim=True)
    return (normal_t1 * normal_t2).sum(dim=-1)


class Trees:
    def __init__(self, args, relations, rep):
        self.tree = None
        self.config = BertConfig.from_pretrained(args.bert_model, output_hidden_states=True, output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(args.bert_model, config=self.config)
        self.bert.eval()
        self.device = torch.device("cuda" if args.cuda else "cpu")  # 确保 device 正确

        # 加载 label mapping
        if args.dataset == "ace":
            from consts import LABEL_TRIGGERS_ACE, NUM_LABEL_ACE
            all_l_triggers, self.l_trigger2idx, self.l_idx2trigger = build_vocab(LABEL_TRIGGERS_ACE)
        elif args.dataset == "ere":
            from consts import LABEL_TRIGGERS_ERE, NUM_LABEL_ERE
            all_l_triggers, self.l_trigger2idx, self.l_idx2trigger = build_vocab(LABEL_TRIGGERS_ERE)
        elif args.dataset == "maven":
            from consts import LABEL_TRIGGERS_MAVEN
            all_l_triggers, self.l_trigger2idx, self.l_idx2trigger = build_vocab(LABEL_TRIGGERS_MAVEN)

        self.rep = rep  # 这里传入的是已知类的实例列表 (List of Tensors)
        self.tree = self.create_prototype(relations)

    def create_prototype(self, rel):
        """
        Baseline 修改: 构建树并计算 Prototype (Mean Vector)
        """
        linkmap = {}
        # 1. 构建树的拓扑结构
        for item in rel:
            # item 结构通常是 [[ParentName, ParentID], [ChildName, ChildID]]
            p_name = item[0][0]
            c_name = item[1][0]

            if p_name not in linkmap: linkmap[p_name] = node(p_name.split(":")[-1], item[0][1])
            if c_name not in linkmap: linkmap[c_name] = node(c_name.split(":")[-1], item[1][1])

            # 建立父子关系
            if linkmap[c_name] not in linkmap[p_name].child:
                linkmap[p_name].child.append(linkmap[c_name])
            linkmap[c_name].parent = linkmap[p_name]

        # 2. 初始化叶子节点的 Prototype (已知 Event Types)
        for name, n in linkmap.items():
            n.rep = None  # 初始化
            if name in self.l_trigger2idx:
                idx = self.l_trigger2idx[name]
                # self.rep[idx] 是 [N, 768] 的 tensor，取平均得到 [768]
                if idx < len(self.rep) and len(self.rep[idx]) > 0:
                    n.rep = torch.mean(self.rep[idx].to(self.device), dim=0)
                else:
                    n.rep = torch.zeros(768).to(self.device)

        # 3. 自底向上递归计算父节点的 Prototype
        # 找到根节点
        root = None
        for name, n in linkmap.items():
            if n.parent is None:
                root = n
                break

        self._compute_proto_recursive(root)

        # 补充一些属性 (height, ancestors) 以防其他代码用到
        for i, n in linkmap.items():
            height = 1
            n_b = n
            while n_b.parent != None:
                n.ancestors.append(n_b.parent)
                height += 1
                n_b = n_b.parent
            n.height = height

        return root

    def _compute_proto_recursive(self, node):
        """递归计算节点 Prototype: Mean of Children Prototypes"""
        if not node.child:
            # 叶子节点，如果前面没赋上值（比如不在训练集中），给个零向量
            if node.rep is None:
                node.rep = torch.zeros(768).to(self.device)
            return node.rep

        child_vecs = []
        for c in node.child:
            child_vecs.append(self._compute_proto_recursive(c))

        # 父节点表示 = 所有子节点表示的平均 (Mean of Means)
        # 也可以改成 Mean of all instances，但在没有 instance 访问权时 Mean of Means 更常用
        if child_vecs:
            stack = torch.stack(child_vecs)
            node.rep = torch.mean(stack, dim=0)
        else:
            node.rep = torch.zeros(768).to(self.device)

        return node.rep

    def search_pair(self, emb, headnode, i=None):
        """
        Baseline 修改: 基于 Cosine Similarity 的搜索
        emb: 新 Cluster 的 Prototype (Vector [768])
        headnode: 当前搜索树节点
        """
        # 1. 计算与当前节点的相似度
        # unsqueeze(0) 变成 [1, 768] 以匹配 cosine_similarity 格式
        current_sim = F.cosine_similarity(emb.unsqueeze(0), headnode.rep.unsqueeze(0)).item()

        best_child = None
        best_child_sim = -float('inf')

        # 2. 遍历子节点，看是否有更相似的
        for child in headnode.child:
            sim = F.cosine_similarity(emb.unsqueeze(0), child.rep.unsqueeze(0)).item()
            if sim > best_child_sim:
                best_child_sim = sim
                best_child = child

        # 3. 决策：如果最好的子节点比当前节点更相似，则向下递归；否则停止
        # 这里使用简单的贪心策略
        if best_child and best_child_sim > current_sim:
            return self.search_pair(emb, best_child, i)
        else:
            return headnode

    def search_pair2(self, emb, headnode, i=None):
        v = headnode.rep
        temp = torch.mm(v, emb.T).sum() / v.size(0) / emb.size(0) / sqrt(headnode.height)
        b = temp
        temp_node = None
        for children in headnode.child:
            v = children.rep

            r = torch.mm(v, emb.T).sum() / v.size(0) / emb.size(0) / sqrt(children.height)

            if r > temp:
                temp_node = children
                temp = r
        if temp > b + 1e-4:
            return self.search_pair(emb, temp_node, i)
        else:
            return headnode

    def display(self, node):
        dict = {}
        dict["name"] = node.name
        dict["children"] = []
        for child in node.child:
            dict["children"].append(self.display(child))
        return dict




def link(args, structure, info_test, rep):
    t = Trees(args, structure, rep)
    res = {}
    for i, item in enumerate(info_test):
        if item["name"] == None:
            continue
        res[item["name"]] = {"sons": [], "fathers": [], "instance": []}

        # --- 修改开始 ---
        # 原代码: node = t.search_pair(item["vec"].detach(), t.tree, item["name"])
        # 新代码: 使用 "emb" (Centroid) 而不是 "vec" (Instance Matrix)
        # info_test 是由 get_test_info 生成的，里面已经计算了 "emb" = mean(vec)

        query_vec = item["emb"].detach().to(t.device)
        node = t.search_pair(query_vec, t.tree, item["name"])
        # --- 修改结束 ---

        while node.name != 'event_type':
            res[item["name"]]['fathers'].append(node.name)
            node = node.parent
        res[item["name"]]["instance"] = item["instance"]
    return res


def link_ori(args, structure, info_test, rep):
    t = Trees(args, structure, rep)
    res = {}
    for i, item in enumerate(info_test):

        if item["name"] == None:
            continue
        res[item["name"]] = {"sons": [], "fathers": [], "instance": []}
        node = t.search_pair2(item["vec"].detach(), t.tree, item["name"])
        while node.name != 'event_type':
            res[item["name"]]['fathers'].append(node.name)
            node = node.parent
        res[item["name"]]["instance"] = item["instance"]
    return res


from bert_score import score


def link_LLM(args, structure, info_test, rep):
    # import openai
    import time
    # openai.api_key = ""

    res = {}
    for i, item in enumerate(info_test):
        if item["name"] == None:
            continue

        res[item["name"]] = {"sons": [], "fathers": [], "instance": []}
        if args.dataset == "ace":
            tree_node = [
                "Root",
                "Life",
                "Movement",
                "Transaction",
                "Conflict",
                "Contact",
                'Personnel',
                "Justice",
                'Justice:Trial-Hearing',
                'Life:Die',
                'Transaction:Transfer-Money',
                'Life:Injure',
                'Personnel:End-Position',
                'Personnel:Elect',
                'Contact:Meet',
                'Contact:Phone-Write',
                'Movement:Transport',
                'Conflict:Attack'
            ]
        elif args.dataset == "ere":
            tree_node = [
                'Root',
                'Conflict',
                'Movement',
                'Transaction',
                'Life',
                'Contact',
                'Transaction',
                'Personnel',
                'Conflict:Attack',
                'Movement:Transport-Person',
                'Transaction:Transfer-Money',
                'Contact:Contact',
                'Life:Die',
                'Contact:Broadcast',
                'Transaction:Transfer-Ownership',
                'Contact:Meet',
                'Personnel:End-Position',
                'Contact:Correspondence'
            ]
        elif args.dataset == "maven":
            tree_node = ['Violence', 'Attack', 'Military_operation', 'Hostile_encounter', 'Killing', 'Motion_vir',
                         'Motion', 'Self_motion', 'Arriving', 'Communication_vir', 'Statement', 'Action_vir',
                         'Social_event', 'Creating', 'Scenario', 'Catastrophe', 'Competition', 'Process_end',
                         'Process_start', 'Influence_vir', 'Causation', 'Conquering', 'AlterBadState', 'Bodily_harm',
                         'Destroying', 'Death', 'Change_vir', 'Coming_to_be', 'Root']

        tree = ",".join(tree_node)
        template = """
        It is known that we have a new event type {} and a hierarchical Tree structure composed
        of these existing events [{}]. Please tell me which existing 
        events should be linked to if you want to add a new event type to this Tree structure? 
        Your answer should be one of these existing event types without any other word!Your answer should be one of these existing event types without any other word!Your answer should be one of these existing event types without any other word!
        , the following is an example:\n

        input word: Personnel:Nominate\n
        answer: Personnel\n

        input word: {}\n
        answer:
        """.format(item["name_LLM"][0], tree, item["name_LLM"][0])
        # the type of the trigger is a mask event
        ans = None
        time.sleep(20)
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages = [
        #         {"role":"system","content":"you have the strong ability to name the event"},
        #         {"role":"user", "content":template}
        #     ],
        #     temperature = 0.2
        # )
        llm_result = chat_with_model(template)
        ans = llm_result.replace(" ", "")

        if ans not in tree_node:
            ans = "Root"

        maxnode = ans

        if maxnode == "Root":
            res[item["name"]]['fathers'] = []
        else:
            if len(maxnode.split(":")) > 1:
                res[item["name"]]['fathers'] = [maxnode.split(":")[0], maxnode]
            else:
                res[item["name"]]['fathers'] = [maxnode]
        res[item["name"]]["instance"] = item["instance"]
    return res


def link_wordnet(args, structure, info_test, rep):
    t = Trees(args, structure, rep)
    res = {}
    for i, item in enumerate(info_test):
        if item["name"] == None:
            continue
        res[item["name"]] = {"sons": [], "fathers": [], "instance": []}
        node = t.search_with_wordnet(item["vec"].detach(), t.tree, item["name"], item["name_LLM"][0])
        while node.name != 'event_type':
            res[item["name"]]['fathers'].append(node.name)
            node = node.parent
        res[item["name"]]["instance"] = item["instance"]
    return res
