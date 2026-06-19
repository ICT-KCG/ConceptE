"""
The code is copied and adapted from https://github.com/Ac-Zyx/RoCORE
"""
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import random

def finetune(model, unfreeze_layers):
    params_name_mapping = ['embeddings', 'layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11', 'layer.12']
    for name, param in model.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if params_name_mapping[ele] in name:
                param.requires_grad = True
                break
    return model


class TaxonomyStructure:
    def __init__(self, hierarchy_relations, taxoid2node):
        """
        hierarchy_relations: {Parent_ID: [Child_IDs]} (ID索引)
        无需传入 root_id，自动推断
        """
        self.relations = hierarchy_relations

        # 1. 自动查找 Root ID
        for k, v in taxoid2node.items():
            if v == 'event_type':
                self.root_id = k

        if self.root_id is None:
            self.root_id = self._find_root()
            # 如果没找到，可能是空字典或只有环，视情况抛出异常或处理
            print("Warning: No root found in hierarchy_relations!")

        # 2. 计算每个节点的 Level 和 祖先集合
        self.node2level = {}  # {node_id: level_int}
        self.level2nodes = {}  # {level_int: [node_ids]}
        self.node2ancestors = {}  # {node_id: set(ancestor_ids)}
        self.taxoid2node = taxoid2node

        # 只有找到 root 才能进行结构分析
        if self.root_id is not None:
            self._analyze_structure()

    def _find_root(self):
        """
        核心逻辑：根节点 = 所有父节点集合 - 所有子节点集合
        """
        if not self.relations:
            return None

        all_children = set()
        for children_list in self.relations.values():
            for child in children_list:
                all_children.add(child)

        all_parents = set(self.relations.keys())

        # 集合差集运算
        potential_roots = list(all_parents - all_children)

        if not potential_roots:
            raise ValueError("Error: Cycle detected or no root found in hierarchy!")

        # 通常情况下应该只有一个根 (如 event_type)
        # 如果有多个(森林结构)，这里默认取第一个作为主根，或者你可以逻辑上支持多个
        return potential_roots[0]

    def _analyze_structure(self):
        # 使用 BFS 初始化层级
        from collections import deque
        queue = deque([(self.root_id, 0, set())])  # (current_id, level, ancestors)

        self.node2level[self.root_id] = 0
        self.level2nodes[0] = [self.root_id]
        self.node2ancestors[self.root_id] = set()

        while queue:
            curr, lvl, ancestors = queue.popleft()

            # 记录当前层级反向索引
            if lvl not in self.level2nodes:
                self.level2nodes[lvl] = []
            if curr not in self.level2nodes[lvl]:  # 避免重复
                self.level2nodes[lvl].append(curr)

            # 找孩子
            if curr in self.relations:
                children = self.relations[curr]
                # 当前节点的祖先 + 当前节点自己 = 孩子的祖先
                new_ancestors = ancestors.copy()
                new_ancestors.add(curr)

                for child in children:
                    self.node2level[child] = lvl + 1
                    self.node2ancestors[child] = new_ancestors
                    queue.append((child, lvl + 1, new_ancestors))

    def get_hard_negatives(self, node_id, num_neg=5):
        # ... (保持之前的负采样逻辑不变) ...
        current_level = self.node2level.get(node_id, -1)
        if current_level == -1: return [], set()

        ancestors = self.node2ancestors.get(node_id, set())
        neg_candidates = []

        # Level 2+ (如 Life:Die): 找 Level 1 的叔叔
        if current_level >= 2:
            parent_level = current_level - 1
            potential_uncles = self.level2nodes.get(parent_level, [])
            # 筛选：非祖先
            hard_negatives = [nid for nid in potential_uncles if nid not in ancestors]
            neg_candidates.extend(hard_negatives)

        # Level 1 (如 Life): 暂时没特异性负样本，依赖随机补足

        return neg_candidates, ancestors

    def display(self):
        """
        以树状结构打印当前的分类体系
        Args:
            taxoid2node: {id: label_name} 的映射字典
        """
        taxoid2node = self.taxoid2node
        print(f"\n[Taxonomy Structure Display] Root ID: {self.root_id}")
        if self.root_id is None:
            print("Empty structure.")
            return

        def _print_node_recursive(node_id, prefix="", is_last=True):
            # 获取名称
            name = taxoid2node.get(node_id, f"ID:{node_id}")
            level = self.node2level.get(node_id, -1)

            # 打印当前节点
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}{name} (L{level})")

            # 处理前缀
            new_prefix = prefix + ("    " if is_last else "│   ")

            # 递归打印子节点
            if node_id in self.relations:
                children = self.relations[node_id]
                count = len(children)
                for i, child_id in enumerate(children):
                    is_last_child = (i == count - 1)
                    _print_node_recursive(child_id, new_prefix, is_last_child)

        # 从根节点开始打印
        root_name = taxoid2node.get(self.root_id, f"ID:{self.root_id}")
        print(f"{root_name} (L0) [Root]")

        if self.root_id in self.relations:
            children = self.relations[self.root_id]
            for i, child in enumerate(children):
                _print_node_recursive(child, "", i == len(children) - 1)
        print("==================================================\n")

class Margin():
    def __init__(self, args, dict):
        from taxo import TaxStruct
        import codecs
        with codecs.open(args.taxo_path, encoding='utf-8') as f:
            tax_lines = f.readlines()
        self.tax_pairs = [line.strip().split(" ") for line in tax_lines]
        self.tax_graph = TaxStruct(self.tax_pairs)
        self.nodes = list(self.tax_graph.nodes.keys())
        self.dict = dict

    def get_margin(self, list_a, list_b):
        margin = []
        for k, i in enumerate(list_a):
            node_a = self.dict[i.item()]
            path_a = self.tax_graph.node2path[node_a]
            node_b = self.dict[list_b[k].item()]
            path_b = self.tax_graph.node2path[node_b]
            com = len(set(path_a).intersection(set(path_b)))
            m = max( min(( abs(len(path_a) - com) + abs(len(path_b) - com) ) / com, 2), 0.5 )
            margin.append(m)
        return margin

# 新增：蕴含打分函数 (非对称)
# 简单的做法是计算距离，复杂的可以用 Order Embedding
class EntailmentScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):  # 减小 hidden_dim 防止过拟合
        super().__init__()
        # 输入维度变成 4倍: [u, v, u*v, |u-v|]
        self.net = nn.Sequential(
            nn.Dropout(0.5),  # 1. 强力 Dropout：防止记住这 10 个样本
            nn.Linear(input_dim * 4, 64),  # 2. 变换特征
            nn.LayerNorm(64),  # 3. 归一化：消除 Known/New 的幅度差异，配合 ReLU
            nn.LeakyReLU(),  # 4. 激活
            nn.Linear(64, 1)  # 5. 输出分数
        )

    def forward(self, child, parent):
        # 1. 必须归一化
        u = F.normalize(child, p=2, dim=-1)
        v = F.normalize(parent, p=2, dim=-1)

        # 2. 构造交互特征
        features = torch.cat([
            u,
            v,
            u * v,  # 元素积 (最重要的特征)
            torch.abs(u - v)  # 绝对差
        ], dim=-1)

        return self.net(features)


# [新增] 递归聚合器
class RecursiveAggregator(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super().__init__()
        self.attn_W = nn.Linear(input_dim, input_dim)
        self.attn_v = nn.Linear(input_dim, 1)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        # 映射层，用于特征变换
        self.project = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU()
        )

    def forward(self, child_reps):
        # child_reps: (batch_size, num_children, input_dim) 或 (num_children, input_dim)
        if child_reps.dim() == 2:
            child_reps = child_reps.unsqueeze(0)  # (1, num_children, dim)

        # Attention
        x = torch.tanh(self.attn_W(child_reps))
        scores = self.attn_v(x)  # (B, N, 1)
        weights = F.softmax(scores, dim=1)

        # Weighted Sum
        agg_rep = torch.sum(weights * child_reps, dim=1)  # (B, dim)

        # Residual & Norm
        mean_rep = torch.mean(child_reps, dim=1)
        out = self.layer_norm(self.project(agg_rep) + mean_rep)
        return out


class ConceptE(nn.Module):  # Pooling as relation description representation
    def __init__(self, args, config, pretrained_model, unfreeze_layers=[]):
        super().__init__()
        self.args = args
        self.max_len = args.max_len
        self.num_class = args.num_class
        self.new_class = args.new_class
        self.hidden_dim = args.hidden_dim
        self.kmeans_dim = args.kmeans_dim
        self.initial_dim = config.hidden_size
        assert config.output_hidden_states is True

        self.unfreeze_layers = unfreeze_layers
        # 共享的 BERT 编码器（句子 + prompt 全部用它）
        self.pretrained_model = finetune(pretrained_model, self.unfreeze_layers)
        self.layer = args.layer  # 使用第几层 hidden_states 作为特征

        self.device = torch.device("cuda" if args.cuda else "cpu")

        # 相似度编码 + 解码（NCC 用）
        self.similarity_encoder = nn.Sequential(
            nn.Linear(self.initial_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.kmeans_dim),
        )
        self.similarity_decoder = nn.Sequential(
            nn.Linear(self.kmeans_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.initial_dim),
        )

        # 聚类 head（对比 + kmeans 空间）
        self.head = nn.Sequential(
            nn.Linear(self.initial_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.kmeans_dim),
        )
        # margin head（层次链接）
        self.margin_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.initial_dim, self.kmeans_dim),
        )

        self.ct_loss_u = CenterLoss_unlabel(
            dim_hidden=self.kmeans_dim, num_classes=self.new_class
        )
        self.ct_loss_l = CenterLoss_label(
            dim_hidden=self.kmeans_dim, num_classes=self.num_class
        )
        self.ce_loss = nn.CrossEntropyLoss()

        # 有监督 / 无监督分类 head
        self.labeled_head = nn.Linear(self.initial_dim, self.num_class)
        self.unlabeled_head = nn.Linear(self.initial_dim, self.new_class)

        # 需要 finetune 的 BERT 参数
        self.bert_params = []
        for name, param in self.pretrained_model.named_parameters():
            if param.requires_grad:
                self.bert_params.append(param)


        # ------- special concept token 的 id（在 main 里通过 tokenizer 填好） -------
        # - 如果没设置，对应 id 默认为 -1，表示该通道关闭
        self.trg_cname_id = getattr(args, "trg_concept_name_id", -1)
        self.trg_cdesc_id = getattr(args, "trg_concept_desc_id", -1)
        self.arg_cname_id = getattr(args, "arg_concept_name_id", -1)
        self.arg_cdesc_id = getattr(args, "arg_concept_desc_id", -1)
        self.sep_id = getattr(args, "sep_id", -1)

        self.use_trg_concept = getattr(args, "use_trg_concept", False)
        self.use_arg_concept = getattr(args, "use_arg_concept", False)
        self.use_arg_emb = getattr(args, "use_arg_emb", False)
        self.fuse_concept = getattr(args, "fuse_concept", False)

        # ------- 融合 trigger / args / concept 的投影层 -------
        # base: trigger span
        self.num_parts = 1

        # + argument span pooling
        if self.use_arg_emb:
            self.num_parts += 1

        # + trigger concept: name / description
        if self.use_trg_concept and self.trg_cname_id >= 0:
            self.num_parts += 1
        if self.use_trg_concept and self.trg_cdesc_id >= 0:
            self.num_parts += 1

        # + argument concept: name / description
        if self.use_arg_concept and self.arg_cname_id >= 0:
            self.num_parts += 1
        if self.use_arg_concept and self.arg_cdesc_id >= 0:
            self.num_parts += 1

        self.fuse_proj = nn.Linear(self.initial_dim * self.num_parts, self.initial_dim)

        # 1. 聚合器 (复用之前定义的 RecursiveAggregator)
        self.aggregator = RecursiveAggregator(self.initial_dim)

        # 2. 蕴含打分器
        self.entailment_scorer = EntailmentScorer(self.initial_dim)

        # 3. 本体树记忆 (Taxonomy Memory)
        # 假设我们有 N_total_nodes 个节点 (Known Leaves + Parents + Root)
        # 需要在外部统计好节点总数和 ID 映射
        self.num_taxo_nodes = getattr(args, "num_taxo_nodes", 100)
        self.taxonomy_embeddings = nn.Embedding(self.num_taxo_nodes, self.initial_dim)
        self.taxo_structure = TaxonomyStructure(args.hierarchy_relations, args.taxoid2node)
        self.taxo_structure.display()
        self.score_neg_loss = args.score_neg_loss
        self.score_pos_loss = args.score_pos_loss
        self.aggregator_loss = args.aggregator_loss
        self.noise_ratio = args.noise_ratio

    def forward_hierarchy(self, instance_feats, labels, hierarchy_relations, label2taxoid):
        """
        全动态递归训练逻辑：
        1. 只要当前 Batch 有实例，就聚合成 Leaf。
        2. 只要 Children 能聚合（或查表），就聚合成 Parent。
        3. Scorer 和 MSE 全部基于这些动态向量计算。
        """
        loss_hierarchy = 0.0
        device = instance_feats.device

        # -------------------------------------------------------
        # 1. 预处理：构建当前 Batch 中存在的 Leaf 动态向量
        # -------------------------------------------------------
        # dynamic_cache: {node_id: dynamic_tensor}
        def cosine_alignment_loss(pred, target):
            # 目标是相似度为 1 (方向相同)
            # CosineEmbeddingLoss input: (x1, x2, target=1) -> 1 - cos(x1, x2)
            # 或者直接用 1 - F.cosine_similarity
            return (1 - F.cosine_similarity(pred, target, dim=-1)).mean()
        dynamic_cache = {}

        unique_labels = torch.unique(labels)
        for label in unique_labels:
            taxo_id = label2taxoid[label.item()]

            # 聚合 Instance -> Leaf
            # 1. 取出正样本 (Clean Samples)
            mask_pos = (labels == label)
            pos_feats = instance_feats[mask_pos]
            # 2. 【核心修改】人工投毒：随机混入 30% 的负样本
            # 这里的负样本来自当前 Batch 里的其他类
            mask_neg = (labels != label)
            if mask_neg.sum() > 0 and self.noise_ratio > 0:
                neg_feats = instance_feats[mask_neg]

                # 决定混入多少噪音 (例如：正样本数量的 30%)
                num_noise = max(1, int(pos_feats.size(0) * self.noise_ratio))

                # 如果负样本不够，就重复采样
                if neg_feats.size(0) < num_noise:
                    indices = torch.randint(0, neg_feats.size(0), (num_noise,))
                    selected_noise = neg_feats[indices]
                else:
                    # 随机选 num_noise 个
                    perm = torch.randperm(neg_feats.size(0))
                    selected_noise = neg_feats[perm[:num_noise]]

                # 3. 混合：把正样本和噪音拼在一起
                mixed_feats = torch.cat([pos_feats, selected_noise], dim=0)
            else:
                mixed_feats = pos_feats

            # Dynamic Leaf
            pred_leaf_vec = self.aggregator(mixed_feats.unsqueeze(0)).squeeze(0)
            dynamic_cache[taxo_id] = pred_leaf_vec

            # Target: Golden Static Leaf (用于 Anchor 住语义空间，不让它飘走)
            target_leaf_vec = self.taxonomy_embeddings(torch.tensor(taxo_id, device=device))

            # Loss Goal 1: Leaf 聚合一致性
            # loss_hierarchy += F.mse_loss(pred_leaf_vec, target_leaf_vec)
            loss_hierarchy += self.aggregator_loss * cosine_alignment_loss(pred_leaf_vec, target_leaf_vec)


        # -------------------------------------------------------
        # 2. 定义递归函数：获取任意节点的动态表示
        # -------------------------------------------------------
        def get_node_dynamic_vec(node_id):
            # A. 如果缓存里有（是 Leaf 且刚算过，或者是已计算的 Parent），直接返回
            if node_id in dynamic_cache:
                return dynamic_cache[node_id]

            # B. 如果是中间节点，尝试递归聚合 Children
            if node_id in hierarchy_relations:
                children_ids = hierarchy_relations[node_id]
                child_vecs = []

                for cid in children_ids:
                    child_vecs.append(get_node_dynamic_vec(cid))

                # 聚合 Children -> Parent
                child_tensor = torch.stack(child_vecs)  # (Num_Child, Dim)

                # 这里的 Aggregator 同时由于处理 Leaf 和 Internal Node，能力被通过复用训练
                parent_vec = self.aggregator(child_tensor.unsqueeze(0)).squeeze(0)

                # 存入缓存，避免重复计算
                dynamic_cache[node_id] = parent_vec
                return parent_vec

            # C. [Fallback] 如果既没实例，又不是父节点（或是叶子但当前Batch没样本）
            # 只能退化为使用 Static Embedding
            # 这是为了工程可行性必须做的妥协
            return self.taxonomy_embeddings(torch.tensor(node_id, device=device))

        # -------------------------------------------------------
        # 3. 遍历关系，计算 Parent 级别的 Loss
        # -------------------------------------------------------
        for parent_id, children_ids in hierarchy_relations.items():
            # 获取 动态 Parent (递归触发计算)
            pred_parent_vec = get_node_dynamic_vec(parent_id)

            # 获取 静态 Golden Parent (用于 MSE 监督)
            real_parent_vec = self.taxonomy_embeddings(torch.tensor(parent_id, device=device))

            # Loss Goal 1 (续): Parent 聚合一致性
            # loss_hierarchy += F.mse_loss(pred_parent_vec, real_parent_vec)
            loss_hierarchy += self.aggregator_loss * cosine_alignment_loss(pred_parent_vec, real_parent_vec)


            # ---------------------------------------------------
            # Loss Goal 2: Scorer 打分 (Dynamic vs Dynamic)
            # ---------------------------------------------------

            # 准备 Children 的动态向量
            child_vecs_list = [get_node_dynamic_vec(cid) for cid in children_ids]
            dynamic_children_vecs = torch.stack(child_vecs_list)

            # 正样本构建: (Dynamic Children, Dynamic Parent)
            expanded_dynamic_parent = pred_parent_vec.unsqueeze(0).expand(len(children_ids), -1)

            pos_logits = self.entailment_scorer(dynamic_children_vecs, expanded_dynamic_parent)
            loss_hierarchy += self.score_pos_loss * F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))

            # ---------------------------------------------------
            # Loss Goal 3: 负采样 (Dynamic Negative) - 1对多
            # ---------------------------------------------------

            neg_vecs_flat = []
            if len(children_ids) == 0:
                continue
            sample_child_id = children_ids[0]
            child_level = self.taxo_structure.node2level.get(sample_child_id, 99)

            # 准备 Level 1 的所有节点 (作为 Level 2 孩子的 Hard Negatives)

            level_up_nodes = self.taxo_structure.level2nodes.get(child_level - 1, [])
            NUM_NEG_TARGET = 10
            for cid in children_ids:
                current_neg_ids = []

                # 获取该节点的祖先集合 (用于过滤，防止把爷爷当负样本)
                my_ancestors = self.taxo_structure.node2ancestors.get(cid, set())

                # === 策略分支 ===

                # Case 1: Level >= 2 (如 Life:Die)
                # 目标：负样本 = Level 1 的非父节点 (叔叔) + 随机 Level 2+
                if child_level >= 2:
                    # 1.1 加入所有的叔叔 (Level 1 中不是我祖先的)
                    uncles = [nid for nid in level_up_nodes if nid not in my_ancestors]
                    current_neg_ids.extend(uncles)
                if len(current_neg_ids) > NUM_NEG_TARGET:
                    current_neg_ids = random.sample(current_neg_ids, NUM_NEG_TARGET)
                while len(current_neg_ids) < NUM_NEG_TARGET:
                    rand_id = torch.randint(0, self.num_taxo_nodes, (1,)).item()

                    # 检查层级条件 (Level >= 1)
                    r_level = self.taxo_structure.node2level.get(rand_id, -1)
                    if rand_id != cid and rand_id not in my_ancestors and r_level >= 1:
                        # 去重 (防止随机到已经加进去的叔叔)
                        if rand_id not in current_neg_ids:
                            current_neg_ids.append(rand_id)

                # === 获取向量并堆叠 ===
                for nid in current_neg_ids:
                    neg_vecs_flat.append(get_node_dynamic_vec(nid))

                # 堆叠所有负样本向量
            flat_neg_parent_vecs = torch.stack(neg_vecs_flat)

            flat_child_vecs = dynamic_children_vecs.unsqueeze(1).expand(-1, NUM_NEG_TARGET, -1).reshape(-1,
                                                                                                 dynamic_children_vecs.size(
                                                                                                     -1))

            # 6. 计算 Loss
            # 输入形状: (N*5, Dim) vs (N*5, Dim)
            neg_logits = self.entailment_scorer(flat_child_vecs, flat_neg_parent_vecs)

            # Target 全为 0
            loss_hierarchy += self.score_neg_loss * F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))

        return loss_hierarchy


    @torch.no_grad()
    def normalize_head(self):
        w = self.labeled_head.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.labeled_head.weight.copy_(w)

        w = self.unlabeled_head.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.unlabeled_head.weight.copy_(w)

    # ================= 句子 + 结构 融合表示（概念已在 prompt 中） ================= #
    def _pool_special_token(self, encoder_layers, input_ids, token_id):
        """
        对 batch 内每个样本，将指定 special token 的所有位置平均池化。
        若该样本中不存在该 token，则返回全零向量。
        encoder_layers: (B, L, H)
        input_ids:      (B, L)
        token_id:       int
        return:         (B, H)
        """
        if token_id < 0:
            # 未配置该 token
            B, _, H = encoder_layers.size()
            return torch.zeros(B, H, device=encoder_layers.device)

        batch_size, seq_len, hidden_dim = encoder_layers.size()
        device = encoder_layers.device

        # mask: (B, L)，标 True 的位置就是 special token
        mask = (input_ids == token_id)

        feats = []
        for i in range(batch_size):
            idx = torch.nonzero(mask[i], as_tuple=False).view(-1)
            if idx.numel() == 0:
                feats.append(torch.zeros(hidden_dim, device=device))
            else:
                # 对该样本中所有出现位置做平均（如果你只想用第一个，可以改成 idx[0:1]）
                feats.append(encoder_layers[i, idx].mean(0))
        return torch.stack(feats, dim=0)


    def _pool_span_between_tokens(self, hidden_states, input_ids, start_token_id, end_token_id, pool_strategy="max"):
        """
        对 start_token_id 和 end_token_id 之间的 tokens 进行 Mean Pooling。
        """
        # 1. 找到 start_token 和 end_token 的位置索引 (Batch_size,)
        # argmax 会返回每一行第一个匹配到的索引
        start_indices = (input_ids == start_token_id).long().argmax(dim=1)
        end_indices = (input_ids == end_token_id).long().argmax(dim=1)

        # 2. 构建掩码 (Mask)
        batch_size, seq_len = input_ids.shape
        # 生成位置索引矩阵: [[0, 1, 2...], [0, 1, 2...]]
        range_vect = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # 掩码逻辑：索引 > start 且 索引 < end
        # 注意：unsqueeze(1) 是为了让维度匹配 (B, L)
        mask = (range_vect > start_indices.unsqueeze(1)) & (range_vect < end_indices.unsqueeze(1))

        # 3. 根据策略进行池化
        if pool_strategy == "mean":
            # Mean Pooling (平均池化)
            # 扩展 mask 维度以匹配 hidden_states: (B, L) -> (B, L, H)
            mask_expanded = mask.unsqueeze(-1).float()

            # 将 mask 外的向量置为 0，然后求和
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)  # (B, H)

            # 计算每个样本在该片段内的 token 数量
            sum_mask = mask_expanded.sum(dim=1)  # (B, H)

            # 避免除以 0 (加上一个极小值 1e-9)
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            # 得到平均向量
            pooled_feat = sum_embeddings / sum_mask

        elif pool_strategy == "max":
            # Max Pooling (最大池化)
            # 将 mask 外的向量置为极小的负值
            mask_expanded = mask.unsqueeze(-1).float()

            # 创建一个值替换张量，将非 mask 区域替换为非常小的值
            # 这样在 max pooling 时不会被选中
            min_value = torch.finfo(hidden_states.dtype).min
            masked_hidden = hidden_states * mask_expanded + (1 - mask_expanded) * min_value

            # 在序列维度上进行最大池化
            pooled_feat, _ = torch.max(masked_hidden, dim=1)  # (B, H)

        else:
            raise ValueError(f"不支持的池化策略: {pool_strategy}。请选择 'mean' 或 'max'。")

        return pooled_feat

    def get_instance_representation(
        self,
        input_ids,
        input_mask,
        valid_mask,
        pos_span,
        mask_span,      # 为兼容旧接口保留，当前实现未用
        arg_spans=None, # tensor (B, max_arg, 2) 或 None
        layer=None,
    ):
        """
        统一构造实例表示：
        - 一定包含 trigger span pooling
        - 若 use_arg_emb=True && 传入 arg_spans，则追加 argument span pooling
        - 若 use_trg_concept / use_arg_concept = True，则从 encoder_layers 中
          取出 [TRIGGER_CONCEPT_NAME]/[TRIGGER_CONCEPT_DESCRIPTION] 等 special token 的表示
        - 最终把所有 part 拼接后，用 self.fuse_proj 投影回 (B, H)
        """
        if layer is None:
            layer = self.layer

        # 1) 编码整句（包含 prompt + 各种 [XXX] 占位符）
        output = self.pretrained_model(
            input_ids, token_type_ids=None, attention_mask=input_mask
        )
        encoder_layers = output.hidden_states[layer]  # (B, L, H)
        batch_size, max_len, feat_dim = encoder_layers.size()

        # 2) 根据 valid_mask 压回到「按 word 对齐」的表示，用于对原句 span 做 pooling
        valid_output = torch.zeros(
            batch_size, max_len, feat_dim, dtype=torch.float, device=self.device
        )
        for i in range(batch_size):
            pos = 0
            for j in range(max_len):
                if valid_mask[i][j].item() == 1:
                    valid_output[i][pos] = encoder_layers[i][j]
                    pos += 1

        # 3) trigger span mean-pooling（依然是基于原句的 word 索引）
        if self.args.pool_strategy == "max":
            trg_feat = torch.stack(
                [
                    torch.max(
                        valid_output[i, pos_span[i][0]: pos_span[i][1] + 1, :], dim=0
                    )[0]
                    for i in range(batch_size)
                ],
                dim=0,
            )  # (B, H)
        else:
            trg_feat = torch.stack(
                [
                    torch.mean(
                        valid_output[i, pos_span[i][0]: pos_span[i][1] + 1, :], dim=0
                    )
                    for i in range(batch_size)
                ],
                dim=0,
            )  # (B, H)

        parts = [trg_feat]

        # 4) argument span pooling（可选，和之前一样）
        if self.use_arg_emb and (arg_spans is not None):
            if isinstance(arg_spans, torch.Tensor):
                arg_spans_ = arg_spans
            else:
                arg_spans_ = torch.tensor(arg_spans, dtype=torch.long, device=self.device)

            arg_feats = []
            for i in range(batch_size):
                spans_i = arg_spans_[i]  # (max_arg, 2)
                cur_arg_feats = []
                for s, e in spans_i:
                    s = int(s.item()) if torch.is_tensor(s) else int(s)
                    e = int(e.item()) if torch.is_tensor(e) else int(e)
                    if s < 0:
                        continue
                    cur_arg_feats.append(
                        torch.max(valid_output[i, s : e + 1, :], dim=0)[0]
                    )
                if len(cur_arg_feats) == 0:
                    arg_feats.append(torch.zeros_like(trg_feat[i]))
                else:
                    arg_feats.append(torch.stack(cur_arg_feats, dim=0).mean(0))
            arg_feat = torch.stack(arg_feats, dim=0)  # (B, H)
            parts.append(arg_feat)

        # 5) trigger 概念 special token：[TRIGGER_CONCEPT_NAME] / [TRIGGER_CONCEPT_DESCRIPTION]
        if self.use_trg_concept and self.fuse_concept:
            # 获取 hidden_states (通常是 encoder_layers 的最后一层输出)
            # 假设 encoder_layers 是 sequence_output，形状为 (B, Seq_Len, Hidden)
            # 如果 encoder_layers 是列表，请取最后一层: hidden_states = encoder_layers[-1]
            hidden_states = encoder_layers

            # 1. 处理 Trigger Concept Name (提取 "transfer - money")
            # 范围：Name token 到 Description token 之间
            if self.trg_cname_id >= 0 and self.trg_cdesc_id >= 0:
                trg_cname_feat = self._pool_span_between_tokens(
                    hidden_states,
                    input_ids,
                    start_token_id=self.trg_cname_id,
                    end_token_id=self.trg_cdesc_id,
                    pool_strategy=self.args.pool_strategy
                )  # (B, H)
                parts.append(trg_cname_feat)

            # 2. 处理 Trigger Concept Description (提取 "a transfer - money event involves...")
            # 范围：Description token 到 SEP token 之间
            if self.trg_cdesc_id >= 0:
                # 获取 SEP token ID (请根据你的 tokenizer 调整，BERT通常是 102)
                # 如果你的类里没有存 sep_id，可能需要手动指定或从 tokenizer 获取

                # if self.use_arg_concept: # 暂不考虑argument
                #     end_token_id = self.arg_cname_id

                sep_token_id = self.sep_id

                trg_cdesc_feat = self._pool_span_between_tokens(
                    hidden_states,
                    input_ids,
                    start_token_id=self.trg_cdesc_id,
                    end_token_id=sep_token_id,
                    pool_strategy=self.args.pool_strategy
                )  # (B, H)
                parts.append(trg_cdesc_feat)

        # 6) argument 概念 special token：[ARG_CONCEPT_NAME] / [ARG_CONCEPT_DESCRIPTION]
        if self.use_arg_concept:
            if self.arg_cname_id >= 0:
                arg_cname_feat = self._pool_special_token(
                    encoder_layers, input_ids, self.arg_cname_id
                )  # (B, H)
                parts.append(arg_cname_feat)

            if self.arg_cdesc_id >= 0:
                arg_cdesc_feat = self._pool_special_token(
                    encoder_layers, input_ids, self.arg_cdesc_id
                )  # (B, H)
                parts.append(arg_cdesc_feat)

        # 7) 融合 + 投影
        if len(parts) == 1:
            fused = parts[0]
        else:
            fused = torch.cat(parts, dim=-1)   # (B, num_parts * H)
            fused = self.fuse_proj(fused)      # (B, H)

        return fused

    # 保留旧接口（目前不在 forward 中直接用；可以后面逐步删掉）
    def get_pretrained_feature(
        self, input_ids, input_mask, valid_mask, pos_span, mask_span, layer=12
    ):
        output = self.pretrained_model(
            input_ids, token_type_ids=None, attention_mask=input_mask
        )
        encoder_layers = output.hidden_states[layer]
        batch_size = encoder_layers.size(0)
        max_len = encoder_layers.size(1)
        feat_dim = encoder_layers.size(2)

        mask_feat = torch.stack(
            [
                encoder_layers[i, mask_span[i] : mask_span[i] + 1, :]
                for i in range(batch_size)
            ],
            dim=0,
        ).squeeze(1)

        valid_output = torch.zeros(
            batch_size, max_len, feat_dim, dtype=torch.float, device=self.device
        )
        for i in range(batch_size):
            pos = 0
            for j in range(max_len):
                if valid_mask[i][j].item() == 1:
                    valid_output[i][pos] = encoder_layers[i][j]
                    pos += 1
        pretrained_feat = torch.stack(
            [
                torch.max(
                    valid_output[i, pos_span[i][0] : pos_span[i][1] + 1, :], dim=0
                )[0]
                for i in range(batch_size)
            ],
            dim=0,
        )
        return pretrained_feat, mask_feat

    def forward(self, data, msg="feat", using_mask=False):
        # 为了兼容 Labeled_Dataset 和 unLabeled_Dataset，这里做一个长度判断
        def _unpack(data):
            """根据 batch 的长度区分是否有 pseudo。"""
            if len(data) == 10:
                # labeled: (ids, mask, valid, label, pos_span, mask_span, arg_spans, trg_concept, arg_concepts, index)
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
                pseudo = None
            elif len(data) == 11:
                # unlabeled: 多一个 pseudo
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
            else:
                raise ValueError(f"Unexpected data tuple length: {len(data)}")
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

        # ----------------- NCC 聚类 head ----------------- #
        if msg == "con":
            (
                input_ids,
                input_mask,
                valid_mask,
                label,
                pos_span,
                mask_span,
                arg_spans,
                trg_concept,   # 保留但不再使用（概念已在 prompt 中）
                arg_concepts,  # 同上
                index,
                pseudo,
            ) = _unpack(data)

            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            valid_mask = valid_mask.to(self.device)
            label = label.to(self.device)
            pos_span = pos_span.to(self.device)
            mask_span = mask_span.to(self.device)
            arg_spans = arg_spans.to(self.device) if arg_spans is not None else None

            fused_feat = self.get_instance_representation(
                input_ids,
                input_mask,
                valid_mask,
                pos_span,
                mask_span,
                arg_spans=arg_spans,
            )
            logits = self.head(fused_feat)
            return logits

        # ----------------- 层级 margin head ----------------- #
        elif msg == "margin":
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
            ) = _unpack(data)

            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            valid_mask = valid_mask.to(self.device)
            label = label.to(self.device)
            pos_span = pos_span.to(self.device)
            mask_span = mask_span.to(self.device)
            arg_spans = arg_spans.to(self.device) if arg_spans is not None else None

            fused_feat = self.get_instance_representation(
                input_ids,
                input_mask,
                valid_mask,
                pos_span,
                mask_span,
                arg_spans=arg_spans,
            )
            logits = self.margin_head(fused_feat)
            return logits

        # ----------------- 相似度编码（对比学习） ----------------- #
        elif msg == "similarity":
            with torch.no_grad():
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
                ) = _unpack(data)

                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                valid_mask = valid_mask.to(self.device)
                label = label.to(self.device)
                pos_span = pos_span.to(self.device)
                mask_span = mask_span.to(self.device)
                arg_spans = (
                    arg_spans.to(self.device) if arg_spans is not None else None
                )

                fused_feat = self.get_instance_representation(
                    input_ids,
                    input_mask,
                    valid_mask,
                    pos_span,
                    mask_span,
                    arg_spans=arg_spans,
                )

            sia_rep = self.similarity_encoder(fused_feat)  # (B, kmeans_dim)
            return sia_rep

        # ----------------- 重构分支（autoencoder） ----------------- #
        elif msg == "reconstruct":
            with torch.no_grad():
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
                ) = _unpack(data)

                input_ids = input_ids.to(self.device)
                input_mask = input_mask.to(self.device)
                valid_mask = valid_mask.to(self.device)
                label = label.to(self.device)
                pos_span = pos_span.to(self.device)
                mask_span = mask_span.to(self.device)
                arg_spans = (
                    arg_spans.to(self.device) if arg_spans is not None else None
                )

                fused_feat = self.get_instance_representation(
                    input_ids,
                    input_mask,
                    valid_mask,
                    pos_span,
                    mask_span,
                    arg_spans=arg_spans,
                )

            sia_rep = self.similarity_encoder(fused_feat)
            rec_rep = self.similarity_decoder(sia_rep)
            rec_loss = (rec_rep - fused_feat).pow(2).mean(-1)
            return sia_rep, rec_loss

        # ----------------- 有监督分类 head ----------------- #
        elif msg == "labeled":
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
            ) = _unpack(data)

            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            valid_mask = valid_mask.to(self.device)
            label = label.to(self.device)
            pos_span = pos_span.to(self.device)
            mask_span = mask_span.to(self.device)
            arg_spans = arg_spans.to(self.device) if arg_spans is not None else None

            fused_feat = self.get_instance_representation(
                input_ids,
                input_mask,
                valid_mask,
                pos_span,
                mask_span,
                arg_spans=arg_spans,
            )
            logits = self.labeled_head(fused_feat)
            return logits

        # ----------------- 无监督新类型 head ----------------- #
        elif msg == "unlabeled":
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
            ) = _unpack(data)

            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            valid_mask = valid_mask.to(self.device)
            label = label.to(self.device)
            pos_span = pos_span.to(self.device)
            mask_span = mask_span.to(self.device)
            arg_spans = arg_spans.to(self.device) if arg_spans is not None else None

            fused_feat = self.get_instance_representation(
                input_ids,
                input_mask,
                valid_mask,
                pos_span,
                mask_span,
                arg_spans=arg_spans,
            )
            logits = self.unlabeled_head(fused_feat)
            return logits

        # ----------------- 只要特征（比如做 kmeans） ----------------- #
        elif msg == "feat":
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
            ) = _unpack(data)

            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            valid_mask = valid_mask.to(self.device)
            label = label.to(self.device)
            pos_span = pos_span.to(self.device)
            mask_span = mask_span.to(self.device)
            arg_spans = arg_spans.to(self.device) if arg_spans is not None else None

            fused_feat = self.get_instance_representation(
                input_ids,
                input_mask,
                valid_mask,
                pos_span,
                mask_span,
                arg_spans=arg_spans,
            )
            return fused_feat

        else:
            raise NotImplementedError("not implemented!")



class CenterLoss_label(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c = 1.0, use_cuda = True):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.delta = 1
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.centers = None
        self.alpha = 0.1

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, y, hidden):
        batch_size = hidden.size()[0]
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0) # (num_class, batch_size, hid_dim) => (batch_size, num_class, hid_dim)
        expanded_centers = self.centers.expand(batch_size, -1, -1) # (batch_size, num_class, hid_dim)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1) # (batch_size, num_class, hid_dim) => (batch_size, num_class)
        intra_distances = distance_centers.gather(1, y.unsqueeze(1)).squeeze() # (batch_size, num_class) => (batch_size, 1) => (batch_size)
        loss = 0.5 * self.lambda_c * torch.mean(intra_distances) # (batch_size) => scalar
        return loss


class CenterLoss_unlabel(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c = 1.0, use_cuda = True):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.delta = 1
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.centers = None
        self.alpha = 1.

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, y, hidden):
        batch_size = hidden.size()[0]
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0) # (num_class, batch_size, hid_dim) => (batch_size, num_class, hid_dim)
        expanded_centers = self.centers.expand(batch_size, -1, -1) # (batch_size, num_class, hid_dim)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1) # (batch_size, num_class, hid_dim) => (batch_size, num_class)
        intra_distances = distance_centers.gather(1, y.unsqueeze(1)).squeeze() # (batch_size, num_class) => (batch_size, 1) => (batch_size)
        q = 1.0/(1.0+distance_centers/self.alpha) # (batch_size, num_class)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        prob = q.gather(1, y.unsqueeze(1)).squeeze() # (batch_size)
        loss = 0.5 * self.lambda_c * torch.mean(intra_distances*prob) # (batch_size) => scalar
        return loss



class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss: https://arxiv.org/pdf/2002.05709.pdf. 
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


