"""
The code is copied and adapted from https://github.com/Ac-Zyx/RoCORE
"""
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

def finetune(model, unfreeze_layers):
    params_name_mapping = ['embeddings', 'layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5', 'layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11', 'layer.12']
    for name, param in model.named_parameters():
        param.requires_grad = False
        for ele in unfreeze_layers:
            if params_name_mapping[ele] in name:
                param.requires_grad = True
                break
    return model

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

    def _pool_span_between_tokens(self, hidden_states, input_ids, start_token_id, end_token_id, pool_strategy="mean"):
        """
        对 start_token_id 和 end_token_id 之间的 tokens 进行池化。

        Args:
            hidden_states: 隐藏状态张量 (batch_size, seq_len, hidden_size)
            input_ids: 输入ID张量 (batch_size, seq_len)
            start_token_id: 起始token的ID
            end_token_id: 结束token的ID
            pool_strategy: 池化策略，可选 "mean" 或 "max"

        Returns:
            池化后的特征 (batch_size, hidden_size)
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
                        valid_output[i, pos_span[i][0] : pos_span[i][1] + 1, :], dim=0
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


