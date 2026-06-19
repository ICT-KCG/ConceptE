"""
Code modified from
https://github.com/wvangansbeke/Unsupervised-Classification
"""
import numpy as np
import torch


class MemoryBank(object):
    def __init__(self, n, dim, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature


    def mine_nearest_neighbors(self, topk, calculate_accuracy=False):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        
        else:
            return indices,distances

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets,idx):
        for i,f in enumerate(features):
            self.features[idx[i]].copy_(f.detach())
        for i,t in enumerate(targets):
            self.targets[idx[i]].copy_(t.detach())

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank, is_l):
    """
    loader 的 batch 结构：

    labeled:   (input_ids, input_mask, valid_mask,
                label, pos_span, mask_span,
                arg_spans, trg_concept, arg_concepts,
                index)

    unlabeled: (input_ids, input_mask, valid_mask,
                label, pos_span, mask_span,
                arg_spans, trg_concept, arg_concepts,
                index, pseudo)

    model.forward(batch, msg="feat") 会内部根据 len(batch) 判断是否有 pseudo。
    这里我们只需要：
      - 把所有 tensor 挪到和 model 一样的 device
      - 从 batch 中拿出 index（idx）和 label_ids
      - 把“完整 batch”丢给 model.forward
    """
    model.eval()
    memory_bank.reset()

    device = next(model.parameters()).device  # 跟随模型所在设备

    for i, batch in enumerate(loader):
        # batch 里既有 tensor，也有 python 对象（trg_concept, arg_concepts）
        new_batch = []
        for t in batch:
            if torch.is_tensor(t):
                new_batch.append(t.to(device, non_blocking=True))
            else:
                new_batch.append(t)
        batch = tuple(new_batch)

        if not is_l:  # unlabel: 最后一个是 pseudo，倒数第二个是 index
            idx = batch[-2]
        else:         # label: 最后一个是 index
            idx = batch[-1]

        label_ids = batch[3]  # 第 4 个始终是 label（即使 unlabeled 其实也有个占位 label）

        # 关键：直接把“完整 batch”传给 model，msg="feat" 会走 fused_feat 分支
        feature = model.forward(batch, msg="feat")

        # 更新 memory_bank：feature, label_ids, idx 都已经在和 memory_bank 一样的 device 上
        memory_bank.update(feature, label_ids, idx)

    print("finish filling memory bank")
