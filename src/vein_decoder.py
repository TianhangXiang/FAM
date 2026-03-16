# import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.embed_size = embed_size
        position = np.arange(max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embed_size, 2) * -(np.log(10000.0) / embed_size))
        
        pe = np.zeros((max_len, embed_size))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.register_buffer('pe', torch.tensor(pe, dtype=torch.float32))

    def forward(self, x):
        """
        x: 输入的张量，形状为 (batch_size, seq_len, embed_size)
        """
        seq_len = x.size(1)
        
        # 截取前 seq_len 个位置编码
        return self.pe[:seq_len].unsqueeze(0).detach()  # (1, seq_len, embed_size)
    
    
class EmbeddingGuidedVisualReconstructor(nn.Module):
    def __init__(self, input_dim=1536, mask_ratio=0.5, num_head=8, num_layers=1, reconstruct_strategy=0, use_projector=False, mask_generate_type="batch", detach_feature=True, add_pos_to_gt=False, dropout=0.1, vein_norm_embedding=False, pos_type='sin'):
        super(EmbeddingGuidedVisualReconstructor, self).__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.randn(input_dim))
        self.pos_type = pos_type
        
        if self.pos_type == 'sin':
            self.position_embedding = PositionalEncoding(input_dim)
        elif self.pos_type == 'embedding':
            self.position_embedding = nn.Embedding(4096, input_dim)
        else:
            pass
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_dim,
            nhead=num_head,
            dim_feedforward=input_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.reconstructor = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.reconstruction_strategy = reconstruct_strategy
        
        self.detach_feature = detach_feature
        
        self.mask_generate_type = mask_generate_type
        
        if self.mask_generate_type == 'sample':
            self.mask_generate_fn = self._generate_mask_per_sample
        elif self.mask_generate_type == 'batch':
            self.mask_generate_fn = self._generate_mask_per_batch
        else:
            pass
    
        self.add_pos_to_gt = add_pos_to_gt
        self.vein_norm_embedding = vein_norm_embedding
        
        """
        1. the 0th reconstruct_strategy is:
            Q: mask and unmasked visual features
            K, V: embedding
            
        2. the 1th reconstruct_strategy is:
            Q: embedding + unmasked visual features
            K, V: unmasked visual features
        """
    
    def _generate_mask_per_sample(self, B, L, device):
        mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        for i in range(B):
            k = max(1, int(L * self.mask_ratio))
            indices = torch.randperm(L, device=device)[:k]
            mask[i, indices] = True
        return mask
    
    def _generate_mask_per_batch(self, B, L, device):
        mask = torch.zeros((B, L), dtype=torch.bool, device=device)
        k = max(1, int(L * self.mask_ratio))
        indices = torch.randperm(L, device=device)[:k]
        mask[:, indices] = True 
        return mask
    
    def forward(self, visual_features, embedding_no_norm, embedding_norm):
        """
        inputs:
            visual_features: (B, L, D)
            embedding: (B, D)
        outputs:
            reconstructed_visual_features: (B, L, D)
            raw_visual_features: (B, L, D)
            mask_indice: (B, L)
        """
        
        if self.vein_norm_embedding:
            embedding = embedding_norm
        else:
            embedding = embedding_no_norm
        
        if visual_features is None:
            return None, None
        
        B, L, D = visual_features.shape

        # if self.detach_feature:
        #     visual_features = visual_features.detach()
        # else:
        #     pass
        
        visual_features = visual_features.detach()
        
        # 1. generate mask and mask sure that the mask is valid
        mask_indice = self.mask_generate_fn(B, L, visual_features.device)
        
        # 2. mask a part of raw visual features
        masked_and_unmasked_visual_features = visual_features.clone()
        masked_and_unmasked_visual_features[mask_indice] = self.mask_token.expand_as(masked_and_unmasked_visual_features[mask_indice])
        
        # 3. add position embedding and to visual_features
        position_indices = torch.arange(L).unsqueeze(0).repeat(B, 1).to(visual_features.device)
        masked_and_unmasked_visual_features = masked_and_unmasked_visual_features + self.position_embedding(position_indices)
        
        if self.add_pos_to_gt:
            visual_features = visual_features + self.position_embedding(visual_features)
        
        if self.reconstruction_strategy == 0:
            # # FIXME: maybe bug here
            # # fetch infomation from embedding to reconstruct visual features
            # embedding = embedding.unsqueeze(1)
            # # reconstructed_visual_features = self.transformer(tgt=masked_visual_features, memory=embedding)
            # for layer in self.reconstructor:
            #     reconstructed_visual_features = layer(tgt=masked_and_unmasked_visual_features, memory=embedding)
            #     masked_and_unmasked_visual_features = reconstructed_visual_features
            
            # reconstructed_visual_feats = reconstructed_visual_features[mask_indice]
            # target_feats = visual_features[mask_indice]
            pass

        elif self.reconstruction_strategy == 1:
            
            # (B, N, D)
            masked_features = masked_and_unmasked_visual_features[mask_indice].contiguous().view(B, -1, D)
            
            # (B, L - N, D)
            unmasked_visual_features = masked_and_unmasked_visual_features[~mask_indice].contiguous().view(B, -1, D)

            # (B, 1 + N, D)
            input_feats = torch.cat([embedding.unsqueeze(1), masked_features], dim=1)
            
            reconstructed_visual_features = self.reconstructor(tgt=input_feats, memory=unmasked_visual_features)
            
            reconstructed_visual_feats = reconstructed_visual_features[:, 1:, :]
            target_feats = visual_features[mask_indice].view(B, -1, D)
            
        elif self.reconstruction_strategy == 2:
            # debug: 直接把所有的 token 输进去，按道理模型要学习到 short cut, 也就是说loss 要降到 0
            # for layer in self.reconstructor:
            #     reconstructed_visual_features = layer(tgt=masked_and_unmasked_visual_features, memory=visual_features)
            #     masked_and_unmasked_visual_features = reconstructed_visual_features
            # reconstructed_visual_feats = reconstructed_visual_features[mask_indice]
            # target_feats = visual_features[mask_indice]
            
            # reconstructed_visual_features = self.reconstructor(tgt=masked_and_unmasked_visual_features, memory=visual_features)
            
            # reconstructed_visual_feats = reconstructed_visual_features[mask_indice]
            # target_feats = visual_features[mask_indice]
            
            pass
            
        elif self.reconstruction_strategy == 3:
            # debug: 看看方案 1的下界在哪
            # masked_features = masked_and_unmasked_visual_features[mask_indice].contiguous().view(B, -1, D)
            # unmasked_visual_features = masked_and_unmasked_visual_features[~mask_indice].contiguous().view(B, -1, D)
            
            # reconstructed_visual_feats = self.reconstructor(masked_features, unmasked_visual_features)
            # target_feats = visual_features[mask_indice].view(B, -1, D)
            pass
        else:
            raise NotImplementedError("reconstruction strategy {} is not implemented")
        
        return reconstructed_visual_feats, target_feats


def debug():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    # 假设已经定义了 EmbeddingGuidedVisualReconstructor 类

    # 超参数设置
    BATCH_SIZE = 32
    L = 128  # 序列长度
    D = 1536  # 特征维度
    MASK_RATIO = 0.5
    NUM_ITERS = 1000000  # 设置为训练的迭代次数
    LEARNING_RATE = 1e-4
    NUM_LAYERS = 1
    
    def cosine_similarity_loss(predictions, targets):
        # 计算余弦相似度，并返回损失 (1 - cosine_similarity)
        cosine_sim = F.cosine_similarity(predictions, targets, dim=-1)
        loss = 1 - cosine_sim  # Cosine similarity 越大，损失越小
        return loss.mean()  # 计算批次的平均损失

    # 模拟数据
    def generate_fake_data(batch_size, seq_length, feature_dim):
        visual_features = torch.randn(batch_size, seq_length, feature_dim)  # 随机生成视觉特征
        embedding = torch.randn(batch_size, feature_dim)  # 随机生成嵌入
        return visual_features, embedding

    # 初始化模型
    model = EmbeddingGuidedVisualReconstructor(input_dim=D, 
                                            mask_ratio=MASK_RATIO, 
                                            num_layers=NUM_LAYERS, 
                                            reconstruct_strategy=1,  # 使用策略2
                                            use_projector=False).cuda()
        
    # 优化器与损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()  # 使用均方误差损失函数

    # 训练循环
    for iteration in range(NUM_ITERS):
        model.train()  # 设置为训练模式
        visual_features, embedding = generate_fake_data(BATCH_SIZE, L, D)  # 生成伪数据
        visual_features = visual_features.cuda()
        embedding = embedding.cuda()
        
        optimizer.zero_grad()  # 清零梯度
        
        # 前向传播
        reconstructed_visual_feats, target_feats = model(visual_features, embedding)
        
        if reconstructed_visual_feats is None or target_feats is None:
            continue
        
        # 计算损失
        loss = cosine_similarity_loss(reconstructed_visual_feats, target_feats)
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        
        # 打印损失
        if iteration % 10 == 0:  # 每100次迭代打印一次
            print(f"Iteration [{iteration+1}/{NUM_ITERS}], Loss: {loss.item():.4f}")

        # 可选: 如果损失足够小，可以提前停止训练
        if loss.item() < 1e-5:
            print("Loss has reached a very small value, stopping training.")
            break
    
if __name__ == "__main__":
    debug()

    # # 模拟输入数据
    # batch_size = 4  # 假设 batch size 是 4
    # seq_len = 64  # 假设每个序列的长度是 10
    # embed_size = 3072
    # input_dim = 3072  # 输入维度
    # hidden_dim = 2 * input_dim  # 隐藏层维度

    # # 随机生成一些数据用于测试
    # visual_features = torch.randn(batch_size, seq_len, input_dim)  # 假设 visual_features 是 B x L x D
    # embedding = torch.randn(batch_size, input_dim)  # 假设 embedding 是 B x D

    # model = EmbeddingGuidedVisualReconstructor(input_dim=input_dim, hidden_dim=hidden_dim, mask_ratio=0.5, num_layers=3)

    # # 测试前向传播
    # sim = model(visual_features, embedding)

    # # 输出结果的形状和一部分值
    # print("Cosine similarity output shape:", sim.shape)
    # print("Cosine similarity output:", sim[:5])  # 打印出前 5 个相似度值