import torch.nn as nn
import torch.nn.functional as F
import torch

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Model_gin_fp(nn.Module):
    def __init__(self, n_output=1, output_dim=512, dropout=0.2, encoder=None):
        super(Model_gin_fp, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output
        self.encoder = encoder
        self.dropout = dropout
        self.cross_attn = CrossModalAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=dropout
        )
        self.pre = nn.Sequential(
            nn.Linear(output_dim*2, 512),
            nn.Dropout(dropout),
            nn.Linear(512, self.n_output)
        )

        self.fc = nn.Sequential(
            nn.Linear(1489,1024),
            nn.Dropout(dropout),
            nn.Linear(1024,output_dim)
        )

    def forward(self, data):
        x, edge_index, batch, y ,edge_attr, fp, w= data.x, data.edge_index, data.batch, data.y, data.edge_attr, data.fps, data.w

        fp = fp.reshape(len(fp)//1489,1489)

        x1 = self.encoder(x, edge_index, edge_attr, batch)
        xf = F.normalize(x1)

        x_fp = self.fc(fp)
        x_fp = F.normalize(x_fp)

        x_sum = self.cross_attn ((xf,x_fp),dim=1)

        xc = self.pre(x_sum)
        out = xc.reshape(-1)

        non_999_indices = y != 999
        out_loss = out[non_999_indices]
        y_loss = y[non_999_indices]
        w_loss = w[non_999_indices]


        return out, y, w, out_loss, y_loss, w_loss


class Model_gat_fp(nn.Module):
    def __init__(self, n_output=1, output_dim=512, dropout=0.2, encoder=None):
        super(Model_gat_fp, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output
        self.encoder = encoder
        self.dropout = dropout
        self.cross_attn = CrossModalAttention(
            embed_dim=output_dim,
            num_heads=8,
            dropout=dropout
        )
        self.pre = nn.Sequential(
            nn.Linear(output_dim*2, 512),
            nn.Dropout(dropout),
            nn.Linear(512, self.n_output)
        )

        self.fc = nn.Sequential(
            nn.Linear(1489,1024),
            nn.Dropout(dropout),
            nn.Linear(1024,output_dim)
        )

    def forward(self, data):
        x, edge_index, batch, y ,edge_attr, fp, w= data.x, data.edge_index, data.batch, data.y, data.edge_attr, data.fps, data.w

        fp = fp.reshape(len(fp)//1489,1489)

        x1, g_weight = self.encoder(x, edge_index, edge_attr, batch)
        xf = F.normalize(x1)

        x_fp = self.fc(fp)
        x_fp = F.normalize(x_fp)

        x_sum = self.cross_attn((x1,x_fp),dim=1)

        xc = self.pre(x_sum)
        out = xc.reshape(-1)

        len_w = edge_index.shape[1]
        weight = g_weight[1]
        edge_weight = weight[:len_w]
        x_weight = weight[len_w:]

        non_999_indices = y != 999
        out_loss = out[non_999_indices]
        y_loss = y[non_999_indices]
        w_loss = w[non_999_indices]

        return out, y, w, edge_weight, x_weight, out_loss, y_loss, w_loss



import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.combined_dim = input_dim * 3
        self.gate = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim),
            nn.LayerNorm(self.combined_dim),
            nn.GELU()
        )

    def forward(self, feat1, feat2, feat3):
        combined = torch.cat([feat1, feat2, feat3], dim=1)
        gate = self.gate(combined)
        transformed = self.transform(combined)
        return gate * transformed




# ====== 新增跨模态注意力模块 ======

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.kv_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)



    def forward(self, query, key_value):

        if query.dim() == 2:
            query = query.unsqueeze(1)  # [batch, 1, dim]
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)  # [batch, 1, dim]

        # 键值投影
        key_value = self.kv_proj(key_value)
        attn_output, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value
        )
        # 残差连接 + 层归一化
        return self.norm(query + 0.5 * self.dropout(attn_output))

class Model_gnn_fp(nn.Module):
    def __init__(self, n_output=1, output_dim=512, dropout=0.3, temperature=0.1,
                 encoder1=None, encoder2=None, use_a=False, use_b=False, use_c=False):
        super(Model_gnn_fp, self).__init__()
        self.output_dim = output_dim
        self.n_output = n_output
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.dropout = dropout
        self.temperature = nn.Parameter(torch.tensor(0.15))
        self.contrast_loss_weight = 0.2  # 对比学习损失权重


        # 新增交叉注意力模块
        self.cross_attn_gnn2fp = CrossModalAttention(embed_dim=output_dim, num_heads=4)
        self.cross_attn_fp2gnn = CrossModalAttention(embed_dim=output_dim, num_heads=4)
        assert output_dim == 512, "output_dim必须保持512"
        self.gated_fusion = GatedFusion(input_dim=output_dim)

        self.pre = nn.Sequential(
            nn.Linear(output_dim * 3, 512),  # 输入维度必须等于实际特征维度
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, self.n_output)
        )

        self.fc = nn.Sequential(
            nn.Linear(1489, 2048),  # 原为1024
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),  # 新增过渡层
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim)
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=output_dim * 3,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        self.mlp_fc = nn.Sequential(
            nn.Linear(output_dim * 3, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim * 3)
        )

        # 对比学习投影层
        self.contrast_dim = 128
        self.proj_gnn = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.contrast_dim)
        )
        self.proj_fp = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.contrast_dim)
        )

    def contrastive_loss(self, z_gnn, z_fp, temperature=0.1):
        """
        计算跨模态对比损失（InfoNCE Loss）
        Args:
            z_gnn: GNN模态特征 [batch, contrast_dim]
            z_fp: 指纹模态特征 [batch, contrast_dim]
            temperature: 温度参数
        """
        batch_size = z_gnn.size(0)

        # 归一化特征
        z_gnn = F.normalize(z_gnn, p=2, dim=1)
        z_fp = F.normalize(z_fp, p=2, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(z_gnn, z_fp.T) / temperature  # [batch, batch]

        # 构造标签：对角线为1（正样本对）
        labels = torch.arange(batch_size).to(z_gnn.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(similarity_matrix, labels)
        loss += F.cross_entropy(similarity_matrix.T, labels)

        return loss / 2

    def forward(self, data):
        x, edge_index, batch, y, edge_attr, fp, ecfp, w = data.x, data.edge_index, data.batch, data.y, data.edge_attr, data.fps, data.ecfp, data.w
        temperature = torch.clamp(self.temperature, 0.05, 0.3)

        # 输入预处理
        fp = fp.reshape(-1, 1489)
        ecfp = ecfp.reshape(-1, 1024)

        # 编码器处理
        x1, g_weight = self.encoder1(x, edge_index, edge_attr, batch)
        xf1 = F.normalize(x1)
        x2 = self.encoder2(x, edge_index, edge_attr, batch)
        xf2 = F.normalize(x2)

        # 指纹特征处理
        x_fp = self.fc(fp)
        x_fp = F.normalize(x_fp)

        # ====== 交叉注意力处理 ======
        # GNN到FP的注意力
        xf1_attn = self.cross_attn_gnn2fp(
            query=xf1.unsqueeze(1),  # [batch, 1, output_dim]
            key_value=x_fp.unsqueeze(1)  # [batch, 1, output_dim]
        ).squeeze(1)  # [batch, output_dim]

        # FP到GNN的注意力
        xf2_attn = self.cross_attn_fp2gnn(
            query=x_fp.unsqueeze(1),
            key_value=xf2.unsqueeze(1)
        ).squeeze(1)

        xf1_combined = xf1 + 0.5 * xf1_attn
        xf2_combined = xf2 + 0.5 * xf2_attn

        # 门控融合（使用增强后的特征）
        xf = self.gated_fusion(xf1_combined, xf2_combined, x_fp)

        # 自注意力处理
        attn_output, _ = self.self_attention(
            xf.unsqueeze(1),
            xf.unsqueeze(1),
            xf.unsqueeze(1)
        )
        xf = attn_output.squeeze(1)

        # MLP特征增强
        xf = self.mlp_fc(xf)

        # 最终分类
        xc = self.pre(xf)
        out = xc.reshape(-1)

        # ====== 对比特征投影 ======
        z_gnn = self.proj_gnn(xf1_attn)  # 使用交叉注意力后的特征
        z_fp = self.proj_fp(xf2_attn)

        # 处理特殊标签
        non_999_indices = y != 999
        out_loss = out[non_999_indices]
        y_loss = y[non_999_indices]
        w_loss = w[non_999_indices]

        # 边权重处理
        len_w = edge_index.shape[1]
        weight = g_weight[1]
        edge_weight = weight[:len_w]
        x_weight = weight[len_w:]




        # ====== 对比损失计算 ======
        contrast_loss = self.contrastive_loss(z_gnn, z_fp, temperature=self.temperature)

        # ====== 分类损失计算 ======
        if self.training:
            classification_loss = F.binary_cross_entropy_with_logits(
                out_loss,
                y_loss,
                weight=w_loss
            )
            total_loss = classification_loss + self.contrast_loss_weight * contrast_loss
        return out, y, w, edge_weight, x_weight, out_loss, y_loss, w_loss, z_gnn, z_fp
