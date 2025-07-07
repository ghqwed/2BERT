import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
BERT模型核心实现
包含嵌入层、自注意力机制和前馈网络
"""

class BertEmbeddings(nn.Module):
    """BERT的嵌入层，组合三种嵌入表示"""
    def __init__(self, config):
        super().__init__()
        # 词嵌入：将token ID映射为向量
        self.word_embeddings = nn.Embedding(config["vocab_size"], config["hidden_size"])
        # 位置嵌入：表示token在序列中的位置
        self.position_embeddings = nn.Embedding(config["max_position_embeddings"], config["hidden_size"])
        # 句子类型嵌入：区分不同句子(如句对中的句子A和B)
        self.token_type_embeddings = nn.Embedding(config["type_vocab_size"], config["hidden_size"])
        # 层归一化：稳定训练过程
        self.LayerNorm = nn.LayerNorm(config["hidden_size"])
        # Dropout：防止过拟合
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, input_ids, token_type_ids=None):
        # 生成位置ID [0, 1, 2,..., seq_len-1]
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # 扩展到batch维度
        
        # 如果没有提供句子类型ID，默认为0(单句情况)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        # 获取三种嵌入表示
        words_embeddings = self.word_embeddings(input_ids)  # 词嵌入
        position_embeddings = self.position_embeddings(position_ids)  # 位置嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # 句子类型嵌入
        
        # 组合三种嵌入表示
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        # 层归一化和Dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    """多头自注意力机制实现"""
    def __init__(self, config):
        super().__init__()
        # 注意力头数量和每个头的维度
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = int(config["hidden_size"] / config["num_attention_heads"])
        
        # Q、K、V矩阵的线性变换
        self.query = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.key = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.value = nn.Linear(config["hidden_size"], config["hidden_size"])
        
        # 注意力权重的Dropout
        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])

    def forward(self, hidden_states, attention_mask=None):
        batch_size = hidden_states.size(0)
        
        # 1. 线性变换得到Q、K、V
        q = self.query(hidden_states)  # [batch, seq_len, hidden_size]
        k = self.key(hidden_states)    # [batch, seq_len, hidden_size]
        v = self.value(hidden_states)  # [batch, seq_len, hidden_size]
        
        # 2. 分割多头：将hidden_size维度分割为num_heads * head_size
        q = q.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        # 3. 计算注意力分数：Q * K^T / sqrt(d_k)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 4. 应用注意力掩码(如padding部分的掩码)
        if attention_mask is not None:
            # 调整掩码形状以匹配注意力分数
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores + attention_mask
            
        # 5. 计算注意力权重(softmax + dropout)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 6. 计算上下文向量：注意力权重 * V
        context = torch.matmul(attention_probs, v)
        # 7. 合并多头：转置并重塑回原始形状
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.num_attention_heads * self.attention_head_size)
        
        return context

class BertLayer(nn.Module):
    """单层Transformer编码器，包含自注意力和前馈网络"""
    def __init__(self, config):
        super().__init__()
        # 自注意力子层
        self.attention = BertSelfAttention(config)
        # 前馈网络中间层
        self.intermediate = nn.Linear(config["hidden_size"], config["intermediate_size"])
        # 前馈网络输出层
        self.output = nn.Linear(config["intermediate_size"], config["hidden_size"])
        # 层归一化
        self.LayerNorm = nn.LayerNorm(config["hidden_size"])
        # Dropout
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, hidden_states, attention_mask=None):
        # 1. 自注意力计算
        attention_output = self.attention(hidden_states, attention_mask)
        
        # 2. 残差连接和层归一化
        attention_output = self.LayerNorm(attention_output + hidden_states)
        
        # 3. 前馈网络(GELU激活)
        intermediate_output = F.gelu(self.intermediate(attention_output))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        
        # 4. 残差连接和层归一化
        layer_output = self.LayerNorm(layer_output + attention_output)
        return layer_output

class BertModel(nn.Module):
    """完整的BERT模型，包含多层Transformer编码器"""
    def __init__(self, config):
        super().__init__()
        # 嵌入层
        self.embeddings = BertEmbeddings(config)
        # 多层Transformer编码器
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config["num_hidden_layers"])])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # 1. 嵌入层处理
        embedding_output = self.embeddings(input_ids, token_type_ids)
        
        # 2. 多层Transformer处理
        hidden_states = embedding_output
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, attention_mask)
            
        return hidden_states
