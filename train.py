import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import BertModel
from tokenizer import BertTokenizer
import json
import random

class BertPretrainDataset:
    """BERT预训练数据集，生成MLM和NSP训练样本"""
    def __init__(self, texts, tokenizer, max_length=128):
        """
        初始化数据集
        :param texts: 文本列表
        :param tokenizer: BERT分词器
        :param max_length: 最大序列长度
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        获取单个训练样本
        :param idx: 样本索引
        :return: 包含input_ids, attention_mask, mlm_labels, is_next_label的字典
        """
        text = self.texts[idx]
        # 1. 分词并截断到最大长度-2(保留空间给[CLS]和[SEP])
        tokens = self.tokenizer.tokenize(text)[:self.max_length-2]
        
        # 2. 添加特殊token: [CLS]在开头，[SEP]在结尾
        tokens = [101] + tokens + [102]  # 101=[CLS], 102=[SEP]
        
        # 3. 创建attention mask(1表示真实token，0表示padding)
        attention_mask = [1] * len(tokens)
        
        # 4. 填充到max_length
        padding = [0] * (self.max_length - len(tokens))
        tokens += padding
        attention_mask += padding
        
        # 5. 创建MLM标签(随机mask 15%的token)
        mlm_labels = tokens.copy()
        for i in range(1, len(tokens)-1):  # 跳过[CLS]和[SEP]
            if random.random() < 0.15:
                mlm_labels[i] = tokens[i]  # 保存原始token作为label
                tokens[i] = 103  # 103=[MASK]
                
        # 6. 简化版NSP: 随机生成is_next_label(0=不连续，1=连续)
        is_next_label = random.randint(0, 1)
        
        return {
            "input_ids": torch.tensor(tokens),  # 输入token IDs
            "attention_mask": torch.tensor(attention_mask),  # 注意力掩码
            "mlm_labels": torch.tensor(mlm_labels),  # MLM任务标签
            "is_next_label": torch.tensor(is_next_label)  # NSP任务标签
        }

class BertForPretraining(nn.Module):
    """BERT预训练模型，包含MLM和NSP两个任务头"""
    def __init__(self, config):
        """
        初始化预训练模型
        :param config: 模型配置字典
        """
        super().__init__()
        # BERT主干模型
        self.bert = BertModel(config)
        # MLM任务头：预测被mask的token
        self.mlm_head = nn.Linear(config["hidden_size"], config["vocab_size"])
        # NSP任务头：预测句子是否连续
        self.nsp_head = nn.Linear(config["hidden_size"], 2)
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播
        :param input_ids: 输入token IDs [batch_size, seq_len]
        :param attention_mask: 注意力掩码 [batch_size, seq_len]
        :return: (mlm_scores, nsp_scores)
        """
        # 1. 通过BERT主干获取序列输出
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs  # [batch_size, seq_len, hidden_size]
        
        # 2. MLM预测：对每个位置的token进行分类预测
        mlm_scores = self.mlm_head(sequence_output)  # [batch_size, seq_len, vocab_size]
        
        # 3. NSP预测：使用[CLS]位置的输出预测句子关系
        cls_output = sequence_output[:, 0]  # 取[CLS]位置的输出 [batch_size, hidden_size]
        nsp_scores = self.nsp_head(cls_output)  # [batch_size, 2]
        
        return mlm_scores, nsp_scores

def train():
    """
    BERT预训练主函数
    执行MLM和NSP任务的联合训练
    """
    # 1. 加载模型配置
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # 2. 初始化模型和分词器
    tokenizer = BertTokenizer()
    model = BertForPretraining(config)
    
    # 3. 准备训练数据(简化版示例)
    texts = [
        "这是一个BERT模型的简化实现",
        "我们将实现Masked Language Model任务",
        "以及Next Sentence Prediction任务", 
        "这是自然语言处理中的重要技术"
    ]
    
    # 4. 创建数据集和数据加载器
    dataset = BertPretrainDataset(texts, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
    
    # 5. 设置优化器(Adam with learning rate 5e-5)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    
    # 6. 训练循环
    for epoch in range(3):  # 训练3个epoch
        for batch in dataloader:
            # 6.1 清空梯度
            optimizer.zero_grad()
            
            # 6.2 前向传播
            mlm_scores, nsp_scores = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            # 6.3 计算损失
            # MLM损失: 预测被mask的token
            mlm_loss = F.cross_entropy(
                mlm_scores.view(-1, config["vocab_size"]),
                batch["mlm_labels"].view(-1)
            )
            # NSP损失: 预测句子是否连续
            nsp_loss = F.cross_entropy(
                nsp_scores,
                batch["is_next_label"]
            )
            # 总损失 = MLM损失 + NSP损失
            total_loss = mlm_loss + nsp_loss
            
            # 6.4 反向传播和参数更新
            total_loss.backward()
            optimizer.step()
            
            # 6.5 打印训练日志
            print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

if __name__ == "__main__":
    # 启动训练
    train()
