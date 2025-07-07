# BERT简化实现指南

## 1. 项目结构
```
BERT/
├── config.json       # 模型配置文件
├── model.py         # BERT模型实现
├── tokenizer.py     # 分词器实现  
├── train.py         # 训练脚本
└── vocab.txt        # 词汇表
```

## 2. 核心组件说明

### 2.1 模型架构
- **BertEmbeddings**: 组合词嵌入、位置嵌入和句子类型嵌入
- **BertSelfAttention**: 实现多头自注意力机制
- **BertLayer**: 单层Transformer编码器
- **BertModel**: 完整的BERT模型

### 2.2 预训练任务
- **MLM (Masked Language Model)**: 预测被mask的token
- **NSP (Next Sentence Prediction)**: 预测句子是否连续

## 3. 使用说明

### 3.1 训练流程
1. 准备训练数据
2. 初始化模型和分词器
3. 创建数据集和数据加载器
4. 设置优化器和训练参数
5. 执行训练循环

### 3.2 关键参数
- 学习率: 5e-5
- Batch Size: 2
- Epochs: 3
- 序列最大长度: 128
- Mask比例: 15%

## 4. 扩展建议
1. 使用更大规模的数据集
2. 增加模型深度和参数规模
3. 添加验证和测试流程
4. 实现更完整的分词器(WordPiece)
5. 添加学习率调度器

## 5. 常见问题
Q: 为什么loss下降不明显?
A: 这是简化实现，使用了很小的数据集和模型规模

Q: 如何评估模型效果?
A: 可以添加验证集准确率评估

Q: 如何保存和加载模型?
A: 使用torch.save()和torch.load()
