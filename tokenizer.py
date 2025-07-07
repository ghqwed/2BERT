import json
from collections import defaultdict

"""
BERT分词器实现
将文本转换为模型可理解的token ID序列
使用WordPiece算法(简化版)
"""
class BertTokenizer:
    def __init__(self, vocab_file="vocab.txt"):
        """
        初始化分词器
        :param vocab_file: 词汇表文件路径，每行一个token
        """
        # 词汇表: {token: id}
        self.vocab = self.load_vocab(vocab_file)
        # 反向词汇表: {id: token}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        # 特殊token及其固定ID
        self.special_tokens = {
            "[PAD]": 0,    # 填充token，用于补齐序列长度
            "[UNK]": 100,  # 未知token，表示词汇表中不存在的词
            "[CLS]": 101,  # 分类token，位于序列开头
            "[SEP]": 102,  # 分隔token，用于分隔句子
            "[MASK]": 103  # 掩码token，用于预训练任务
        }
        
    def load_vocab(self, vocab_file):
        """
        加载词汇表文件
        :param vocab_file: 词汇表文件路径
        :return: 词汇表字典
        """
        vocab = defaultdict(int)
        try:
            with open(vocab_file, "r", encoding="utf-8") as f:
                for index, line in enumerate(f):
                    token = line.strip()  # 去除换行符和空格
                    vocab[token] = index  # 为每个token分配唯一ID
        except FileNotFoundError:
            print(f"Warning: {vocab_file} not found, using special tokens only")
        return vocab
    
    def tokenize(self, text):
        """
        基础分词方法(简化版)
        实际BERT使用WordPiece算法，这里简化为按空格分词
        :param text: 输入文本
        :return: token ID列表
        """
        # 1. 文本转为小写
        # 2. 按空格分割单词
        # 3. 将每个单词转换为对应的ID，未知词用[UNK]代替
        tokens = text.lower().split()
        return [self.vocab.get(token, self.special_tokens["[UNK]"]) 
                for token in tokens]
    
    def convert_tokens_to_ids(self, tokens):
        """
        将token列表转换为ID列表
        :param tokens: token列表
        :return: 对应的ID列表
        """
        return [self.vocab.get(token, self.special_tokens["[UNK]"]) 
                for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """
        将ID列表转换回token列表
        :param ids: token ID列表
        :return: 对应的token列表
        """
        return [self.inv_vocab.get(id, "[UNK]") for id in ids]

# 示例用法
if __name__ == "__main__":
    # 创建分词器实例
    tokenizer = BertTokenizer()
    # 测试分词功能
    print(tokenizer.tokenize("Hello world"))
