import torch
from tqdm import tqdm
import time
from datetime import timedelta

# 构建词汇表（Vocabulary）

CLS =  "[CLS]"  # padding符号, bert中综合信息符号


def build_dataset(config):
    """
    根据配置信息构建模型训练所需的数据集。
    参数：
    - config (object): 配置信息对象，包含有关数据集和模型的相关参数。
    返回：
    - train, dev, test (tuple): 包含三个元组，分别是训练集、验证集和测试集。
    """
    def load_dataset(path, pad_size=32):
        """
        加载并处理单个数据集文件。
        参数：
        - path (str): 数据集文件路径。
        - pad_size (int): 填充到的序列长度，默认为32。
        返回：
        - contents (list): 包含处理后的数据的列表。
        """
        contents = []  # 用于存储处理后的数据的列表
        with open(path, "r", encoding="UTF-8") as f:
            for line in tqdm(f):  # 逐行遍历文件内容
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split("\t")  # 分割每行的内容和标签
                token = config.tokenizer.tokenize(content)  # 使用分词器对内容进行分词
                token = [CLS] + token  # 在分词结果前加入[CLS]标记
                seq_len = len(token)  # 计算序列长度
                mask = []  # 用于存储填充掩码
                token_ids = config.tokenizer.convert_tokens_to_ids(token)  # 将分词结果转换为词汇索引
                if pad_size:
                    if len(token) < pad_size:  # 如果序列长度小于设定的填充长度
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))  # 创建填充掩码
                        token_ids += [0] * (pad_size - len(token))  # 对词汇索引列表进行填充
                    else:
                        mask = [1] * pad_size  # 如果序列长度大于等于填充长度，填充掩码为全1
                        token_ids = token_ids[:pad_size]  # 对词汇索引列表进行截断
                        seq_len = pad_size  # 更新序列长度为填充长度
                # 将处理后的数据添加到contents列表
                # print("(token_ids, int(label)", (token_ids, int(label)))
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    # 使用load_dataset函数加载训练集、验证集和测试集
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    # 返回三个元组，分别是训练集、验证集和测试集
    return train, dev, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, model_name):
        """
        数据集迭代器的初始化函数。
        参数：
        - batches (list): 包含样本的列表。
        - batch_size (int): 每个批次的大小。
        - device (str): 数据加载到的设备（CPU或GPU）。
        - model_name (str): 使用的模型名称。
        """
        self.batch_size = batch_size  # 每个批次的大小
        self.batches = batches  # 包含样本的列表
        self.model_name = model_name  #使用的模型名称
        self.n_batches = len(batches) // batch_size  #批次的数量
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0  # 当前批次的索引
        self.device = device  # 数据加载到的设备（CPU或GPU）

    def _to_tensor(self, datas):
        """
        将样本数据转换为Tensor。
        """

        # data : (token_ids, int(label), seq_len, mask)
        # 0 -> x
        # 1 -> y
        # 2 -> seq_len
        # 3 -> mask

        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # 若为BERT模型，返回(x, seq_len, mask), y；
        if self.model_name == "bert" :
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
            return (x, seq_len, mask), y
        # 若为TextCNN模型，返回(x, seq_len), y
        if self.model_name == "textCNN":
            return (x, seq_len), y

    def __next__(self):
        """
        获取下一个批次的样本。
        """
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        """
        返回迭代器对象本身。
        """
        return self

    def __len__(self):
        """
        获取迭代器的长度。
        """
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    """
    根据配置信息构建数据集迭代器。
    """
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.model_name)
    return iter


def get_time_dif(start_time):
    """
    计算已使用的时间差。
    """
    # 获取当前时间
    end_time = time.time()
    # 计算时间差
    time_dif = end_time - start_time
    # 将时间差转换为整数秒，并返回时间差对象
    return timedelta(seconds=int(round(time_dif)))