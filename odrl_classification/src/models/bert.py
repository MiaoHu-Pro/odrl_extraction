import torch
import torch.nn as nn
import os
from transformers import BertModel, BertTokenizer,BertConfig




class Config(object):
    def __init__(self):
        """
        配置类，包含模型和训练所需的各种参数。
        """
        self.model_name = "bert" # 模型名称
        self.data_path = "/home/mh1f25/scratch/full_score_code/04-bert/data/data2/" #数据集的根路径
        self.train_path = self.data_path + "train.txt"  # 训练集
        self.dev_path = self.data_path + "dev.txt"  # 验证集
        self.test_path = self.data_path + "test.txt"  # 测试集
        self.class_list = [x.strip() for x in open(self.data_path + "class.txt").readlines()]  # 类别名单

        self.save_path = "/home/mh1f25/scratch/full_score_code/04-bert/src/saved_dic" #模型训练结果保存路径
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.save_path += "/" + self.model_name + "ep10bz16.pt"  # 模型训练结果
        print("save path : ", self.save_path)
        self.save_path2 = "/home/mh1f25/scratch/full_score_code/04-bert/src/saved_dic1" # 量化模型存储结果路径
        if not os.path.exists(self.save_path2):
            os.mkdir(self.save_path2)
        self.save_path2 += "/" + self.model_name + "_quantized.pt"  # 量化模型存储结果

        # 模型训练+预测的时候, 放开下一行代码, 在GPU上运行.
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 训练设备，如果GPU可用，则为cuda，否则为cpu
        # 模型量化的时候, 放开下一行代码, 在CPU上运行.
        self.device = 'cpu'

        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 10  # epoch数
        self.batch_size = 16  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        # self.bert_path = "../data/bert_pretrain" # 预训练BERT模型的路径
        self.bert_path = "/home/mh1f25/scratch/llms_model/chinese-macbert-base" # 预训练BERT模型的路径
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path) # BERT模型的分词器
        self.bert_config = BertConfig.from_pretrained(self.bert_path + '/config.json') # BERT模型的配置
        self.hidden_size = 768 # BERT模型的隐藏层大小


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 预训练BERT模型
        self.bert = BertModel.from_pretrained(config.bert_path, config=config.bert_config)
        # 全连接层，用于文本分类
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x: 模型输入，包含句子、句子长度和填充掩码。
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        # _是占位符，接收模型的所有输出，而 pooled 是池化的结果,将整个句子的信息压缩成一个固定长度的向量
        _, pooled = self.bert(context, attention_mask=mask, return_dict=False)
        # 模型输出，用于文本分类
        out = self.fc(pooled)
        return out

