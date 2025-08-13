# config.py
import os
import torch
from transformers import AutoTokenizer, AutoConfig

# import your models
from model.bert_idcnn_crf import BERT_IDCNN_CRF
from model.bert_lstm_crf import BERT_LSTM_CRF
from model.deberta_crf import DeBERTa_CRF


class Config(object):
    """
    Central configuration + model factory.
    All parameters are initialized from `args`.
    """
    def __init__(self, args):
        # ---- Required / core ----
        self.flag = args.model                              # "bert-idcnn-crf" | "bert-lstm-crf"
        self.model_name = getattr(args, "model_name", self.flag)

        # Data
        self.data_path = args.data_path
        self.train_path = getattr(args, "train_path", os.path.join(self.data_path, "train.txt"))
        self.dev_path   = getattr(args, "dev_path",   os.path.join(self.data_path, "dev.txt"))
        self.test_path  = getattr(args, "test_path",  os.path.join(self.data_path, "test.txt"))

        # Saving
        self.save_dir   = args.save_dir
        self.quant_dir  = getattr(args, "quant_dir", os.path.join(self.save_dir, "quantized"))
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.quant_dir, exist_ok=True)

        self.num_epochs  = args.num_epochs
        self.batch_size  = args.batch_size
        # self.pad_size    = args.pad_size
        self.max_length = getattr(args, "max_length",
                                  getattr(args, "pad_size", 128))
        self.learning_rate = args.learning_rate

        # Device
        self.device = torch.device(getattr(args, "device", ("cuda" if torch.cuda.is_available() else "cpu")))
        self.use_cuda = (str(self.device).startswith("cuda"))

        # self.device = torch.device("cpu")
        # self.use_cuda = (str(self.device).startswith("cuda"))


        # ---- Backbone / tokenizer ----
        # Path or HF hub id for your pretrained encoder (e.g., "bert-base-cased", local dir, etc.)
        self.word_embedding_model_path = args.word_embedding_model_path
        print("self.word_embedding_model_path:", self.word_embedding_model_path)

        # Load tokenizer/config generically so you can swap models later (BERT/Roberta/DeBERTa/etc.)
        self.tokenizer = AutoTokenizer.from_pretrained(self.word_embedding_model_path, use_fast=True, add_prefix_space=True )
        self.hf_config = AutoConfig.from_pretrained(self.word_embedding_model_path)

        # Hidden size fallback for unusual models (T5 uses d_model)
        self.hidden_size = getattr(self.hf_config, "hidden_size", getattr(self.hf_config, "d_model", 768))

        # If you choose GPT-2 later: ensure there is a PAD token
        if self.tokenizer.pad_token is None:
            # Safe default: use eos_token as pad for decoder-only models like GPT-2
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Model hyperparams (from args) ----
        # IDCNN branch
        self.num_filters = getattr(args, "num_filters", 128)
        self.kernel_size = getattr(args, "kernel_size", 3)
        self.num_block = getattr(args, "num_block", 4)

        self.cnn_dropout = getattr(args, "cnn_dropout", 0.1)



        # LSTM branch
        self.lstm_hidden_dim = getattr(args, "hidden_dim", 768)
        self.rnn_layers      = getattr(args, "rnn_layers", 1)
        self.rnn_dropout     = getattr(args, "rnn_dropout", 0.1)   # LSTM internal dropout (between layers)
        self.proj_dropout    = getattr(args, "proj_dropout", 0.4)  # Dropout before linear head

        # ---- Labels / tag mapping ----
        # Your policy/contract BIO tag set
        base_labels = [
            "O",
            "B-PARTY_ASSIGNEE", "I-PARTY_ASSIGNEE",
            "B-PARTY_ASSIGNER", "I-PARTY_ASSIGNER",
            "B-ASSET", "I-ASSET",
            "B-ACTION", "I-ACTION",
            "B-RULE_TYPE", "I-RULE_TYPE",
            "B-CONSTRAINT", "I-CONSTRAINT",
            "B-DUTY", "I-DUTY",
            "B-PURPOSE", "I-PURPOSE",
        ]

        # Whether to include <pad>, <start>, <eos> in the label space as in your previous setup
        # If your CRF already adds start/stop internally, you can set this False to avoid double counting.
        self.include_special_labels = getattr(args, "include_special_labels", True)

        labels = list(base_labels)
        if self.include_special_labels:
            labels += ["<pad>", "<start>", "<eos>"]

        self.labels = labels
        self.l2i = {lbl: i for i, lbl in enumerate(self.labels)}
        self.i2l = {i: lbl for lbl, i in self.l2i.items()}

        print(self.i2l)
        print(self.l2i)

        self.tagset_size = len(self.l2i)

        # ---- Derived save paths ----

        if self.flag == "bert-idcnn-crf":
            self.save_path = os.path.join(self.save_dir,
                                          f"{self.model_name}_ep{self.num_epochs}_bz{self.batch_size}_max_len{self.max_length}_kernel{self.kernel_size}_num_block{self.num_block}")
        elif self.flag == "bert-lstm-crf":
            self.save_path = os.path.join(self.save_dir,
                                          f"{self.model_name}_ep{self.num_epochs}_bz{self.batch_size}_max_len{self.max_length}_num_layers{self.rnn_layers}")

        else:
            self.save_path = os.path.join(self.save_dir,
                                          f"{self.model_name}_ep{self.num_epochs}_bz{self.batch_size}_max_len{self.max_length}")



        # self.save_path  = os.path.join(self.save_dir,  f"{self.model_name}_ep{self.num_epochs}_bz{self.batch_size}_max_len{self.max_length}")
        self.quant_path = os.path.join(self.quant_dir, f"{self.model_name}_quantized.pt")

    # ------------------------------
    # Factory: build the selected model
    # ------------------------------
    def build_model(self):
        """
        Instantiates and returns the requested model on the configured device.
        Also resizes token embeddings if tokenizer size changed (e.g., added PAD for GPT-2).
        """
        if self.flag == "bert-idcnn-crf":
            model = BERT_IDCNN_CRF(
                vocab_size=self.tokenizer.vocab_size,         # kept for API compatibility
                tagset_size=self.tagset_size,
                embedding_dim=self.hidden_size,               # unused in BERT version; OK to pass
                num_filters=self.num_filters,
                max_length = self.max_length,
                kernel_size = self.kernel_size,
                num_block = self.num_block,
                dropout=self.cnn_dropout,
                use_cuda=self.use_cuda,
                bert_model_name_or_path=self.word_embedding_model_path,
            )
        elif self.flag == "bert-lstm-crf":
            model = BERT_LSTM_CRF(
                bert_config=self.word_embedding_model_path,   # path or HF id
                tagset_size=self.tagset_size,
                embedding_dim=self.hidden_size,               # kept for API compatibility
                hidden_dim=self.lstm_hidden_dim,
                rnn_layers=self.rnn_layers,
                dropout_ratio=self.rnn_dropout,               # LSTM internal dropout
                dropout1=self.proj_dropout,                   # dropout before linear head
                use_cuda=self.use_cuda,
            )

        elif self.flag == "deberta-crf":


            model = DeBERTa_CRF(
                deberta_model_name_or_path=self.word_embedding_model_path,
                tagset_size=self.tagset_size,
                use_cuda= self.use_cuda,
            )
        else:
            raise ValueError(f"Unknown model flag: {self.flag}. Use 'bert-idcnn-crf' or 'bert-lstm-crf'.")

        # Put on device
        model.to(self.device)

        # If special tokens changed vocab size, make sure the BERT embeddings match tokenizer size
        if hasattr(model, "bert"):
            try:
                current = model.bert.get_input_embeddings().num_embeddings
                needed  = len(self.tokenizer)
                if current != needed:
                    model.bert.resize_token_embeddings(needed)
            except Exception:
                pass

        return model
