# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import DebertaModel
from model.crf import CRF  # same CRF as your BERT-LSTM-CRF

class DeBERTa_CRF(nn.Module):
    def __init__(
        self,
        deberta_model_name_or_path,
        tagset_size,
        dropout_rate=0.1,
        use_cuda=True
    ):
        super(DeBERTa_CRF, self).__init__()


        print("deberta_model_name_or_path:", deberta_model_name_or_path)
        print("tagset_size:", tagset_size)
        print("dropout_rate:", dropout_rate)
        print("use_cuda:", use_cuda)

        # Load DeBERTa backbone
        self.deberta = DebertaModel.from_pretrained(deberta_model_name_or_path, ignore_mismatched_sizes=True)
        self.backbone_param_prefixes = ("deberta.",)
        hidden_size = self.deberta.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, tagset_size + 2)  # +2 if your CRF needs START/STOP

        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)

    def get_output_score(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)  # (batch, seq_len, tagset_size + 2)
        return emissions

    def forward(self, input_ids, attention_mask):
        emissions = self.get_output_score(input_ids, attention_mask)
        _, tag_seq = self.crf._viterbi_decode(emissions, attention_mask.bool())
        return tag_seq

    def neg_log_likelihood_loss(self, input_ids, attention_mask, tags):
        emissions = self.get_output_score(input_ids, attention_mask)
        loss = self.crf.neg_log_likelihood_loss(emissions, attention_mask, tags)
        return loss / float(input_ids.size(0))
