# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import BertModel
from model.crf import CRF


class BERT_LSTM_CRF(nn.Module):
    def __init__(
        self,
        bert_config,
        tagset_size,
        embedding_dim,          # kept for API compatibility, not used after removing IDCNN
        hidden_dim,
        rnn_layers,
        dropout_ratio,
        dropout1,
        use_cuda
    ):
        super(BERT_LSTM_CRF, self).__init__()
        # BERT backbone
        self.word_embeds = BertModel.from_pretrained(bert_config)
        bert_hidden = self.word_embeds.config.hidden_size  # e.g., 768 for bert-base

        # Model dims
        self.embedding_dim = bert_hidden
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.tagset_size = tagset_size
        self.use_cuda = use_cuda

        print("embedding_dim:", self.embedding_dim)
        print("hidden_dim:", self.hidden_dim)
        print("rnn_layers:", self.rnn_layers)
        print("dropout_ratio:", dropout_ratio)


        # BiLSTM over BERT embeddings
        self.lstm = nn.LSTM(
            input_size=bert_hidden,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout_ratio if rnn_layers > 1 else 0.0,
            batch_first=True
        )

        self.dropout1 = nn.Dropout(p=dropout1)

        # Emission projection (+2 for START/STOP as in your original)
        self.liner = nn.Linear(hidden_dim * 2, tagset_size + 2)

        # CRF
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)

    def rand_init_hidden(self, batch_size, device):
        # zeros are standard for LSTM init; shapes: (num_layers * num_directions, batch, hidden_dim)
        h0 = torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim, device=device)
        return h0, c0

    def get_output_score(self, sentence, attention_mask=None):
        """
        sentence: (batch, seq_len) input_ids
        attention_mask: (batch, seq_len) 1 for tokens, 0 for pads
        returns: emissions (batch, seq_len, tagset_size+2)
        """
        batch_size = sentence.size(0)
        device = sentence.device

        # BERT embeddings

        outputs = self.word_embeds(input_ids=sentence, attention_mask=attention_mask)
        embeds = outputs.last_hidden_state  # (batch, seq_len, bert_hidden)

        # LSTM over embeddings
        hidden = self.rand_init_hidden(batch_size, device)
        lstm_out, _ = self.lstm(embeds, hidden)  # (batch, seq_len, 2*hidden_dim)

        # Dropout + projection to tag emissions
        lstm_out = self.dropout1(lstm_out).contiguous()
        emissions = self.liner(lstm_out)  # (batch, seq_len, tagset_size+2)

        return emissions

    # def forward(self, sentence, masks):
    #     """
    #     Viterbi decode tags.
    #     sentence: input_ids  (batch, seq_len)
    #     masks:    attention_mask/CRF mask (batch, seq_len) with 1 for valid tokens
    #     """
    #     emissions = self.get_output_score(sentence, attention_mask=masks)
    #     scores, tag_seq = self.crf._viterbi_decode(emissions, masks.bool())
    #     return tag_seq

    def compute_emissions(self, input_ids, attention_mask):
        return self.get_output_score(input_ids, attention_mask=attention_mask)

    def forward(self, sentence, masks, return_emissions=False):
        if return_emissions:
            return type("Out", (), {"emissions": self.get_output_score(sentence, attention_mask=masks)})
        emissions = self.get_output_score(sentence, attention_mask=masks)
        _, tag_seq = self.crf._viterbi_decode(emissions, masks.bool())
        return tag_seq


    def neg_log_likelihood_loss(self, sentence, mask, tags):
        """
        mask: same mask used for attention + CRF (1 for tokens, 0 for pads)
        tags: (batch, seq_len) gold labels (no START/STOP)
        """
        emissions = self.get_output_score(sentence, attention_mask=mask)
        loss_value = self.crf.neg_log_likelihood_loss(emissions, mask, tags)
        batch_size = emissions.size(0)
        loss_value /= float(batch_size)
        return loss_value
