import torch
import torch.nn as nn
from transformers import BertModel
from model.crf import CRF
from model.idcnn import IDCNN


class BERT_IDCNN_CRF(nn.Module):
    def __init__(
        self,
        vocab_size,                 # kept for API compatibility (unused)
        tagset_size,
        embedding_dim,              # kept for API compatibility (unused)
            kernel_size,
            num_block,
        num_filters,
        max_length,
        dropout=0.4,
        use_cuda=True,
        bert_model_name_or_path="bert-base-uncased"  # change to your model, e.g. "bert-base-chinese"
    ):
        super(BERT_IDCNN_CRF, self).__init__()
        # Core dims / flags
        self.tagset_size = tagset_size
        self.num_filters = num_filters
        self.use_cuda = use_cuda

        # 1) BERT as contextual embedding
        self.bert = BertModel.from_pretrained(bert_model_name_or_path)
        bert_hidden = self.bert.config.hidden_size  # e.g., 768

        # 2) IDCNN over BERT embeddings
        self.idcnn = IDCNN(input_size=bert_hidden, filters=num_filters, num_tokens=max_length,
                           kernel_size=kernel_size, num_block=num_block)

        # 3) Dropout
        self.dropout = nn.Dropout(p=dropout)

        # 4) CRF layer
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)

        # 5) Projection to CRF emissions (+2 for START/STOP as in your original)
        self.linear = nn.Linear(num_filters, tagset_size + 2)

    def get_output_score(self, sentence, attention_mask=None):
        """
        sentence: input_ids for BERT  (batch, seq_len)
        attention_mask: 1 for valid tokens, 0 for padding (batch, seq_len)
        returns: emissions (batch, seq_len, tagset_size+2)
        """
        batch_size, seq_length = sentence.size(0), sentence.size(1)

        # 1) BERT embeddings
        outputs = self.bert(input_ids=sentence, attention_mask=attention_mask)
        embeds = outputs.last_hidden_state                 # (batch, seq_len, bert_hidden)

        # 2) IDCNN
        idcnn_out = self.idcnn(embeds, seq_length)         # (batch, seq_len, num_filters)

        # 3) Dropout
        idcnn_out = self.dropout(idcnn_out)

        # 4) Linear projection to emissions
        out = self.linear(idcnn_out)                       # (batch, seq_len, tagset_size+2)

        # 5) Keep (batch, seq_len, -1) shape
        feats = out.contiguous().view(batch_size, seq_length, -1)
        return feats

    def forward(self, sentence, masks):
        """
        Inference: Viterbi decode.
        `masks` should be 1 for real tokens (and typically 0 for padding and
        for non-first wordpieces if you only label the first subword).
        """
        feats = self.get_output_score(sentence, attention_mask=masks)
        scores, tag_seq = self.crf._viterbi_decode(feats, masks.bool())
        return tag_seq

    def neg_log_likelihood_loss(self, sentence, mask, tags):
        """
        Training: negative log-likelihood from CRF.
        `mask` is the same attention/CRF mask used in forward.
        """
        feats = self.get_output_score(sentence, attention_mask=mask)
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value
