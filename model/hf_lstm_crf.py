import torch
import torch.nn as nn
from transformers import AutoModel

from model.crf import CRF


class HF_LSTM_CRF(nn.Module):
    """
    Generic Transformer encoder + BiLSTM + CRF stack.

    Works with any HuggingFace AutoModel checkpoint (encoder-only,
    decoder-only, or encoder-decoder). The transformer backbone is
    treated purely as a contextual embedding layer; decoding is handled
    via the BiLSTM+CRF stack.
    """

    def __init__(
        self,
        encoder_name_or_path,
        tagset_size,
        hidden_dim,
        rnn_layers,
        dropout_ratio,
        proj_dropout,
        use_cuda,
        trust_remote_code=False,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            encoder_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.backbone_param_prefixes = ("encoder.",)

        encoder_hidden = getattr(
            self.encoder.config,
            "hidden_size",
            getattr(self.encoder.config, "d_model", None),
        )
        if encoder_hidden is None:
            raise ValueError(
                f"Could not determine hidden size for {encoder_name_or_path}. "
                "Expected `hidden_size` or `d_model` in config."
            )

        self.tagset_size = tagset_size
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.use_cuda = use_cuda

        self.lstm = nn.LSTM(
            input_size=encoder_hidden,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout_ratio if rnn_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=proj_dropout)
        self.linear = nn.Linear(hidden_dim * 2, tagset_size + 2)
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)

    @property
    def _is_encoder_decoder(self):
        return bool(getattr(self.encoder.config, "is_encoder_decoder", False))

    def _encode(self, input_ids, attention_mask=None):
        if self._is_encoder_decoder:
            encoder = getattr(self.encoder, "encoder", None)
            if encoder is None:
                raise AttributeError(
                    "Encoder-decoder model is missing `.encoder`. "
                    "Please load a supported architecture."
                )
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return outputs.last_hidden_state

    def rand_init_hidden(self, batch_size, device):
        h0 = torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(2 * self.rnn_layers, batch_size, self.hidden_dim, device=device)
        return h0, c0

    def get_output_score(self, sentence, attention_mask=None):
        batch_size = sentence.size(0)
        device = sentence.device

        embeddings = self._encode(sentence, attention_mask=attention_mask)
        hidden = self.rand_init_hidden(batch_size, device)
        lstm_out, _ = self.lstm(embeddings, hidden)
        lstm_out = self.dropout(lstm_out).contiguous()
        emissions = self.linear(lstm_out)
        return emissions

    def compute_emissions(self, input_ids, attention_mask):
        return self.get_output_score(input_ids, attention_mask=attention_mask)

    def forward(self, sentence, masks, return_emissions=False):
        emissions = self.get_output_score(sentence, attention_mask=masks)
        if return_emissions:
            return type("Out", (), {"emissions": emissions})
        _, tag_seq = self.crf._viterbi_decode(emissions, masks.bool())
        return tag_seq

    def neg_log_likelihood_loss(self, sentence, mask, tags):
        emissions = self.get_output_score(sentence, attention_mask=mask)
        loss_value = self.crf.neg_log_likelihood_loss(emissions, mask, tags)
        return loss_value / float(emissions.size(0))
