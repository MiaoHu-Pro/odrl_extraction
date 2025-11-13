import torch
from transformers import BertModel, BertTokenizer

model_path = '/home/mh1f25/scratch/llms_model/bert-base-cased'

# Load directly from your local directory
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)


def get_bert_encode_for_single(text: str):
    """
    Tokenize a single Chinese sentence and return its token embeddings.
    """
    indexed_token = tokenizer.encode(text)[1:-1]
    token_tensor = torch.LongTensor([indexed_token])

    with torch.no_grad():
        encode_layer = model(token_tensor)

    return encode_layer[0]


if __name__ == '__main__':
    text = "你好，周杰伦"
    embeddings = get_bert_encode_for_single(text)
    print(embeddings)  # tensor of shape (seq_len, hidden_size)
    print(embeddings.shape)  # e.g. torch.Size([7, 768])
