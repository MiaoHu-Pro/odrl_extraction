# -*- coding: utf-8 -*-
import logging
import os
import argparse
from model.config import Config
import torch
from utils import print_and_save_params, _words_from_text
from predict import run_test
from flask import Flask, render_template, request, jsonify, make_response
from hybrid_rules import postprocess_tags

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)  # Only show errors, not warnings

app = Flask(__name__)


import torch

def get_input(config, sample):
    """
    Build a single-sample batch (input_ids, attention_mask) for inference.
    Mirrors _encode_sample's "keep first subword" policy, but no labels needed.
    Returns:
        input_ids: LongTensor [1, max_length]
        masks:     LongTensor [1, max_length]
        words:     List[str]  (the truncated word-level tokens used)
    """
    tokenizer  = config.tokenizer
    max_length = config.max_length

    # Special token ids with safe fallbacks
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else (
        tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 101
    )
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else (
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 102
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Word-level tokens (your own helper that splits words/punct like in training)
    words = _words_from_text(sample, tokenizer)

    # keep room for [CLS] and [SEP]
    max_words = max_length - 2
    words = words[:max_words]

    # Build ids/masks (keep only the FIRST subword for each word)
    input_ids = [cls_id]
    masks     = [1]

    for w in words:
        sub_tokens = tokenizer.tokenize(w)
        if not sub_tokens:
            sub_tokens = [tokenizer.unk_token or "[UNK]"]
        first_piece = sub_tokens[0]
        tok_id = tokenizer.convert_tokens_to_ids(first_piece)
        if tok_id is None:
            tok_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 100
        input_ids.append(tok_id)
        masks.append(1)

    # [SEP]
    input_ids.append(sep_id)
    masks.append(1)

    # pad to max_length
    if len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids.extend([pad_id] * pad_len)
        masks.extend([0] * pad_len)
    else:
        input_ids = input_ids[:max_length]
        masks     = masks[:max_length]

    # to tensors (batch=1)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    masks     = torch.tensor([masks],     dtype=torch.long)

    return input_ids, masks, words


def model2predict(model, config, sample, apply_post_rules=True):
    """
    Run the model on a single sentence and return:
        tokens_used (word-level, truncated to fit),
        pred_labels (BIO tags aligned 1:1 to tokens_used)
    """
    model.eval()
    device = config.device
    # device = next(model.parameters()).device  # robust way to get current device

    # Build inputs
    input_ids, masks, words = get_input(config, sample)
    input_ids = input_ids.to(device)
    masks     = masks.to(device)

    with torch.no_grad():
        # model returns Viterbi-decoded tag ids per token position
        tag_seq = model(input_ids, masks)  # (batch, seq_len) as list or tensor

    if isinstance(tag_seq, torch.Tensor):
        tag_seq = tag_seq.tolist()

    pred_ids = tag_seq[0]                        # first (and only) sample
    valid_len = int(masks[0].sum().item())       # includes [CLS] and [SEP]
    pred_ids = pred_ids[:valid_len]              # cut any right padding
    pred_ids = pred_ids[1:-1]                    # drop [CLS] and [SEP]

    # align to the word count we actually fed (safety if anything drifted)
    word_count = len(words)
    pred_ids   = pred_ids[:word_count]

    # map ids -> string labels
    i2l = config.i2l
    pred_labels = [i2l.get(tid, "O") for tid in pred_ids]


    result = {"text": words, "labels": pred_labels}

    return result



def query_test(cfg, text):
    print('do testing ...')
    logger.info('do testing ...')

    print('device: {}'.format(cfg.device))

    model = cfg.build_model()

    best_model_path = cfg.save_path + '_best_f1.pth'
    last_model_path = cfg.save_path + '_last_model.pth'

    # Check if best_f1.pth exists, otherwise load last_model.pth
    if os.path.isfile(best_model_path):
        print(f"Loading best model from {best_model_path}")
        logger.info(f"Loading best model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=cfg.device, weights_only=True))
    elif os.path.isfile(last_model_path):
        print(f"Loading last model from {last_model_path}")
        logger.info(f"Loading last model from {last_model_path}")
        model.load_state_dict(torch.load(last_model_path, map_location=cfg.device, weights_only=True))
    else:
        raise FileNotFoundError(f"Neither {best_model_path} nor {last_model_path} exists.")

    # 模型训练
    result = model2predict(model, cfg, text)
    logger.info('------------------test result------------------')
    logger.info(result)

    # 结果输出
    print(f"text : {result['text']}")
    print(f"label : {result['labels']}")
    return result


def fun(text):

    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_path", required=True, default="./data/")
    parser.add_argument("--save_dir", required=True, default="./save_model/")
    parser.add_argument("--word_embedding_model_path", required=True)  # e.g., "/home/.../bert-base-cased"
    parser.add_argument("--label_type", choices=["BIO", "BIOES"], default="BIO",
                        help="Tagging scheme to use for labels.")
    parser.add_argument("--trust_remote_code", action="store_true")

    # Training settings
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    # parser.add_argument("--pad_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda")

    # IDCNN hyperparams
    parser.add_argument("--num_filters", type=int, default=64)
    parser.add_argument("--cnn_dropout", type=float, default=0.4)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--num_block", type=int, default=4)



    # LSTM hyperparams
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--rnn_dropout", type=float, default=0.1)
    parser.add_argument("--proj_dropout", type=float, default=0.4)


    parser.add_argument("--max_length", type=int, default=400)

    # Labels handling
    parser.add_argument("--include_special_labels", action="store_true", default=True)


    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")


    args = parser.parse_args()

    cfg = Config(args)


    return query_test(cfg, text)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    question = None
    if request.method == 'POST':
        question = request.form['text']
        result = fun(question)
    return render_template('index.html', question=question, result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    # python app.py --model bert-lstm-crf  --data_path ./data/ --save_dir ./save_model --word_embedding_model_path /home/mh1f25/scratch/llms_model/bert-base-cased   --do_test  --num_epochs 10

    # python app.py --model contract-bert-lstm-crf  --data_path ./data/ --save_dir ./save_model --word_embedding_model_path /home/mh1f25/scratch/llms_model/bert-base-uncased-contracts  --do_test  --num_epochs 20 --rnn_layers 1 --max_length 400 --label_type BIO

