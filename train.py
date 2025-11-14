# train.py
import os
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from hybrid_rules import apply_rules_to_predictions
from hybrid_bias import decode_with_rule_bias
from utils import recover_label, get_ner_fmeasure


def _to_device(x, device):
    return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x


def _device_from_cfg(cfg):
    if isinstance(cfg.device, torch.device):
        return cfg.device
    return torch.device(str(cfg.device))


def _split_param_groups(model, bert_lr, head_lr, weight_decay=0.01):
    """
    Small LR for BERT, larger LR for the rest.
    Supports both:
      - BERT_IDCNN_CRF (backbone attribute: model.bert)
      - BERT_LSTM_CRF  (backbone attribute: model.word_embeds)
    """
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")

    bert_decay, bert_nodecay, head_params = [], [], []
    backbone_prefixes = getattr(
        model,
        "backbone_param_prefixes",
        ("bert.", "word_embeds.", "deberta.", "encoder."),
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_backbone = any(name.startswith(prefix) for prefix in backbone_prefixes)
        if is_backbone:
            (bert_nodecay if any(nd in name for nd in no_decay) else bert_decay).append(param)
        else:
            head_params.append(param)

    groups = []
    if bert_decay:
        groups.append({"params": bert_decay, "lr": bert_lr, "weight_decay": weight_decay})
    if bert_nodecay:
        groups.append({"params": bert_nodecay, "lr": bert_lr, "weight_decay": 0.0})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": 0.0})
    return groups

#
@torch.no_grad()
def evaluate(model, cfg, dev_loader, dev_features):
    """
    Run Viterbi decoding on dev set and compute NER metrics.
    Returns: (acc, precision, recall, f1)
    """
    device = _device_from_cfg(cfg)
    model.eval()

    pred, gold = [], []
    for batch in dev_loader:
        sentence, masks, tags = batch
        sentence = _to_device(sentence, device)
        masks    = _to_device(masks, device)
        tags     = _to_device(tags, device)

        tag_seq = model(sentence, masks)  # may be Tensor or list[list[int]]
        if isinstance(tag_seq, torch.Tensor):
            tag_seq = tag_seq.tolist()

        pred.extend(tag_seq)
        gold.extend(tags.tolist())



    # map back to string labels & compute metrics
    pred_label, gold_label = recover_label(pred, gold, cfg.l2i, cfg.i2l)
    #
    # # using hybrid rule only
    # if dev_features is not None:
    #     pred_label = apply_rules_to_predictions(pred_label, dev_features)

    acc, p, r, f = get_ner_fmeasure(gold_label, pred_label, label_type='BIOES')


    model.train()
    return acc, p, r, f


def _sanitize_paths(pred_ids, label2id):
    C = len(label2id)
    o = int(label2id.get("O", 0))
    clean = []
    for seq in pred_ids:
        clean.append([(int(t) if int(t) < C else o) for t in seq])
    return clean

# using decode_with_rule_bias
# @torch.no_grad()
# def evaluate(model, cfg, dev_loader, dev_features):
#     """
#     Run Viterbi decoding on dev set and compute NER metrics.
#     Returns: (acc, precision, recall, f1)
#     """
#     device = _device_from_cfg(cfg)
#     model.eval()
#
#     gold = []
#     all_pred_ids = []
#     offset = 0
#     for batch in dev_loader:
#         bsz = batch[0].size(0)  # input_ids
#         batch_features = dev_features[offset: offset + bsz]
#         offset += bsz
#
#         sentence, masks, tags = batch
#         # >>> Rule-biased decode (adds logits, then Viterbi via CRF) <<<
#         pred_ids = decode_with_rule_bias(
#             model, batch, batch_features,
#             device=device,
#             label2id=cfg.l2i,  # your BIO_HEAD id map
#             alpha_pos=3.0  # bias strength; try 2.0–4.0
#         )
#
#         pred_ids = _sanitize_paths(pred_ids, cfg.l2i)  # <<< add this line
#
#         all_pred_ids.extend(pred_ids)
#
#         gold.extend(tags.tolist())
#
#
#
#     # map back to string labels & compute metrics
#     pred_label, gold_label = recover_label(all_pred_ids, gold, cfg.l2i, cfg.i2l)
#
#     acc, p, r, f = get_ner_fmeasure(gold_label, pred_label, label_type='BIO')
#
#     model.train()
#     return acc, p, r, f

def run_train(model, cfg, train_dataloader, dev_dataloader, dev_features):
    """
    Train loop with:
      - AdamW param groups (BERT vs. heads)
      - Optional linear warmup scheduler (cfg.use_scheduler)
      - AMP (automatic mixed precision) if CUDA
      - Gradient clipping
      - Saves best_f1.pth and last_model.pth
    """
    device = _device_from_cfg(cfg)

    print ("device: ", device)

    model.to(device)
    model.train()

    # --- hyperparams from cfg (with sensible defaults)
    bert_lr       = getattr(cfg, "bert_lr", getattr(cfg, "learning_rate", 2e-5))
    head_lr       = getattr(cfg, "head_lr", 1e-3)
    weight_decay  = getattr(cfg, "weight_decay", 0.01)
    max_grad_norm = getattr(cfg, "max_grad_norm", 1.0)
    use_scheduler = getattr(cfg, "use_scheduler", False)
    warmup_ratio  = getattr(cfg, "warmup_ratio", 0.1)
    num_epochs    = getattr(cfg, "num_epochs", 5)

    # --- optimizer & (optional) scheduler
    param_groups = _split_param_groups(model, bert_lr=bert_lr, head_lr=head_lr, weight_decay=weight_decay)
    optimizer = AdamW(param_groups)

    if use_scheduler:
        total_steps  = len(train_dataloader) * num_epochs
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        scheduler = None

    # --- AMP
    use_amp = torch.cuda.is_available() and str(device).startswith("cuda") and getattr(cfg, "use_amp", True)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # --- save locations
    # Prefer a directory; if cfg.save_path is a file, we drop into its parent dir.
    save_dir = getattr(cfg, "save_dir", None)
    if not save_dir:
        sp = getattr(cfg, "save_path", "./checkpoints")
        save_dir = sp if os.path.isdir(sp) else (os.path.dirname(sp) or "./checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    best_path = cfg.save_path + '_best_f1.pth'
    last_path = cfg.save_path + '_last_model.pth'

    # best_path = os.path.join(cfg.save_path, "_best_f1.pth")
    # last_path = os.path.join(cfg.save_path, "_last_model.pth")

    print(f"[train] device={device} | epochs={num_epochs} | use_scheduler={use_scheduler} | AMP={use_amp}")
    print(f"[train] save_dir={save_dir}\n       best -> {best_path}\n       last -> {last_path}")

    best_f1 = -1.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}\n", leave=False)

        for step, batch in enumerate(pbar, 1):
            sentence, masks, tags = batch
            sentence = _to_device(sentence, device)
            masks    = _to_device(masks, device)
            tags     = _to_device(tags, device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:

                # with torch.cuda.amp.autocast():
                with torch.amp.autocast('cuda'):
                    loss = model.neg_log_likelihood_loss(sentence, masks, tags)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.neg_log_likelihood_loss(sentence, masks, tags)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(1, len(train_dataloader))
        print(f"[epoch {epoch+1}] train loss: {avg_loss:.6f}")

        # --- dev eval
        acc, p, r, f = evaluate(model, cfg, dev_dataloader, dev_features)
        print(f"[epoch {epoch+1}] dev  acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f:.4f}")

        # --- save best & last
        if f > best_f1:
            torch.save(model.state_dict(), best_path)
            best_f1 = f
            print(f"  ↳ new best F1={best_f1:.4f} (saved to {best_path})")

        torch.save(model.state_dict(), last_path)

    print(f"Training finished. Best F1={best_f1:.4f}.")
