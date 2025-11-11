# predict.py
import os
from typing import Optional, List, Tuple
# from hybrid_rules import apply_rules_to_predictions
import torch
import pandas as pd
from tqdm import tqdm

# adjust imports to where you defined them
from utils import recover_label, get_ner_fmeasure, InputFeatures
from utils import compute_entity_metrics_by_type, compute_token_metrics_by_tag

def _device_from_cfg(cfg):
    if isinstance(cfg.device, torch.device):
        return cfg.device
    return torch.device(str(cfg.device))


@torch.no_grad()
def run_test(
    model,
    cfg,
    test_dataloader,
    test_features: Optional[List[InputFeatures]] = None,
    out_path: Optional[str] = None,
    return_details: bool = False,
):
    """
    Runs Viterbi decoding on test_dataloader, computes BIO metrics,
    and writes predict_result.txt with: text-->true_label-->...-->predict_label-->...

    Args:
        model: trained model with .forward(input_ids, masks)
        cfg:   config with .device, .l2i, .i2l and (optionally) .save_dir
        test_dataloader: DataLoader yielding (input_ids, masks, tags)
        test_features: optional list[InputFeatures] aligned with dataloader order
        out_path: optional path for predict_result.txt (default: cfg.save_dir/predict_result.txt)
        return_details: if True, also return (metrics_df, details_df)

    Returns:
        metrics_df (and details_df if return_details=True)
    """
    device = _device_from_cfg(cfg)
    model.eval()
    model.to(device)

    preds, golds = [], []

    for batch in tqdm(test_dataloader, desc="Testing", leave=False):
        input_ids, masks, tags = batch
        input_ids = input_ids.to(device)
        masks     = masks.to(device)
        tags      = tags.to(device)

        tag_seq = model(input_ids, masks)  # may be Tensor or list[list[int]]
        if isinstance(tag_seq, torch.Tensor):
            tag_seq = tag_seq.tolist()

        preds.extend(tag_seq)
        golds.extend(tags.tolist())

    # Map back to string labels (cuts to <start>...<eos>)
    pred_label, gold_label = recover_label(preds, golds, cfg.l2i, cfg.i2l)

    # # using hybrid rules only
    # if test_features is not None:
    #     pred_label = apply_rules_to_predictions(pred_label, test_features)


    base_labels = [
        "O",
        "B-PARTY_ASSIGNEE", "I-PARTY_ASSIGNEE",
        # "B-PARTY_ASSIGNER", "I-PARTY_ASSIGNER",
        "B-ASSET", "I-ASSET",
        "B-ACTION", "I-ACTION",
        "B-RULE_TYPE", "I-RULE_TYPE",
        "B-CONSTRAINT", "I-CONSTRAINT",
        # "B-DUTY", "I-DUTY",
        "B-PURPOSE", "I-PURPOSE",
        # "B-CONSEQUENCE", "I-CONSEQUENCE"
    ]

    # pred_label, gold_label = recover_label(...)

    per_type_df = compute_entity_metrics_by_type(gold_label, pred_label)  # ASSET/ACTION/...
    per_tag_df = compute_token_metrics_by_tag(gold_label, pred_label, base_labels)  # B-*/I-*/O

    print(per_type_df)
    print(per_tag_df)

    # optionally save
    per_type_df.to_csv(cfg.save_path + "_metrics_per_type.txt", index=True, sep=',', mode='a')
    per_tag_df.to_csv(cfg.save_path + "_metrics_per_tag.txt", index=True, sep=',', mode='a')


    # Metrics (BIO)
    acc, p, r, f = get_ner_fmeasure(gold_label, pred_label, label_type="BIO")
    metrics_df = pd.DataFrame([{
        "num_sentences": len(gold_label),
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f,
    }])

    # Prepare output path
    save_dir = getattr(cfg, "save_dir", None) or os.path.dirname(getattr(cfg, "save_path", "")) or "."
    os.makedirs(save_dir, exist_ok=True)
    out_path = cfg.save_path + "_predict_result.txt"


    # Build per-sample lines
    lines = []
    for i, (gold_seq, pred_seq) in enumerate(zip(gold_label, pred_label)):
        # Try to recover original text from features
        text_str = ""
        if test_features is not None and i < len(test_features):
            feat = test_features[i]
            # feat.text may be a list of tokens (often with [CLS]/[SEP]) or a raw string
            if isinstance(feat.text, list):
                toks = [t for t in feat.text if t not in ("[CLS]", "[SEP]")]
                # Align length defensively
                if len(toks) != len(gold_seq):
                    toks = toks[:len(gold_seq)]
                text_str = " ".join(toks)
            else:
                text_str = str(feat.text)
        # Fallback: join labels’ positions as text if no feature text
        if not text_str:
            text_str = f"<sample_{i}>"

        line = f"{text_str}-->true_label--> {' '.join(gold_seq)}-->predict_label--> {' '.join(pred_seq)}"
        lines.append(line)

    # Write file
    with open(out_path, "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(lines))

    # Optionally return a details DataFrame
    if return_details:
        details_df = pd.DataFrame({
            "text": [l.split('-->true_label-->')[0] for l in lines],
            "gold_labels": [" ".join(seq) for seq in gold_label],
            "pred_labels": [" ".join(seq) for seq in pred_label],
        })
        return metrics_df, details_df

    return metrics_df
