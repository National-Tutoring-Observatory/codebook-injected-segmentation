import os
# MUST be set before importing torch
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, set_seed

from model import SegModel


def depth_score_cal(scores):
    output_scores = []
    for i in range(len(scores)):
        lflag, rflag = scores[i], scores[i]
        if i == 0:
            for r in range(i + 1, len(scores)):
                if rflag <= scores[r]:
                    rflag = scores[r]
                else:
                    break
        elif i == len(scores) - 1:
            for l in range(i - 1, -1, -1):
                if lflag <= scores[l]:
                    lflag = scores[l]
                else:
                    break
        else:
            for r in range(i + 1, len(scores)):
                if rflag <= scores[r]:
                    rflag = scores[r]
                else:
                    break
            for l in range(i - 1, -1, -1):
                if lflag <= scores[l]:
                    lflag = scores[l]
                else:
                    break
        depth_score = 0.5 * (lflag + rflag - 2 * scores[i])
        output_scores.append(depth_score)
    return output_scores


def apply_min_gap(boundary_indices, min_gap):
    if not boundary_indices:
        return []
    filtered = [boundary_indices[0]]
    for idx in boundary_indices[1:]:
        if idx - filtered[-1] >= min_gap:
            filtered.append(idx)
    return filtered


def to_python(obj):
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def predict_for_folder(
    data_dir,
    model_path,
    device,
    window_size=2,
    pick_num=4,
    avg_seg_len=None,
    alpha=0.5,
    min_gap=3,
    seed=3407,
    batch_size=4,
    use_amp=True,
    amp_dtype="fp16",
):
    set_seed(seed)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # Load model on CPU then move to GPU
    model = SegModel()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    model.to(device)
    model.eval()

    input_files = [
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f)) and f != ".DS_Store"
    ]

    all_results = {}

    amp_ok = (use_amp and device.type == "cuda")
    if amp_dtype == "bf16":
        amp_torch_dtype = torch.bfloat16
    else:
        amp_torch_dtype = torch.float16

    for file in tqdm(input_files, desc="Predicting"):
        path = os.path.join(data_dir, file)
        text = []

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "=======" in line:
                        continue
                    text.append(line)
        except Exception as e:
            print(f"[WARN] Failed to read file {file}: {e}")
            all_results[file] = {"error": f"file_read_error: {e}"}
            continue

        if len(text) < 2:
            all_results[file] = {
                "utterances": text,
                "boundary_indices": [],
                "segment_lengths": [len(text)] if text else [],
                "scores": {"raw_scores": [], "depth_scores": []},
            }
            continue

        # For each potential boundary i (between i and i+1):
        # - coherence inputs: one row per i
        # - topic inputs: concatenated rows; topic_num tells how many rows per i
        id_inputs = []
        type_ids = []
        coheren_att_masks = []
        topic_input_rows = [[], []]       # list of 1D tensors (token ids)
        topic_mask_rows = [[], []]        # list of 1D tensors (attn mask)
        topic_num = [[], []]              # per-boundary counts (len N)

        try:
            for i in range(len(text) - 1):
                context, cur = [], []
                l, r = i, i + 1

                for _ in range(window_size):
                    if l > -1:
                        context.append(text[l][:128])
                        l -= 1
                    if r < len(text):
                        cur.append(text[r][:128])
                        r += 1
                context.reverse()

                # topic encoder inputs
                topic_con = tokenizer(
                    context, truncation=True, padding=True, max_length=256, return_tensors="pt"
                )
                topic_cur = tokenizer(
                    cur, truncation=True, padding=True, max_length=256, return_tensors="pt"
                )

                # NOTE: these are concatenated across all i
                topic_input_rows[0].extend(list(topic_con["input_ids"]))
                topic_input_rows[1].extend(list(topic_cur["input_ids"]))
                topic_mask_rows[0].extend(list(topic_con["attention_mask"]))
                topic_mask_rows[1].extend(list(topic_cur["attention_mask"]))
                topic_num[0].append(len(context))
                topic_num[1].append(len(cur))

                # coherence (NSP-style pair)
                sent1 = "".join([sen + "[SEP]" for sen in context])
                sent2 = text[i + 1]

                enc1 = tokenizer.encode(sent1, add_special_tokens=True, max_length=256, truncation=True)
                enc2 = tokenizer.encode(sent2, add_special_tokens=True, max_length=256, truncation=True)

                if len(enc1) == 0 or len(enc2) == 0:
                    continue

                pair_ids = enc1[:-1] + enc2[1:]
                pair_types = [0] * len(enc1[:-1]) + [1] * len(enc2[1:])

                id_inputs.append(pair_ids)
                type_ids.append(pair_types)

        except Exception as e:
            print(f"[WARN] Error building inputs for file {file}: {e}")
            all_results[file] = {"utterances": text, "error": f"build_inputs_error: {e}"}
            continue

        if len(id_inputs) == 0:
            all_results[file] = {"utterances": text, "error": "no_valid_pairs"}
            continue

        # Pad coherence inputs ON CPU (numpy arrays)
        MAX_LEN = 512
        id_inputs_np = pad_sequences(
            id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post"
        )
        type_ids_np = pad_sequences(
            type_ids, maxlen=MAX_LEN, dtype="long", value=1, truncating="post", padding="post"
        )

        # attention mask on CPU
        for sent in id_inputs_np:
            coheren_att_masks.append([int(tok > 0) for tok in sent])

        # Pad topic inputs ON CPU (torch tensors)
        # These tensors are big across the whole dialogue -> keep on CPU, slice per batch
        try:
            topic_input_cpu = [
                pad_sequence(topic_input_rows[0], batch_first=True),
                pad_sequence(topic_input_rows[1], batch_first=True),
            ]
            topic_mask_cpu = [
                pad_sequence(topic_mask_rows[0], batch_first=True),
                pad_sequence(topic_mask_rows[1], batch_first=True),
            ]
        except Exception as e:
            print(f"[WARN] Topic padding error for file {file}: {e}")
            all_results[file] = {"utterances": text, "error": f"topic_padding_error: {e}"}
            continue

        # Build prefix offsets so we can slice the concatenated topic rows per boundary batch
        # offsets[0] = 0, offsets[i+1] = sum(topic_num[:i+1])
        topic_offsets0 = np.zeros(len(topic_num[0]) + 1, dtype=np.int64)
        topic_offsets1 = np.zeros(len(topic_num[1]) + 1, dtype=np.int64)
        topic_offsets0[1:] = np.cumsum(np.array(topic_num[0], dtype=np.int64))
        topic_offsets1[1:] = np.cumsum(np.array(topic_num[1], dtype=np.int64))

        # TRUE batched inference: only move [s:e] to GPU each time
        scores_all = []
        N = id_inputs_np.shape[0]

        try:
            with torch.inference_mode():
                for s in range(0, N, batch_size):
                    e = min(s + batch_size, N)

                    # coherence batch -> GPU
                    coheren_inputs = torch.as_tensor(id_inputs_np[s:e], dtype=torch.long, device=device)
                    coheren_type_ids = torch.as_tensor(type_ids_np[s:e], dtype=torch.long, device=device)
                    coheren_masks = torch.as_tensor(coheren_att_masks[s:e], dtype=torch.long, device=device)

                    # topic slice ranges in the concatenated tensors
                    c0_s, c0_e = int(topic_offsets0[s]), int(topic_offsets0[e])
                    c1_s, c1_e = int(topic_offsets1[s]), int(topic_offsets1[e])

                    topic_input_b = [
                        topic_input_cpu[0][c0_s:c0_e].to(device),
                        topic_input_cpu[1][c1_s:c1_e].to(device),
                    ]
                    topic_mask_b = [
                        topic_mask_cpu[0][c0_s:c0_e].to(device),
                        topic_mask_cpu[1][c1_s:c1_e].to(device),
                    ]
                    topic_num_b = [topic_num[0][s:e], topic_num[1][s:e]]

                    if amp_ok:
                        with torch.autocast(device_type="cuda", dtype=amp_torch_dtype):
                            out = model.infer(
                                coheren_inputs, coheren_masks, coheren_type_ids,
                                topic_input_b, topic_mask_b, topic_num_b
                            )
                    else:
                        out = model.infer(
                            coheren_inputs, coheren_masks, coheren_type_ids,
                            topic_input_b, topic_mask_b, topic_num_b
                        )

                    # move outputs to CPU plain floats
                    if torch.is_tensor(out):
                        out = out.detach().float().cpu().tolist()
                    else:
                        out = [float(x) for x in out]
                    scores_all.extend(out)

                    # free per-batch GPU tensors
                    del coheren_inputs, coheren_masks, coheren_type_ids, topic_input_b, topic_mask_b, out
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        except Exception as e:
            print(f"[WARN] Model inference error for file {file}: {e}")
            all_results[file] = {"utterances": text, "error": f"inference_error: {e}"}
            continue

        # --- Depth scores & boundary selection ---
        depth_scores = depth_score_cal(scores_all)
        depth_arr = np.array(depth_scores, dtype=np.float32)

        num_possible = len(depth_arr)
        if avg_seg_len is not None and avg_seg_len > 0:
            est_segments = max(1, int(round(len(text) / avg_seg_len)))
            max_boundaries = max(0, min(est_segments - 1, num_possible))
        else:
            max_boundaries = min(pick_num, num_possible)

        if num_possible == 0 or max_boundaries == 0:
            boundary_indices, mu, sigma = [], 0.0, 0.0
        else:
            mu = float(depth_arr.mean())
            sigma = float(depth_arr.std())

            if sigma == 0:
                candidate_indices = np.argsort(depth_arr)[-max_boundaries:] if max_boundaries > 0 else []
            else:
                thr = mu + alpha * sigma
                candidate_indices = np.where(depth_arr >= thr)[0]

                if len(candidate_indices) > max_boundaries > 0:
                    candidate_indices = sorted(candidate_indices, key=lambda i: depth_arr[i], reverse=True)[:max_boundaries]
                elif len(candidate_indices) < max_boundaries and max_boundaries > 0:
                    topk = np.argsort(depth_arr)[-max_boundaries:]
                    candidate_indices = np.unique(np.concatenate([candidate_indices, topk]))
                    candidate_indices = sorted(candidate_indices, key=lambda i: depth_arr[i], reverse=True)[:max_boundaries]

            candidate_indices = sorted(list(candidate_indices))
            boundary_indices = apply_min_gap(candidate_indices, min_gap=min_gap)

        # Convert boundary indices to segment lengths
        seg_p_labels = [0] * (len(depth_scores) + 1)
        for i in boundary_indices:
            if 0 <= i < len(seg_p_labels):
                seg_p_labels[i] = 1

        seg_lengths = []
        tmp = 0
        for b in seg_p_labels:
            tmp += 1
            if b == 1:
                seg_lengths.append(tmp)
                tmp = 0
        seg_lengths.append(tmp)

        all_results[file] = {
            "utterances": text,
            "boundary_indices": boundary_indices,
            "segment_lengths": seg_lengths,
            "scores": {
                "raw_scores": [float(x) for x in scores_all],
                "depth_scores": [float(x) for x in depth_scores],
                "mu": float(mu),
                "sigma": float(sigma),
            },
        }

        # extra cleanup between files
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1, help="CUDA device id (use 1 on your machine)")
    parser.add_argument("--batch_size", type=int, default=4, help="Pairs per forward pass. Reduce if OOM (2 or 1).")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision autocast.")
    parser.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_json", required=True)

    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--pick_num", type=int, default=4)
    parser.add_argument("--avg_seg_len", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--min_gap", type=int, default=3)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    results = predict_for_folder(
        data_dir=args.data_dir,
        model_path=args.ckpt,
        device=device,
        window_size=args.window_size,
        pick_num=args.pick_num,
        avg_seg_len=args.avg_seg_len,
        alpha=args.alpha,
        min_gap=args.min_gap,
        seed=args.seed,
        batch_size=args.batch_size,
        use_amp=(not args.no_amp),
        amp_dtype=args.amp_dtype,
    )

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(to_python(results), f, indent=2, ensure_ascii=False)

    print(f"Saved predictions to {args.out_json}")


if __name__ == "__main__":
    main()
