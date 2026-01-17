import pandas as pd
import os, json, pickle, random, argparse
from tqdm import tqdm
from transformers import BertTokenizer
import torch
from keras.preprocessing.sequence import pad_sequences


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ===========================================================
# 1. Load CSV and group into dialogues
# ===========================================================

def load_dialogues(csv_path):
    df = pd.read_csv(csv_path)

    # ---- FIX NaN issues ----
    df["Content"] = df["content"].fillna("[EMPTY]").astype(str)
    df["Human"] = df["Human"].fillna("None").astype(str)
    df["AI"] = df["AI"].fillna("None").astype(str)

    # unify move label
    def pick_move(row):
        if row["Human"] != "None":
            return row["Human"]
        if row["AI"] != "None":
            return row["AI"]
        return "None"

    df["move"] = df.apply(pick_move, axis=1)

    # group dialogues
    sessions = {}
    for sid, sub in df.groupby("session_id"):
        sub = sub.sort_values("Utterance ID")
        sessions[sid] = list(zip(sub["Content"], sub["move"]))

    return sessions


# ===========================================================
# 2. Build NSP triples and topic-level input (like original code)
# ===========================================================

def build_triples(sessions, history=2):
    data = []
    topic_data = []

    for sid, utts in tqdm(sessions.items(), desc="Building triples"):
        texts = [u[0] for u in utts]
        moves = [u[1] for u in utts]
        L = len(texts)
        if L < 3:
            continue

        for mid in range(1, L):  # center utterance index
            context = []
            context_moves = []

            # left side context of size `history`
            left = max(0, mid-history)
            for i in range(left, mid):
                context.append(texts[i])
                context_moves.append(moves[i])

            # positive sample (next utterances)
            pos_utts = []
            pos_moves = []
            right = min(L, mid+history+1)
            for i in range(mid, right):
                pos_utts.append(texts[i])
                pos_moves.append(moves[i])

            # random negative within same dialogue
            neg_idx = random.choice([i for i in range(L) if i < left or i >= right])
            neg_utts = [texts[neg_idx]]
            neg_moves = [moves[neg_idx]]

            # hard negative: pick random other session
            other = random.choice(list(sessions.keys()))
            if other == sid:
                other = random.choice(list(sessions.keys()))
            other_utts, other_moves = sessions[other][0]
            hard_neg_utts = [other_utts]
            hard_neg_moves = [other_moves]

            # append triple: pos, neg, hard-neg
            data.append([
                ((context, context_moves), (pos_utts, pos_moves)),
                ((context, context_moves), (neg_utts, neg_moves)),
                ((context, context_moves), (hard_neg_utts, hard_neg_moves)),
            ])

            topic_data.append((texts, moves, mid))

    return data, topic_data


# ===========================================================
# 3. Tokenize + convert to training pickle
# ===========================================================

def encode_and_save(data, topic_data, save_path, history=2):
    """
    Convert triples + topic_data into an EEDI-compatible pickle:

    Output tuple:
        (
            train_inputs,          # [N, 2, MAX_LEN] long
            train_masks,           # [N, 2, MAX_LEN] long
            train_types,           # [N, 2, MAX_LEN] long
            topic_pairs,           # length N, each is [pos_tuple, neg_tuple]
            topic_trains,          # length N, each is [dial_len, 512] long
            topic_trains_mask,     # same shape as topic_trains
            topic_nums,            # length N, each is (dial_len, mid)
            topic_train_moves      # length N, each is list of moves for that dialogue
        )

    where each pos_tuple / neg_tuple is:
        (
            ctx_tok,    # tokenizer(ctx_utts, ...)  dict with 'input_ids', 'attention_mask'
            cur_tok,    # tokenizer(cur_utts, ...)
            len_ctx,    # int
            len_cur,    # int
            ctx_moves,  # list of move strings
            cur_moves   # list of move strings
        )
    """
    MAX_LEN = 512

    def pad_to_len(seq, maxlen, value):
        seq = list(seq)
        if len(seq) > maxlen:
            return seq[:maxlen]
        return seq + [value] * (maxlen - len(seq))

    train_inputs_list = []   # will become [N, 2, MAX_LEN]
    train_types_list = []    # [N, 2, MAX_LEN]
    topic_pairs = []         # [N]
    topic_trains = []        # [N]
    topic_trains_mask = []   # [N]
    topic_nums = []          # [N]
    topic_train_moves = []   # [N]

    assert len(data) == len(topic_data), "data and topic_data must have same length"

    for i, triple in tqdm(enumerate(data), total=len(data), desc="Encoding NSP + topic pairs"):
        # triple = [pos_pair, neg_pair, hard_neg_pair]
        pos_pair, neg_pair, _ = triple

        (ctx_utts_pos, ctx_moves_pos), (cur_utts_pos, cur_moves_pos) = pos_pair
        (ctx_utts_neg, ctx_moves_neg), (cur_utts_neg, cur_moves_neg) = neg_pair

        # ---------------- POS NSP ----------------
        sent1_pos = ctx_utts_pos[-1] if len(ctx_utts_pos) > 0 else ""
        sent2_pos = cur_utts_pos[0] if len(cur_utts_pos) > 0 else ""

        enc1_pos = tokenizer.encode(
            sent1_pos,
            add_special_tokens=True,
            truncation=True,
            max_length=256
        )
        enc2_pos = tokenizer.encode(
            sent2_pos,
            add_special_tokens=True,
            truncation=True,
            max_length=256
        )

        ids_pos = enc1_pos[:-1] + enc2_pos
        types_pos = [0] * len(enc1_pos[:-1]) + [1] * len(enc2_pos)

        ids_pos = pad_to_len(ids_pos, MAX_LEN, value=0)
        types_pos = pad_to_len(types_pos, MAX_LEN, value=0)  # type-ids default 0 is fine

        # ---------------- NEG NSP ----------------
        sent1_neg = ctx_utts_neg[-1] if len(ctx_utts_neg) > 0 else ""
        sent2_neg = cur_utts_neg[0] if len(cur_utts_neg) > 0 else ""

        enc1_neg = tokenizer.encode(
            sent1_neg,
            add_special_tokens=True,
            truncation=True,
            max_length=256
        )
        enc2_neg = tokenizer.encode(
            sent2_neg,
            add_special_tokens=True,
            truncation=True,
            max_length=256
        )

        ids_neg = enc1_neg[:-1] + enc2_neg
        types_neg = [0] * len(enc1_neg[:-1]) + [1] * len(enc2_neg)

        ids_neg = pad_to_len(ids_neg, MAX_LEN, value=0)
        types_neg = pad_to_len(types_neg, MAX_LEN, value=0)

        # One training example = stack [pos, neg]
        train_inputs_list.append([ids_pos, ids_neg])
        train_types_list.append([types_pos, types_neg])

        # ---------------- Topic-level (context + windows) ----------------
        ctx_tok_pos = tokenizer(
            ctx_utts_pos,
            truncation=True,
            max_length=256
        )
        cur_tok_pos = tokenizer(
            cur_utts_pos,
            truncation=True,
            max_length=256
        )

        pos_tuple = (
            ctx_tok_pos,
            cur_tok_pos,
            len(ctx_utts_pos),
            len(cur_utts_pos),
            ctx_moves_pos,
            cur_moves_pos,
        )

        ctx_tok_neg = tokenizer(
            ctx_utts_neg,
            truncation=True,
            max_length=256
        )
        cur_tok_neg = tokenizer(
            cur_utts_neg,
            truncation=True,
            max_length=256
        )

        neg_tuple = (
            ctx_tok_neg,
            cur_tok_neg,
            len(ctx_utts_neg),
            len(cur_utts_neg),
            ctx_moves_neg,
            cur_moves_neg,
        )

        topic_pairs.append([pos_tuple, neg_tuple])

        # ---------------- Full dialogue topic_train ----------------
        dial_utts, dial_moves, mid = topic_data[i]
        tok_full = tokenizer(
            dial_utts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        topic_trains.append(tok_full["input_ids"])      # [dial_len, 512]
        topic_trains_mask.append(tok_full["attention_mask"])
        topic_nums.append((len(dial_utts), mid))
        topic_train_moves.append(dial_moves)

    # ------------- Convert NSP lists to tensors -------------
    train_inputs = torch.tensor(train_inputs_list, dtype=torch.long)     # [N, 2, MAX_LEN]
    train_types = torch.tensor(train_types_list, dtype=torch.long)       # [N, 2, MAX_LEN]
    train_masks = (train_inputs > 0).long()                              # [N, 2, MAX_LEN]

    # ------------- Save everything as a single pickle -------------
    pickle.dump(
        (
            train_inputs,          # 0
            train_masks,           # 1
            train_types,           # 2
            topic_pairs,           # 3
            topic_trains,          # 4
            topic_trains_mask,     # 5
            topic_nums,            # 6
            topic_train_moves      # 7
        ),
        open(save_path, "wb")
    )

    print(f"\n✅ Saved taxonomy-aware dataset → {save_path}")
    print(f"   #examples = {len(train_inputs)}")




# ===========================================================
# MAIN ENTRY
# ===========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--save_name", default="data/TalkMoves_all_w_move_taxonomy.pkl")
    parser.add_argument("--history", type=int, default=2)
    args = parser.parse_args()

    print("Loading CSV…")
    sessions = load_dialogues(args.csv_path)

    print("Building triples…")
    data, topic_data = build_triples(sessions, history=args.history)

    print("Encoding and saving pickle…")
    encode_and_save(data, topic_data, args.save_name, history=args.history)
