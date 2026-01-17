import re
import os
import json
import torch
import random
import pickle
import argparse
from tqdm import tqdm

from model_CBRAG import SegModel

from torch.cuda import amp
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, Dataset

from transformers import (
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.optim import AdamW


DATASET = {
    '711': 'dialseg711',
    'eedi': 'eedi_train',
    'eedi_move': 'train_w_move_taxonomy',
    'CLASS_all':'CLASS_all',
    'TalkMoves_all':'TalkMoves_all'}


def get_mask(tensor: torch.Tensor) -> torch.Tensor:
    attention_masks = []
    for sent in tensor:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return torch.tensor(attention_masks)


# =========================
#  Base dataset (no moves)
# =========================
class BaseSegDataset(Dataset):
    """
    For old-format pickles:

        (
            train_inputs, train_masks, train_types,
            topic_pairs, topic_trains, topic_trains_mask, topic_nums
        )
    """

    def __init__(self, loaded_data):
        self.loaded_data = loaded_data

    def __getitem__(self, idx):
        return [item[idx] for item in self.loaded_data]

    def __len__(self):
        return len(self.loaded_data[0])

    def collect_fn(self, examples):
        # coherence inputs
        coheren_inputs = pad_sequence([ex[0] for ex in examples], batch_first=True)
        coheren_mask = pad_sequence([ex[1] for ex in examples], batch_first=True)
        coheren_type = pad_sequence([ex[2] for ex in examples], batch_first=True)

        # topic context / pos / neg (text only)
        topic_context = pad_sequence(
            [torch.tensor(j) for ex in examples for j in ex[3][0][0]['input_ids']],
            batch_first=True
        )
        topic_pos = pad_sequence(
            [torch.tensor(j) for ex in examples for j in ex[3][0][1]['input_ids']],
            batch_first=True
        )
        topic_neg = pad_sequence(
            [torch.tensor(j) for ex in examples for j in ex[3][1][1]['input_ids']],
            batch_first=True
        )

        topic_context_num = [ex[3][0][2] for ex in examples]
        topic_pos_num = [ex[3][0][3] for ex in examples]
        topic_neg_num = [ex[3][1][3] for ex in examples]

        topic_context_mask = get_mask(topic_context)
        topic_pos_mask = get_mask(topic_pos)
        topic_neg_mask = get_mask(topic_neg)

        # full-dialogue topic_train
        topic_train = pad_sequence(
            [row for ex in examples for row in ex[4]],
            batch_first=True
        )
        topic_train_mask = pad_sequence(
            [row for ex in examples for row in ex[5]],
            batch_first=True
        )
        topic_num = [ex[6] for ex in examples]

        return (
            coheren_inputs, coheren_mask, coheren_type,
            topic_context, topic_pos, topic_neg,
            topic_context_mask, topic_pos_mask, topic_neg_mask,
            topic_context_num, topic_pos_num, topic_neg_num,
            topic_train, topic_train_mask, topic_num
        )


# ================================
#  Taxonomy-aware dataset (moves)
# ================================
class TaxonomySegDataset(Dataset):
    """
    For taxonomy-aware pickles:

        (
            train_inputs,          # 0
            train_masks,           # 1
            train_types,           # 2
            topic_pairs,           # 3
            topic_trains,          # 4
            topic_trains_mask,     # 5
            topic_nums,            # 6
            topic_train_moves      # 7
        )
    """

    def __init__(self, loaded_data):
        self.train_inputs = loaded_data[0]
        self.train_masks = loaded_data[1]
        self.train_types = loaded_data[2]
        self.topic_pairs = loaded_data[3]
        self.topic_trains = loaded_data[4]
        self.topic_trains_mask = loaded_data[5]
        self.topic_nums = loaded_data[6]
        self.topic_train_moves = loaded_data[7]

        # ---- determine a safe length where ALL lists are valid ----
        len_inputs = len(self.train_inputs)
        len_pairs = len(self.topic_pairs)
        len_trains = len(self.topic_trains)
        len_trains_mask = len(self.topic_trains_mask)
        len_nums = len(self.topic_nums)
        len_moves = len(self.topic_train_moves)

        self.num_examples = min(
            len_inputs, len_pairs, len_trains, len_trains_mask, len_nums, len_moves
        )

        print(
            f"[TaxonomySegDataset] Lengths -> "
            f"inputs={len_inputs}, pairs={len_pairs}, trains={len_trains}, "
            f"nums={len_nums}, moves={len_moves} -> using num_examples={self.num_examples}"
        )

        # ---- build move_vocab from topic_pairs + topic_train_moves ----
        all_moves = set()

        # from full dialogues (only up to num_examples to stay consistent)
        for moves in self.topic_train_moves[:self.num_examples]:
            for m in moves:
                all_moves.add(m)

        # from topic_pairs (also only up to num_examples)
        for pair in self.topic_pairs[:self.num_examples]:
            pos_tuple, neg_tuple = pair
            # unpack
            _, _, len_c_pos, len_p_pos, ctx_moves_pos, cur_moves_pos = pos_tuple
            _, _, len_c_neg, len_n_neg, ctx_moves_neg, cur_moves_neg = neg_tuple

            for m in ctx_moves_pos:
                all_moves.add(m)
            for m in cur_moves_pos:
                all_moves.add(m)
            for m in ctx_moves_neg:
                all_moves.add(m)
            for m in cur_moves_neg:
                all_moves.add(m)

        all_moves = list(all_moves)
        # ensure "None" exists and is index 0
        if "None" in all_moves:
            all_moves.remove("None")
            all_moves = ["None"] + all_moves
        else:
            all_moves = ["None"] + all_moves

        self.move_vocab = {m: i for i, m in enumerate(all_moves)}
        self.none_id = self.move_vocab["None"]

        print(f"[TaxonomySegDataset] Built move_vocab with {len(self.move_vocab)} moves.")

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        return (
            self.train_inputs[idx],
            self.train_masks[idx],
            self.train_types[idx],
            self.topic_pairs[idx],
            self.topic_trains[idx],
            self.topic_trains_mask[idx],
            self.topic_nums[idx],
            self.topic_train_moves[idx],
        )

    def _move_to_id(self, m: str) -> int:
        return self.move_vocab.get(m, self.none_id)

    def collect_fn(self, examples):
        # ----- coherence inputs -----
        coheren_inputs = pad_sequence([ex[0] for ex in examples], batch_first=True)
        coheren_mask = pad_sequence([ex[1] for ex in examples], batch_first=True)
        coheren_type = pad_sequence([ex[2] for ex in examples], batch_first=True)

        # ----- topic-level (context / pos / neg) -----
        topic_context_list, topic_pos_list, topic_neg_list = [], [], []
        topic_context_num, topic_pos_num, topic_neg_num = [], [], []
        topic_context_move_ids, topic_pos_move_ids, topic_neg_move_ids = [], [], []

        for ex in examples:
            topic_pair = ex[3]
            pos_tuple, neg_tuple = topic_pair

            context_pos, cur_pos, len_c_pos, len_p_pos, ctx_moves_pos, cur_moves_pos = pos_tuple
            context_neg, cur_neg, len_c_neg, len_n_neg, ctx_moves_neg, cur_moves_neg = neg_tuple

            # context (pos side)
            assert len(ctx_moves_pos) == len_c_pos
            for inp_ids, m in zip(context_pos['input_ids'], ctx_moves_pos):
                topic_context_list.append(torch.tensor(inp_ids))
                topic_context_move_ids.append(self._move_to_id(m))
            topic_context_num.append(len_c_pos)

            # pos window
            assert len(cur_moves_pos) == len_p_pos
            for inp_ids, m in zip(cur_pos['input_ids'], cur_moves_pos):
                topic_pos_list.append(torch.tensor(inp_ids))
                topic_pos_move_ids.append(self._move_to_id(m))
            topic_pos_num.append(len_p_pos)

            # neg window
            assert len(cur_moves_neg) == len_n_neg
            for inp_ids, m in zip(cur_neg['input_ids'], cur_moves_neg):
                topic_neg_list.append(torch.tensor(inp_ids))
                topic_neg_move_ids.append(self._move_to_id(m))
            topic_neg_num.append(len_n_neg)

        topic_context = pad_sequence(topic_context_list, batch_first=True)
        topic_pos = pad_sequence(topic_pos_list, batch_first=True)
        topic_neg = pad_sequence(topic_neg_list, batch_first=True)

        topic_context_mask = get_mask(topic_context)
        topic_pos_mask = get_mask(topic_pos)
        topic_neg_mask = get_mask(topic_neg)

        topic_context_move = torch.tensor(topic_context_move_ids, dtype=torch.long)
        topic_pos_move = torch.tensor(topic_pos_move_ids, dtype=torch.long)
        topic_neg_move = torch.tensor(topic_neg_move_ids, dtype=torch.long)

        # ----- full-dialogue topic_train (text + moves) -----
        topic_train_list, topic_train_mask_list = [], []
        topic_num_list = []
        topic_train_move_ids = []

        for ex in examples:
            topic_train_ids = ex[4]
            topic_train_mask = ex[5]
            dial_len, mid = ex[6]
            moves = list(ex[7])

            # align moves with dial_len
            if len(moves) < dial_len:
                moves = moves + ["None"] * (dial_len - len(moves))
            elif len(moves) > dial_len:
                moves = moves[:dial_len]

            for row in topic_train_ids:
                topic_train_list.append(row)
            for row in topic_train_mask:
                topic_train_mask_list.append(row)

            topic_num_list.append((dial_len, mid))

            for m in moves:
                topic_train_move_ids.append(self._move_to_id(m))

        topic_train = pad_sequence(topic_train_list, batch_first=True)
        topic_train_mask = pad_sequence(topic_train_mask_list, batch_first=True)
        topic_train_move = torch.tensor(topic_train_move_ids, dtype=torch.long)

        return (
            coheren_inputs, coheren_mask, coheren_type,
            topic_context, topic_pos, topic_neg,
            topic_context_mask, topic_pos_mask, topic_neg_mask,
            topic_context_num, topic_pos_num, topic_neg_num,
            topic_train, topic_train_mask, topic_num_list,
            topic_context_move, topic_pos_move, topic_neg_move,
            topic_train_move
        )


# =========================
#         TRAINING
# =========================
def main(args):
    data_path = f'./data/{DATASET[args.dataset]}{args.data_name}.pkl'
    print(f"Loading data from {data_path}")
    loaded_data = pickle.load(open(data_path, 'rb'))

    # >>> NEW: max_samples slicing <<<
    if args.max_samples > 0:
        max_n = min(args.max_samples, len(loaded_data[0]))
        print(f"[main] Using only first {max_n} samples (out of {len(loaded_data[0])})")
        sliced = []
        for i, part in enumerate(loaded_data):
            try:
                sliced.append(part[:max_n])
            except TypeError:
                # if something is not sliceable, leave as is
                sliced.append(part)
        loaded_data = tuple(sliced)
    # <<< END NEW >>>

    epochs = args.epoch
    global_step = continue_from_global_step = 0

    # choose dataset class
    if args.dataset == 'eedi_move':
        train_data = TaxonomySegDataset(loaded_data)
        taxonomy_mode = True
        print("Using TaxonomySegDataset (taxonomy-aware).")
    else:
        train_data = BaseSegDataset(loaded_data)
        taxonomy_mode = False
        print("Using BaseSegDataset (no moves).")

    train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        sampler=train_sampler,
        batch_size=args.batch_size,
        collate_fn=train_data.collect_fn
    )

    scaler = amp.GradScaler(enabled=(not args.no_amp))
    model = SegModel(
        margin=args.margin,
        train_split=args.train_split,
        window_size=args.window_size
    ).to(args.device)

    if args.resume and args.ckpt:
        ckpt_path = f'{args.root}/model/{args.ckpt}'
        print(f"Resuming from checkpoint {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path), False)

    if args.local_rank != -1:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    epoch_loss = {}
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(total_steps * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    for epoch_i in tqdm(range(epochs)):
        print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
        total_loss = 0.0
        model.train()
        epoch_iterator = tqdm(train_dataloader, disable=args.local_rank != -1)
        window_size = args.window_size

        for step, batch in enumerate(epoch_iterator):
            if global_step < continue_from_global_step:
                if (step + 1) % args.accum == 0:
                    scheduler.step()
                    global_step += 1
                continue

            if not taxonomy_mode:
                input_data = {
                    'coheren_inputs': batch[0].to(args.device),
                    'coheren_mask': batch[1].to(args.device),
                    'coheren_type': batch[2].to(args.device),
                    'topic_context': batch[3].to(args.device),
                    'topic_pos': batch[4].to(args.device),
                    'topic_neg': batch[5].to(args.device),
                    'topic_context_mask': batch[6].to(args.device),
                    'topic_pos_mask': batch[7].to(args.device),
                    'topic_neg_mask': batch[8].to(args.device),
                    'topic_context_num': batch[9],
                    'topic_pos_num': batch[10],
                    'topic_neg_num': batch[11],
                    'topic_train': batch[12].to(args.device),
                    'topic_train_mask': batch[13].to(args.device),
                    'topic_num': batch[14],
                }
            else:
                input_data = {
                    'coheren_inputs': batch[0].to(args.device),
                    'coheren_mask': batch[1].to(args.device),
                    'coheren_type': batch[2].to(args.device),
                    'topic_context': batch[3].to(args.device),
                    'topic_pos': batch[4].to(args.device),
                    'topic_neg': batch[5].to(args.device),
                    'topic_context_mask': batch[6].to(args.device),
                    'topic_pos_mask': batch[7].to(args.device),
                    'topic_neg_mask': batch[8].to(args.device),
                    'topic_context_num': batch[9],
                    'topic_pos_num': batch[10],
                    'topic_neg_num': batch[11],
                    'topic_train': batch[12].to(args.device),
                    'topic_train_mask': batch[13].to(args.device),
                    'topic_num': batch[14],
                    'topic_context_move': batch[15].to(args.device),
                    'topic_pos_move': batch[16].to(args.device),
                    'topic_neg_move': batch[17].to(args.device),
                    'topic_train_move': batch[18].to(args.device),
                }

            model.zero_grad()

            with amp.autocast(enabled=(not args.no_amp)):
                loss, margin_loss, topic_loss = model(input_data, window_size)

            if args.n_gpu > 1:
                loss = loss.mean()

            total_loss += loss.item()

            if not args.no_amp:
                scaler.scale(loss).backward()
                if (step + 1) % args.accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    global_step += 1
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if (step + 1) % args.accum == 0:
                    optimizer.step()
                    scheduler.step()
                    global_step += 1

        avg_train_loss = total_loss / len(train_dataloader)
        epoch_loss[epoch_i] = avg_train_loss

        if args.local_rank in [-1, 0]:
            print(f'=========== Epoch {epoch_i} loss: {avg_train_loss:.6f}')
            PATH = f'./models/{args.save_model_name}/{str(epoch_i)}-{str(global_step)}'
            model_to_save = model.module if hasattr(model, 'module') else model

            if continue_from_global_step <= global_step:
                print('Saving model to ' + PATH)
                torch.save(model_to_save.state_dict(), PATH)

        if epoch_i == args.epoch:
            break

    return epoch_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="One of: 711, eedi, eedi_move, CLASS_all, TalkMoves_all")
    parser.add_argument("--save_model_name", required=True)

    # model parameters
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument("--train_split", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=5)

    # path parameters
    parser.add_argument("--ckpt")
    parser.add_argument("--data_name", default='')
    parser.add_argument("--root", default='.')
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3407)

    # train parameters
    parser.add_argument('--accum', type=int, default=1)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Use only first N samples from the pickle (for debugging/speed).")

    # device parameters
    parser.add_argument("--no_amp", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = torch.cuda.device_count()

    args.device = device
    out_path = f'{args.root}/model/{args.save_model_name}'
    os.makedirs(out_path, exist_ok=True)
    epoch_loss = main(args)
    json.dump(epoch_loss, open(f'{out_path}/loss.json', 'w'))
