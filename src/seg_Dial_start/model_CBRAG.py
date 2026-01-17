import torch
import numpy as np
import faiss
import bisect
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel
from typing import Optional

# ------------------------------------------------------
#  MoveMemory: stores only labeled utterances
# ------------------------------------------------------
class MoveMemory:
    def __init__(self, hidden_size, num_moves, device):
        self.device = device
        self.hidden_size = hidden_size
        self.num_moves = num_moves

        self.embeddings = []   # list of tensors [H]
        self.move_ids = []     # list of ints
        self.index = None      # faiss index

    def add(self, h, move_id):
        """Add a single labeled utterance into memory."""
        if move_id < 0:  # skip unlabeled
            return
        self.embeddings.append(h.detach().cpu())
        self.move_ids.append(move_id)

    def build(self):
        """Build a FAISS index for nearest-neighbor search."""
        if len(self.embeddings) == 0:
            return

        mat = torch.stack(self.embeddings).numpy().astype("float32")
        self.faiss_matrix = mat

        self.index = faiss.IndexFlatIP(self.hidden_size)   # cosine sim via normalized dot-product
        faiss.normalize_L2(mat)
        self.index.add(mat)

        self.move_ids = torch.tensor(self.move_ids, dtype=torch.long).to(self.device)

    def retrieve(self, query_h, K=15):
        """Retrieve K nearest labeled moves and return aggregated move embedding."""
        if self.index is None or len(self.move_ids) == 0:
            return None, None

        q = query_h.detach().cpu().unsqueeze(0).numpy().astype("float32")
        faiss.normalize_L2(q)

        scores, idx = self.index.search(q, K)
        idx = idx[0]

        # filter invalid -1
        valid = [i for i in idx if i >= 0]
        if len(valid) == 0:
            return None, None

        retrieved_moves = self.move_ids[valid]  # [K]
        retrieved_scores = torch.tensor(scores[0][:len(valid)], device=self.move_ids.device)

        # softmax-normalize retrieval strength
        w = F.softmax(retrieved_scores, dim=0)

        return retrieved_moves, w


# ------------------------------------------------------
    # SegModel with CBRAG integrated everywhere
# ------------------------------------------------------
class MarginRankingLoss():
    def __init__(self, margin):
        self.margin = margin

    def __call__(self, p_scores, n_scores):
        scores = self.margin - (p_scores - n_scores)
        scores = scores.clamp(min=0)
        return scores.mean()


class SegModel(nn.Module):
    def __init__(self, model_path='',
                 margin=1,
                 train_split=5,
                 window_size=5,
                 num_moves=4,
                 move_loss_weight=0.5,
                 alpha=0.5,   # CBRAG fusion strength
                 K=15):

        super().__init__()

        self.margin = margin
        self.train_split = train_split
        self.window_size = window_size
        self.move_loss_weight = move_loss_weight
        self.alpha = alpha
        self.K = K
        self.num_moves = num_moves

        # -----------------------
        # encoders
        # -----------------------
        self.topic_model = AutoModel.from_pretrained(
            model_path + "princeton-nlp/sup-simcse-bert-base-uncased",
            use_safetensors=True
        )
        self.hidden_size = self.topic_model.config.hidden_size

        self.coheren_model = BertForNextSentencePrediction.from_pretrained(
            model_path + "bert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=True
        )

        # -----------------------
        # move embedding + classifier
        # -----------------------
        self.move_embedding = nn.Embedding(num_moves, self.hidden_size)
        self.move_classifier = nn.Linear(self.hidden_size, num_moves)
        self.move_loss_fct = nn.CrossEntropyLoss()

        self.score_loss = MarginRankingLoss(self.margin)

        self.move_memory = None   # Will be built dynamically each epoch


    # ------------------------------------------------------
    # Build MoveMemory from topic_train embeddings + move labels
    # ------------------------------------------------------
    def build_move_memory(self, topic_embeddings, move_ids):
        device = topic_embeddings.device
        self.move_memory = MoveMemory(self.hidden_size, self.num_moves, device)

        for h, m in zip(topic_embeddings, move_ids):
            m = m.item()
            if m >= 0:  # only labeled utterances
                self.move_memory.add(h, m)

        self.move_memory.build()

    # ------------------------------------------------------
    # Apply CBRAG to a batch of embeddings
    # ------------------------------------------------------
    def fuse_with_rag(self, topic_embs):
        """topic_embs: [N, H]"""
        if self.move_memory is None:
            return topic_embs

        fused = []
        for h in topic_embs:
            retrieved_moves, w = self.move_memory.retrieve(h, K=self.K)

            if retrieved_moves is None:
                fused.append(h)
                continue

            move_vecs = self.move_embedding(retrieved_moves)  # [K, H]
            r = (w.unsqueeze(1) * move_vecs).sum(dim=0)        # [H]

            fused.append(h + self.alpha * r)

        return torch.stack(fused)


    # ------------------------------------------------------
    # forward()
    # ------------------------------------------------------
    def forward(self, input_data, window_size=None):

        device = input_data['coheren_inputs'].device
        window_size = window_size or self.window_size

        # NSP coherence scoring
        coheren_pos_scores, _ = self.coheren_model(
            input_data['coheren_inputs'][:, 0, :],
            attention_mask=input_data['coheren_mask'][:, 0, :],
            token_type_ids=input_data['coheren_type'][:, 0, :]
        )

        coheren_neg_scores, _ = self.coheren_model(
            input_data['coheren_inputs'][:, 1, :],
            attention_mask=input_data['coheren_mask'][:, 1, :],
            token_type_ids=input_data['coheren_type'][:, 1, :]
        )

        batch_size = len(input_data['topic_context_num'])

        # ---------------------------------------------
        # Topic embeddings (SimCSE)
        # ---------------------------------------------
        topic_context = self.topic_model(
            input_data['topic_context'],
            input_data['topic_context_mask']
        )[1]

        topic_pos = self.topic_model(
            input_data['topic_pos'],
            input_data['topic_pos_mask']
        )[1]

        topic_neg = self.topic_model(
            input_data['topic_neg'],
            input_data['topic_neg_mask']
        )[1]

        # ------------------------------------------------------
        # 1) Build move memory using topic_train embeddings
        # ------------------------------------------------------
        topic_all_train = self.topic_model(
            input_data['topic_train'],
            input_data['topic_train_mask']
        )[1]   # [N_all, H]

        move_all_train = input_data['topic_train_move']  # [N_all]

        self.build_move_memory(topic_all_train, move_all_train)

        # ------------------------------------------------------
        # 2) Fuse topic embeddings via CBRAG
        # ------------------------------------------------------
        topic_context = self.fuse_with_rag(topic_context)
        topic_pos = self.fuse_with_rag(topic_pos)
        topic_neg = self.fuse_with_rag(topic_neg)
        topic_all_train = self.fuse_with_rag(topic_all_train)

        # ------------------------------------------------------
        # topic label prediction auxiliary loss
        # ------------------------------------------------------
        move_logits = self.move_classifier(topic_all_train)
        aux_move_loss = self.move_loss_fct(
            move_logits.view(-1, self.num_moves),
            move_all_train.view(-1)
        )

        # ------------------------------------------------------
        # Pseudo-segmentation loss (topic_train)
        # ------------------------------------------------------
        topic_loss = self.topic_train(input_data, topic_all_train, window_size)

        # ------------------------------------------------------
        # Margin loss between positive and negative coherence + topic
        # ------------------------------------------------------
        topic_context_mean, topic_pos_mean, topic_neg_mean = [], [], []
        cidx = pidx = nidx = 0

        for c, p, n in zip(
            input_data['topic_context_num'],
            input_data['topic_pos_num'],
            input_data['topic_neg_num']
        ):
            topic_context_mean.append(topic_context[cidx:cidx+c].mean(0))
            topic_pos_mean.append(topic_pos[pidx:pidx+p].mean(0))
            topic_neg_mean.append(topic_neg[nidx:nidx+n].mean(0))
            cidx += c; pidx += p; nidx += n

        topic_context_mean = pad_sequence(topic_context_mean, batch_first=True)
        topic_pos_mean = pad_sequence(topic_pos_mean, batch_first=True)
        topic_neg_mean = pad_sequence(topic_neg_mean, batch_first=True)

        pos_scores = coheren_pos_scores[0][:, 0] + F.cosine_similarity(topic_context_mean, topic_pos_mean)
        neg_scores = coheren_neg_scores[0][:, 0] + F.cosine_similarity(topic_context_mean, topic_neg_mean)

        margin_loss = self.score_loss(pos_scores, neg_scores)

        # ------------------------------------------------------
        # final loss
        # ------------------------------------------------------
        loss = margin_loss + topic_loss + self.move_loss_weight * aux_move_loss

        return loss, margin_loss, topic_loss


    # ------------------------------------------------------
    # topic_train() (pseudo segmentation)
    # ------------------------------------------------------
    def topic_train(self, input_data, topic_all, window_size):
        device = topic_all.device
        batch_size = len(input_data['topic_context_num'])

        # same pseudo-segmentation you originally used
        topic_margin_loss = torch.tensor(0.0, device=device)
        count = 0
        valid = batch_size

        for b in range(batch_size):
            cur_num = input_data['topic_num'][b]
            dial_len, cur_utt = cur_num[0], cur_num[1]
            cur = topic_all[count:count + dial_len]

            # compute cosine similarities
            top_cons, top_curs = [], []
            for i in range(1, dial_len):
                left = max(0, i - 2)
                right = min(dial_len, i + 2)
                top_cons.append(cur[left:i].mean(0))
                top_curs.append(cur[i:right].mean(0))

            top_cons = pad_sequence(top_cons, batch_first=True)
            top_curs = pad_sequence(top_curs, batch_first=True)

            scores = F.cosine_similarity(top_cons, top_curs)
            depth = tet(torch.sigmoid(scores))
            depth = np.array(depth)

            segs = np.argsort(depth)[-self.train_split:] + 1
            segs = [0] + segs.tolist() + [dial_len]
            segs.sort()

            mid_idx = bisect.bisect(segs, cur_utt)
            left = segs[mid_idx-1]
            right = segs[mid_idx]

            pos_left = max(left, cur_utt - window_size)
            pos_right = min(right, cur_utt + window_size + 1)

            neg_left = min(segs[max(0, mid_idx-1)], cur_utt - window_size)
            neg_right = max(segs[mid_idx], cur_utt + window_size + 1)

            anchor = cur[cur_utt].unsqueeze(0)
            pos = torch.cat((cur[pos_left:cur_utt], cur[cur_utt+1:pos_right]), dim=0)
            if pos.shape[0] == 0:
                valid -= 1
                count += dial_len
                continue

            neg = torch.cat((topic_all[:count+neg_left], topic_all[count+neg_right:]), dim=0)

            pos_score = F.cosine_similarity(anchor, pos)
            neg_score = F.cosine_similarity(anchor, neg)

            pos_expand = pos_score.unsqueeze(0).repeat(neg_score.shape[0], 1).T.flatten()
            neg_expand = neg_score.repeat(pos_score.shape[0])

            loss = self.score_loss(pos_expand, neg_expand)
            topic_margin_loss += loss

            count += dial_len

        topic_margin_loss /= max(valid, 1)
        return topic_margin_loss


    # ------------------------------------------------------
    # inference (CBRAG included)
    # ------------------------------------------------------
    def infer(self, coheren_input, coheren_mask, coheren_type,
              topic_input, topic_mask,
              topic_num,
              topic_move_input=None):

        device = coheren_input.device

        coheren_scores, _ = self.coheren_model(
            coheren_input,
            attention_mask=coheren_mask,
            token_type_ids=coheren_type
        )

        topic_context = self.topic_model(topic_input[0], topic_mask[0])[1]
        topic_cur = self.topic_model(topic_input[1], topic_mask[1])[1]

        # CBRAG fusion
        topic_context = self.fuse_with_rag(topic_context)
        topic_cur = self.fuse_with_rag(topic_cur)

        ctx_count = cur_count = 0
        ctx_mean, cur_mean = [], []

        for c, j in zip(topic_num[0], topic_num[1]):
            ctx_mean.append(topic_context[ctx_count:ctx_count+c].mean(0))
            cur_mean.append(topic_cur[cur_count:cur_count+j].mean(0))
            ctx_count += c
            cur_count += j

        ctx_mean = pad_sequence(ctx_mean, batch_first=True)
        cur_mean = pad_sequence(cur_mean, batch_first=True)

        topic_scores = F.cosine_similarity(ctx_mean, cur_mean)
        final_scores = coheren_scores[0][:, 0] + topic_scores

        return torch.sigmoid(final_scores).cpu().detach().numpy().tolist()



# ------------------------------------------------------
# tet() function preserved from your code
# ------------------------------------------------------
def tet(scores):
    output = []
    scores = scores.detach().cpu()
    for i in range(len(scores)):
        lflag = rflag = scores[i]
        for r in range(i+1, len(scores)):
            if rflag <= scores[r]:
                rflag = scores[r]
            else:
                break
        for l in range(i-1, -1, -1):
            if lflag <= scores[l]:
                lflag = scores[l]
            else:
                break
        depth = 0.5 * (lflag + rflag - 2 * scores[i])
        output.append(depth)
    return output
