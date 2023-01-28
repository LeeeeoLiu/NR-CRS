import io
import os
import pickle
import torch
import faiss
from torch import nn
import torch.nn.functional as F
from loguru import logger
import wandb

from .tools import SelfAttentionSeq, GateScore
from .reslayer import ResidualNet

KNN_CAND_NUM = 1024


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class Vector_KNN(object):
    """
    Faiss-based knn module for finding neighboring conversations
    """

    def __init__(self, hidden_size, nlist=10, nprobe=10, use_gpu=False):
        '''
        :param hidden_size:
        :param nlist: number of cluster center
        :param nprobe:
        '''
        self.nprobe = nprobe
        quantizer = faiss.IndexFlatIP(hidden_size)
        self.cpu_index = faiss.IndexIVFFlat(quantizer, hidden_size, nlist,
                                            faiss.METRIC_INNER_PRODUCT)
        if use_gpu:
            res = faiss.StandardGpuResources()
            gpu_index_ivf = faiss.index_cpu_to_gpu(res, 1, self.cpu_index)
            self.act_index = gpu_index_ivf
        else:
            self.act_index = self.cpu_index

    def index(self, vector_to_index, add_with_ids=None):
        if torch.cuda.is_available():
            vector_to_index = vector_to_index.cpu()
        vector_to_index = vector_to_index.numpy()
        self.act_index.train(vector_to_index)
        if add_with_ids is not None:
            self.act_index.add_with_ids(vector_to_index, add_with_ids)
        else:
            self.act_index.add(vector_to_index)

    def search(self, vector, k, ignore_k_neighbor_idx=None):
        self.act_index.nprobe = self.nprobe
        gpu_device = vector.device
        if torch.cuda.is_available():
            vector = vector.cpu()

        if ignore_k_neighbor_idx is not None:
            D, I = self.act_index.search(vector.numpy(), 2 * k)
            I = torch.tensor(I, device=gpu_device)
            filter_I = []
            for bidx in range(len(ignore_k_neighbor_idx)):
                if ignore_k_neighbor_idx[bidx] not in I[bidx]:
                    filter_I.append(I[bidx, :k])
                else:
                    list_I = I[bidx].tolist()
                    list_I.remove(ignore_k_neighbor_idx[bidx])
                    filter_I.append(torch.tensor(list_I[:k],
                                                 device=gpu_device))

            return torch.stack(filter_I)
        else:
            D, I = self.act_index.search(vector.numpy(), k)
            I = torch.tensor(I, device=gpu_device)

            return I


def vknn_conv_item(conv_cands_vecs,
                   nlist=100,
                   nprobe=100,
                   item2vec_embs=None):
    gpu_device = item2vec_embs.device
    vdim = item2vec_embs.size(1)
    vknn = Vector_KNN(vdim, nlist=nlist, nprobe=nprobe, use_gpu=False)
    vector_to_index = torch.stack([conv_vs[0]
                                  for conv_vs in conv_cands_vecs.values()])
    vknn.index(vector_to_index.to(gpu_device))
    return vknn


class CPU_Unpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class NRCRS(nn.Module):
    """
    Neighboring relation for conversational recommendation inlcudes:
    - neighboring conversations
    - neighboring items
    """

    def __init__(self, config, user_model):
        super().__init__()
        self.user_model = user_model
        self.config = config
        self.sample_num = config.sample_num
        self.kg_emb_dim = config.kg_emb_dim
        self.n_entity = config.n_entity
        self.ablation = config.ablation
        self.deep_analysis = config.deep_analysis
        self.nr_conv_init(config)

    def nr_conv_init(self, config):
        self.conv_item_dict = {}
        self.vknn = None

        if self.config.neighboring_convs:
            if self.config.sim_crs_norm_type == 'residual':
                if self.config.XXXX_17 in ['pre_rec_attn', 'pre_rec_avg', 'pre_rec_sum']:
                    self.sim_crs_norm = ResidualNet(
                        self.kg_emb_dim * 2, self.kg_emb_dim, self.kg_emb_dim, dropout_probability=self.config.dropout)
                else:
                    self.sim_crs_norm = ResidualNet(
                        self.kg_emb_dim, self.kg_emb_dim, self.kg_emb_dim, dropout_probability=self.config.dropout)
            else:
                if self.config.XXXX_17 in ['pre_rec_attn', 'pre_rec_avg', 'pre_rec_sum']:
                    self.sim_crs_norm = nn.Sequential(
                        nn.Linear(self.kg_emb_dim * 2, self.kg_emb_dim),
                        nn.LeakyReLU(),
                        nn.Dropout(p=self.config.dropout),
                        nn.Linear(self.kg_emb_dim, self.kg_emb_dim))
                else:
                    self.sim_crs_norm = nn.Sequential(
                        nn.Linear(self.kg_emb_dim, self.kg_emb_dim),
                        nn.LeakyReLU(),
                        nn.Dropout(p=self.config.dropout),
                        nn.Linear(self.kg_emb_dim, self.kg_emb_dim))

            if 'attn' in self.config.XXXX_17:
                self.XXXX_14 = SelfAttentionSeq(self.kg_emb_dim,
                                                       self.kg_emb_dim)
            self.sample_mask_score = config.sample_mask_score

        if self.config.knn_score_type == 'gate':
            self.gate_layer = GateScore(self.kg_emb_dim, self.config.dropout)

    def get_key_vec(self,
                    batch,
                    mode,
                    nr_conv_key_vec='user_pref',
                    init_flag=False):

        cur_user_reps, item_reps = self.user_model.get_user_rep(batch, mode)

        if nr_conv_key_vec == 'rec_tgt':
            key_vec = item_reps[batch['movie_to_rec']]
        elif nr_conv_key_vec == 'user_pref':
            key_vec = cur_user_reps
        else:
            raise NotImplementedError

        return key_vec

    def update_step(self, batch, mode):
        key_vecs = self.get_key_vec(
            batch,
            mode,
            nr_conv_key_vec=self.config.neighboring_conv_key_vec)
        rec_labels = batch["movie_to_rec"]
        context_entities = batch['context_entities']
        conv_ids = batch['conv_ids']

        conv_example_entities = {}
        for _conv_id, _key_vec, _entity, _rec_label in zip(conv_ids, key_vecs.clone().detach(), context_entities, rec_labels):
            if len(_entity) > 0:
                if (_conv_id not in conv_example_entities) or (
                        _conv_id in conv_example_entities and
                        conv_example_entities[_conv_id] < len(_entity)):
                    conv_example_entities[_conv_id] = len(_entity)
                else:
                    continue

                _rec_label_str = str(_rec_label.item())
                if _rec_label_str in _conv_id:
                    conv_item_dict_key = _conv_id.split('_')[0]
                else:
                    conv_item_dict_key = _conv_id

                if conv_item_dict_key not in self.conv_item_dict:
                    self.conv_item_dict[conv_item_dict_key] = [
                        _key_vec, _entity, _rec_label]
                else:
                    _old_key_vec, _old_entities, _ = self.conv_item_dict[conv_item_dict_key]
                    logger.info(
                        f'{conv_item_dict_key} exist in collection, old {len(_old_entities)}, new {len(_entity)}'
                    )
                    self.conv_item_dict[conv_item_dict_key] = [
                        _key_vec, _entity, _rec_label
                    ]

    def get_nr_conv_reps(self, batch_k_neighbor, mode):
        assert self.vknn is not None
        prev_entities = [vecs[1] for vecs in self.conv_item_dict.values()]
        prev_rec_labels = torch.stack(
            [vecs[2] for vecs in self.conv_item_dict.values()]).long()
        unique_conv_ids = sorted(
            list(set(batch_k_neighbor.reshape(-1).tolist())))
        involved_convs = [prev_entities[ucid] for ucid in unique_conv_ids]
        nr_conv_reps, item_reps = self.user_model.get_user_rep(
            {'context_entities': involved_convs}, mode)
        cands_kg_user_rep = torch.zeros_like(item_reps)
        cands_kg_user_rep[unique_conv_ids] += nr_conv_reps
        return item_reps[prev_rec_labels][batch_k_neighbor], cands_kg_user_rep[
            batch_k_neighbor]

    def forward(self, batch, mode, epoch=-1, XXXX_31=None):
        add_loss = 0.0

        if self.config.neighboring_convs:
            cur_user_reps, item_reps = self.user_model.get_user_rep(
                batch, mode)
            ignore_k_neighbor = [conv_id.split(
                '_')[0] for conv_id in batch['conv_ids']]

            pos_batch_k_neighbor = self.get_k_neighbor(
                cur_user_reps, ignore_k_neighbor=ignore_k_neighbor)

            b_k_nb_tgt_reps, b_k_nr_conv_reps = self.get_nr_conv_reps(
                pos_batch_k_neighbor, mode)

            XXXX_34 = self.gate_layer(
                cur_user_reps.unsqueeze(1).expand_as(b_k_nr_conv_reps),
                b_k_nr_conv_reps).squeeze(2)

            if not self.config.XXXX_32:
                XXXX_18 = self.XXXX_29(
                    batch, pos_batch_k_neighbor, XXXX_34, item_reps)
            else:
                XXXX_18 = 0.0

            if not self.config.XXXX_33:
                XXXX_25 = self.XXXX_27(
                    b_k_nr_conv_reps, b_k_nb_tgt_reps, XXXX_34, item_reps)
            else:
                XXXX_25 = 0.0

            if not self.config.XXXX_33 and not self.config.XXXX_32:
                XXXX_35 = self.config.XXXX_28 * \
                    XXXX_18 + (1 - self.config.XXXX_28) * XXXX_25
            else:
                XXXX_35 = XXXX_18 + XXXX_25
            wandb.log({
                'relevance_score': XXXX_34.mean().item(),
            })

            if type(XXXX_18 + XXXX_25) != float:
                XXXX_36 = torch.mm(XXXX_35, item_reps)
            else:
                XXXX_36 = None
        else:
            XXXX_36 = None
            XXXX_35 = None
            item_reps = None

        return XXXX_36, XXXX_35, {}, item_reps, add_loss

    def XXXX_27(self, b_k_nr_conv_reps, b_k_nb_tgt_reps,
                     XXXX_34, item_rep):

        if self.config.XXXX_17 in ['pre_rec_attn', 'pre_rec_avg', 'pre_rec_sum']:
            hybrid_vecs = self.sim_crs_norm(
                torch.cat([b_k_nr_conv_reps, b_k_nb_tgt_reps], dim=-1))
        elif self.config.XXXX_17 in ['pre_attn', 'pre_avg', 'pre_sum']:
            hybrid_vecs = self.sim_crs_norm(b_k_nr_conv_reps)

        if 'attn' in self.config.XXXX_17:
            attn_mask = XXXX_34 < self.sample_mask_score
            attn_conda_user_reps = self.XXXX_14(hybrid_vecs, attn_mask)
            XXXX_25 = torch.mm(F.normalize(attn_conda_user_reps, p=1, dim=1),
                                F.normalize(item_rep, p=1, dim=1).t())
        elif 'avg' in self.config.XXXX_17:
            expand_item_ep = item_rep.t().unsqueeze(0).expand(
                hybrid_vecs.size(0), self.kg_emb_dim, self.config.n_entity)
            XXXX_25 = (torch.bmm(F.normalize(hybrid_vecs, p=1, dim=2),
                                  F.normalize(expand_item_ep, p=1, dim=1)) *
                        XXXX_34.unsqueeze(2)).mean(1)
        elif 'sum' in self.config.XXXX_17:
            expand_item_ep = item_rep.t().unsqueeze(0).expand(
                hybrid_vecs.size(0), self.kg_emb_dim, self.config.n_entity)
            XXXX_25 = (torch.bmm(F.normalize(hybrid_vecs, p=1, dim=2),
                                  F.normalize(expand_item_ep, p=1, dim=1)) *
                        XXXX_34.unsqueeze(2)).sum(1)
        else:
            raise NotImplementedError

        correlation_mask = torch.zeros(b_k_nr_conv_reps.size(0),
                                       self.config.n_entity,
                                       device=XXXX_34.device)
        correlation_mask[:, self.config.cand_movies_ids] = 1

        return XXXX_25 * correlation_mask

    def XXXX_24(self, item_reps, XXXX_18, sim_entities,
                              para_alpha):
        assert sim_entities.size() == para_alpha.size()
        gpu_device = XXXX_18.device
        cand_movies_ids_tensor = torch.tensor(
            self.config.cand_movies_ids, device=gpu_device).long()
        a_b_cos_sim_scores = sim_matrix(
            item_reps[sim_entities].reshape(-1, self.kg_emb_dim), item_reps[cand_movies_ids_tensor])
        a_b_cos_sim_scores = a_b_cos_sim_scores.reshape(
            sim_entities.size(0), -1, len(self.config.cand_movies_ids))
        if self.config.entity_item_topk > 0:
            a_b_cos_sim_scores
            topkvals, _ = torch.topk(a_b_cos_sim_scores,
                                     self.config.entity_item_topk,
                                     dim=2)
            topk_lower_bound, _ = topkvals.min(2)
            pos_idx = a_b_cos_sim_scores >= topk_lower_bound.unsqueeze(2)
            a_b_cos_sim_scores = pos_idx * a_b_cos_sim_scores

        if self.config.entity_item_op == 'sum':
            XXXX_23 = (a_b_cos_sim_scores *
                             para_alpha.unsqueeze(2)).sum(1)
        elif self.config.entity_item_op == 'mean':
            XXXX_23 = (a_b_cos_sim_scores *
                             para_alpha.unsqueeze(2)).mean(1)
        else:
            raise NotImplementedError

        # B X N
        XXXX_18[:, self.config.cand_movies_ids] += XXXX_23

    def XXXX_29(self, batch, batch_k_neighbor, XXXX_34, item_reps):
        gpu_device = XXXX_34.device
        XXXX_18 = torch.zeros(batch_k_neighbor.size(0),
                                self.config.n_entity,
                                device=gpu_device)
        prev_entities = [vecs[1] for vecs in self.conv_item_dict.values()]
        prev_rec_labels = torch.stack([
            vecs[2] for vecs in self.conv_item_dict.values()
        ]).long().to(gpu_device)

        if 'context_entities_kbrd' in batch:
            context_kb_entities = batch['context_entities_kbrd']
        else:
            context_kb_entities = batch['context_entities']

        XXXX_19 = XXXX_18.clone()
        XXXX_20 = XXXX_18.clone()
        XXXX_21 = XXXX_18.clone()
        XXXX_22 = XXXX_18.clone()

        max_len = max([len(entities) for entities in context_kb_entities])
        row_index = torch.stack(
            [torch.arange(XXXX_34.size(0), device=gpu_device)] * max_len).t()

        pad_context_kb_entities = torch.tensor([entities + [0] * (max_len - len(
            entities)) for entities in context_kb_entities], device=gpu_device).long()

        a_linear_index = row_index * self.config.n_entity + pad_context_kb_entities
        XXXX_19.put_(a_linear_index, torch.ones_like(
            a_linear_index, device=gpu_device).float(), accumulate=True)
        XXXX_19[:, 0] = 0

        prev_entities_max_len = max([len(entities)
                                    for entities in prev_entities])
        pad_prev_entities = torch.tensor([entities + [0] * (prev_entities_max_len - len(
            entities)) for entities in prev_entities], device=gpu_device).long()
        col_index = torch.cat([pad_prev_entities[batch_k_neighbor],
                              prev_rec_labels[batch_k_neighbor].unsqueeze(2)], dim=2)
        ep_knn_scores = XXXX_34.unsqueeze(2).expand_as(col_index)

        c_row_index = torch.stack([torch.arange(XXXX_34.size(0), device=gpu_device)] * (
            prev_entities_max_len + 1)).t().unsqueeze(1).expand_as(col_index)
        c_linear_index = c_row_index * self.config.n_entity + col_index
        XXXX_21.put_(c_linear_index, ep_knn_scores, accumulate=True)
        XXXX_21[:, 0] = 0

        self.XXXX_24(item_reps, XXXX_20, pad_context_kb_entities, (
            pad_context_kb_entities != 0) * torch.ones_like(pad_context_kb_entities, device=gpu_device).float())
        XXXX_20[:, 0] = 0

        batch_size = batch_k_neighbor.size(0)
        self.XXXX_24(item_reps, XXXX_22, col_index.reshape(
            batch_size, -1), (col_index.reshape(batch_size, -1) != 0) * ep_knn_scores.reshape(batch_size, -1))
        XXXX_22[:, 0] = 0

        correlation_mask = torch.zeros(batch_k_neighbor.size(
            0), self.config.n_entity, device=gpu_device)
        correlation_mask[:, self.config.cand_movies_ids] = 1

        XXXX_18 = XXXX_19 + XXXX_20 + XXXX_21 + XXXX_22

        return XXXX_18 * correlation_mask

    def get_item_reps(self):
        if torch.cuda.is_available():
            item_reps = self.user_model.kGModel.kg_encoder(
                None, self.config.edge_idx.cuda(),
                self.config.edge_type.cuda())
        else:
            item_reps = self.user_model.kGModel.kg_encoder(
                None, self.config.edge_idx, self.config.edge_type)

        return item_reps

    def init_vknn_index(self):
        item_reps = self.get_item_reps()
        self.vknn = vknn_conv_item(
            self.conv_item_dict,
            self.config.sample_nlist,
            self.config.sample_nlist,
            item2vec_embs=item_reps)
        logger.info('finish initializing vknn index')

    def get_k_neighbor(self, cur_key_vecs, ignore_k_neighbor=None):
        ignore_k_neighbor_idx = []
        for conv_id in ignore_k_neighbor:
            if conv_id in list(self.conv_item_dict.keys()):
                ignore_k_neighbor_idx.append(
                    list(self.conv_item_dict.keys()).index(conv_id))
            else:
                ignore_k_neighbor_idx.append(999999)

        return self.vknn.search(cur_key_vecs.detach(),
                                self.sample_num,
                                ignore_k_neighbor_idx=ignore_k_neighbor_idx)

    def save(self, path):
        if len(self.conv_item_dict.values()) > 10:
            with open(path, 'wb') as fp:
                pickle.dump(self.conv_item_dict, fp)
                print(
                    f'Save {len(self.conv_item_dict.values())} XXXX_15 conversations to [{path}]')

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.conv_item_dict = CPU_Unpickler(f).load()
                self.init_vknn_index()
            print(
                f'Restore {len(self.conv_item_dict.values())} XXXX_15 conversations from [{path}]')
        else:
            print(f'{path} doesnt exist!!!')
