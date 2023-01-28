# Some functions are borrowed from C2CRS

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from scipy import stats


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class EuclideanDistance(nn.Module):
    __constants__ = ['norm', 'eps', 'keepdim']
    norm: float
    eps: float
    keepdim: bool

    def __init__(self,
                 p: float = 2.,
                 eps: float = 1e-6,
                 keepdim: bool = False) -> None:
        super(EuclideanDistance, self).__init__()
        self.norm = p
        self.eps = eps
        self.keepdim = keepdim

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        dis = F.pairwise_distance(
            x1, x2, self.norm, self.eps, self.keepdim) + self.eps
        dis /= dis.max(0, keepdim=True)[0]
        return 1 - dis


class PearsonCorrelation(nn.Module):
    def __init__(self):
        super(PearsonCorrelation, self).__init__()

    def get_corr(self, x, y):
        new_corrs = stats.pearsonr(x.detach().numpy(), y.detach().numpy())[0]
        return abs(new_corrs)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        device = x1.device
        x1 = x1.cpu()
        x2 = x2.cpu()
        corrs = torch.Tensor([self.get_corr(x, y)
                             for x, y in zip(x1, x2)]).to(device)
        corrs = (corrs > 0) * corrs + (~(corrs > 0)) * torch.tensor(0.001)
        return corrs


def pad_nested_sequences(sequences,
                         max_sent_len=1,
                         max_word_len=10,
                         dtype='int32'):

    for sent in sequences:
        max_sent_len = max(len(sent), max_sent_len)
        for word in sent:
            max_word_len = max(len(word), max_word_len)

    x = np.zeros((len(sequences), max_sent_len, max_word_len)).astype(dtype)
    for i, sent in enumerate(sequences):
        for j, word in enumerate(sent):
            if j < max_word_len:
                x[i, j, :len(word)] = word

    return x


def pad_nested_sequence(sequence,
                        max_sent_len=1,
                        dtype='int32'):

    for sent in sequence:
        max_sent_len = max(len(sent), max_sent_len)

    x = np.zeros((len(sequence), max_sent_len)).astype(dtype)
    for i, sent in enumerate(sequence):
        for j, word in enumerate(sent):
            x[i, j] = word

    return x


class LabelSmoothingLossCanonical(nn.Module):
    def __init__(self, smoothing=0.0, dim=-1):
        super(LabelSmoothingLossCanonical, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            true_dist += self.smoothing / pred.size(self.dim)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target):
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)


class ScaleSigmoid(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        input /= input.max(dim=1, keepdim=True)[0] * 5
        return torch.sigmoid(input)


class CustomMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=200, dropout=0.0) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class CustomResConMLP(nn.Module):
    def __init__(self, out_dim):
        super(CustomResConMLP, self).__init__()
        self.layer1 = nn.Linear(out_dim, out_dim)
        self.relu = nn.LeakyReLU()
        self.layer2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        # out = self.ln1(out)
        out = self.relu(out)
        out = self.layer2(out)
        # out = self.ln2(out)
        out += residual
        out = self.relu(out)
        return out


class SelfAttentionSeq(nn.Module):
    def __init__(self, dim, da, alpha=0.2, dropout=0.5):
        super(SelfAttentionSeq, self).__init__()
        self.dim = dim
        self.da = da
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(
            size=(self.dim, self.da)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(
            size=(self.da, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        nn.init.xavier_uniform_(self.b.data, gain=1.414)

    def forward(self, h, mask=None, return_logits=False):
        """
        For the padding tokens, its corresponding mask is True
        if mask==[1, 1, 1, ...]
        """
        # h: (batch, seq_len, dim), mask: (batch, seq_len)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.a)),
                         self.b)  # (batch, seq_len, 1)
        if mask is not None:
            full_mask = -1e30 * mask.float()
            # for all padding one, the mask=0
            batch_mask = torch.sum(
                (mask == False), -1).bool().float().unsqueeze(-1)
            mask = full_mask * batch_mask
            e += mask.unsqueeze(-1)
        attention = F.softmax(e, dim=1)  # (batch, seq_len, 1)
        # (batch, dim)
        if return_logits:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1), attention.squeeze(-1)
        else:
            return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1)


class GateLayer(nn.Module):
    def __init__(self, input_dim):
        super(GateLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 2, input_dim)
        self._norm_layer2 = nn.Linear(input_dim, 1)

    def forward(self, input1, input2):
        norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1))
        gate = torch.sigmoid(self._norm_layer2(norm_input))  # (bs, 1)
        gated_emb = gate * input1 + (1 - gate) * input2  # (bs, dim)
        return gated_emb


class GateScore(nn.Module):
    def __init__(self, input_dim, dropout=0.0):
        super(GateScore, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim * 2, input_dim)
        self._norm_layer2 = nn.Linear(input_dim, 1)
        self._dropout = nn.Dropout(p=dropout)
        self._relu = nn.LeakyReLU()

    def forward(self, input1, input2):
        norm_input = self._norm_layer1(torch.cat([input1, input2], dim=-1))
        norm_input = self._relu(norm_input)
        norm_input = self._dropout(norm_input)
        gate = torch.sigmoid(self._norm_layer2(norm_input))  # (bs, 1)
        return gate


class FusionLayer(nn.Module):
    def __init__(self, input_dim_1, input_dim_2):
        super(FusionLayer, self).__init__()
        self._norm_layer1 = nn.Linear(input_dim_1 + input_dim_2, 1)

    def forward(self, input1, input2):
        gate = torch.sigmoid(self._norm_layer1(
            torch.cat([input1, input2], dim=-1)))  # (bs, 1)
        gated_emb = gate * input1 + (1 - gate) * input2  # (bs, dim)
        return gated_emb


class ScaleGateLayer(nn.Module):
    def __init__(self, in_dim1, in_dim2, out_dim):
        super(ScaleGateLayer, self).__init__()
        self._layer1 = nn.Linear(in_dim1, out_dim)
        self._layer2 = nn.Linear(in_dim2, out_dim)
        self.relu_func = nn.LeakyReLU()
        self.gate_layer = GateLayer(out_dim)

    def forward(self, input1, input2):
        input1 = self.relu_func(self._layer1(input1))
        input2 = self.relu_func(self._layer2(input2))
        return self.gate_layer(input1, input2)


def edge_to_pyg_format(edge, type='RGCN'):
    if type == 'RGCN':
        edge_sets = torch.as_tensor(edge, dtype=torch.long)
        edge_idx = edge_sets[:, :2].t()
        edge_type = edge_sets[:, 2]
        return edge_idx, edge_type
    elif type == 'GCN':
        edge_set = [[co[0] for co in edge], [co[1] for co in edge]]
        return torch.as_tensor(edge_set, dtype=torch.long)
    else:
        raise NotImplementedError('type {} has not been implemented', type)


def matrix_cos_sim(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
