import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model


class XAttn(nn.Module):
    def __init__(self, n_head, d_in, d_key, norm_flag):
        super(XAttn, self).__init__()
        self.n_head = n_head
        self.d_in = d_in
        self.temperature = d_key ** 0.5

        self.w_h = nn.Linear(d_in, n_head * d_in, bias=False)
        self.w_r = nn.Linear(d_in, n_head * d_in, bias=False)

        self.k_h = nn.Linear(d_in, d_key, bias=False)
        self.k_r = nn.Linear(d_in, d_key, bias=False)

        nn.init.xavier_uniform_(self.w_h.weight.data)
        nn.init.xavier_uniform_(self.w_r.weight.data)
        nn.init.xavier_uniform_(self.k_h.weight.data)
        nn.init.xavier_uniform_(self.k_r.weight.data)

        self.norm = norm_flag

        if not norm_flag:
            self.ln = nn.LayerNorm(d_in)

    def forward(self, h_emb, r_emb, mode):
        residual = h_emb
        hyper_batch = h_emb.size(0)
        if mode == 'normal':
            h_batch_size = hyper_batch
            r_batch_size = r_emb.size(0)
        else:
            h_batch_size = h_emb.size(1)
            r_batch_size = h_batch_size
        h1_emb = self.w_h(h_emb).view(-1, h_batch_size, self.n_head, self.d_in)
        r1_emb = self.w_r(r_emb).view(-1, r_batch_size, self.n_head, self.d_in)
        h1_key = self.k_h(h1_emb)
        r1_key = self.k_r(r1_emb)
        xattn = torch.matmul(h1_key / self.temperature, r1_key.transpose(-2, -1))
        xattn = xattn.view(-1, h_batch_size, self.n_head * self.n_head)
        xattn = F.softmax(xattn, dim=-1)
        xattn = xattn.view(-1, h_batch_size, self.n_head, self.n_head)
        h_xattn = torch.sum(xattn, dim=-1).unsqueeze(-1)
        r_xattn = torch.sum(xattn, dim=-2).unsqueeze(-1)
        t_emb = h1_emb * h_xattn + r1_emb * r_xattn
        t_emb = torch.sum(t_emb, dim=-2)
        t_emb += residual
        if not self.norm:
            t_emb = self.ln(t_emb)
        return t_emb


class TransX(Model):
    def __init__(self, ent_tot, rel_tot, dim=100, n_head=4, d_key=50, p_norm=1, norm_flag=True, margin=None,
                 epsilon=None):
        super(TransX, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

        # add
        self.h_attn = XAttn(n_head=n_head, d_in=dim, d_key=d_key, norm_flag=norm_flag)
        self.t_attn = XAttn(n_head=n_head, d_in=dim, d_key=d_key, norm_flag=norm_flag)

        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _calc(self, h, t, r, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']


        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)

        # add

        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])

        h = self.h_attn(h, r, mode)
        t = self.t_attn(t, r, mode)

        score = self._calc(h, t, r, mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()