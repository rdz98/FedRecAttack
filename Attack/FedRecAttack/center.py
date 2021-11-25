import torch
import numpy as np
import torch.nn as nn
from parse import args


class FedRecAttackCenter(nn.Module):
    def __init__(self, users_train_ind, target_items, m_item, dim):
        super().__init__()
        self.users_train_ind = users_train_ind
        self.target_items = target_items
        self.n_user = len(users_train_ind)
        self.m_item = m_item
        self.dim = dim

        self._opt = None
        self.items_emb = None
        self.users_emb = None
        self.items_emb_grad = None

        self.users_emb_ = nn.Embedding(self.n_user, dim)
        nn.init.normal_(self.users_emb_.weight, std=0.01)

    @property
    def opt(self):
        if self._opt is None:
            self._opt = torch.optim.Adam([self.users_emb_.weight], lr=args.attack_lr)
        return self._opt

    def loss(self, batch_users):
        users, pos_items, neg_items = [], [], []
        for user in batch_users:
            for pos_item in self.users_train_ind[user]:
                users.append(user)
                pos_items.append(pos_item)
                neg_item = np.random.randint(self.m_item)
                while neg_item in self.users_train_ind[user]:
                    neg_item = np.random.randint(self.m_item)
                neg_items.append(neg_item)
        users = torch.Tensor(users).long().to(args.device)
        pos_items = torch.Tensor(pos_items).long().to(args.device)
        neg_items = torch.Tensor(neg_items).long().to(args.device)

        users_emb = self.users_emb_(users)
        pos_items_emb = self.items_emb[pos_items]
        neg_items_emb = self.items_emb[neg_items]

        pos_scores = torch.sum(users_emb * pos_items_emb, dim=-1)
        neg_scores = torch.sum(users_emb * neg_items_emb, dim=-1)
        loss = -(pos_scores - neg_scores).sigmoid().log().sum()
        return loss

    def train_(self, new_items_emb, batch_size, eps=1e-4):
        new_items_emb = new_items_emb.clone().detach()
        if (self.items_emb is not None) and ((new_items_emb - self.items_emb).abs().sum() < eps):
            return
        self.items_emb = new_items_emb

        if len(self.users_train_ind):
            rand_users = np.arange(self.n_user)
            np.random.shuffle(rand_users)
            total_loss = 0.
            for i in range(0, len(rand_users), batch_size):
                loss = self.loss(rand_users[i: i + batch_size])
                total_loss += loss.cpu().item()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            total_loss /= len(rand_users)
        else:
            nn.init.normal_(self.users_emb_.weight, std=0.1)
        self.users_emb = self.users_emb_.weight.clone().detach()

        rand_users = np.arange(self.n_user)
        ignore_users, ignore_items = [], []
        for idx, user in enumerate(rand_users):
            for item in self.target_items:
                if item not in self.users_train_ind[user]:
                    ignore_users.append(idx)
                    ignore_items.append(item)
            for item in self.users_train_ind[user]:
                ignore_users.append(idx)
                ignore_items.append(item)
        ignore_users = torch.Tensor(ignore_users).long().to(args.device)
        ignore_items = torch.Tensor(ignore_items).long().to(args.device)

        with torch.no_grad():
            users_emb = self.users_emb[torch.Tensor(rand_users).long().to(args.device)]
            items_emb = self.items_emb
            scores = torch.matmul(users_emb, items_emb.t())
            scores[ignore_users, ignore_items] = - (1 << 10)
            _, top_items = torch.topk(scores, 10)
        top_items = top_items.cpu().tolist()

        users, pos_items, neg_items = [], [], []
        for idx, user in enumerate(rand_users):
            for item in self.target_items:
                if item not in self.users_train_ind[user]:
                    users.append(user)
                    pos_items.append(item)
                    neg_items.append(top_items[idx].pop())
        users = torch.Tensor(users).long().to(args.device)
        pos_items = torch.Tensor(pos_items).long().to(args.device)
        neg_items = torch.Tensor(neg_items).long().to(args.device)
        users_emb = self.users_emb[users]
        items_emb = self.items_emb.clone().detach().requires_grad_(True)
        pos_items_emb = items_emb[pos_items]
        neg_items_emb = items_emb[neg_items]
        pos_scores = torch.sum(users_emb * pos_items_emb, dim=-1)
        neg_scores = torch.sum(users_emb * neg_items_emb, dim=-1)
        loss = neg_scores - pos_scores
        loss[loss < 0] = torch.exp(loss[loss < 0]) - 1
        loss = loss.sum()
        loss.backward()
        self.items_emb_grad = items_emb.grad
