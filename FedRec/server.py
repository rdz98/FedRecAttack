import torch
import torch.nn as nn
from parse import args


class FedRecServer(nn.Module):
    def __init__(self, m_item, dim):
        super().__init__()
        self.m_item = m_item
        self.dim = dim
        self.items_emb = nn.Embedding(m_item, dim)
        nn.init.normal_(self.items_emb.weight, std=0.01)

    def train_(self, clients, batch_clients_idx):
        batch_loss = []
        batch_items_emb_grad = torch.zeros_like(self.items_emb.weight)

        for idx in batch_clients_idx:
            client = clients[idx]
            items, items_emb_grad, loss = client.train_(self.items_emb.weight)

            with torch.no_grad():
                norm = items_emb_grad.norm(2, dim=-1, keepdim=True)
                too_large = norm[:, 0] > args.grad_limit
                items_emb_grad[too_large] /= (norm[too_large] / args.grad_limit)
                batch_items_emb_grad[items] += items_emb_grad

            if loss is not None:
                batch_loss.append(loss)

        with torch.no_grad():
            self.items_emb.weight.data.add_(batch_items_emb_grad, alpha=-args.lr)
        return batch_loss

    def eval_(self, clients):
        test_cnt = 0
        test_results = 0.
        target_cnt = 0
        target_results = 0.

        for client in clients:
            test_result, target_result = client.eval_(self.items_emb.weight)
            if test_result is not None:
                test_cnt += 1
                test_results += test_result
            if target_result is not None:
                target_cnt += 1
                target_results += target_result
        return test_results / test_cnt, target_results / target_cnt
