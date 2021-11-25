import torch
import numpy as np
from parse import args


class FedRecAttackClient(object):
    def __init__(self, center, train_all_size):
        self._center = center
        self._train_all_size = train_all_size
        self.train_all = None

    def eval_(self, _items_emb):
        return None, None

    @staticmethod
    def noise(shape, std):
        noise = np.random.multivariate_normal(
            mean=np.zeros(shape[1]), cov=np.eye(shape[1]) * std, size=shape[0]
        )
        return torch.Tensor(noise).to(args.device)

    def train_(self, items_emb, std=1e-7):
        self._center.train_(items_emb, args.attack_batch_size)

        with torch.no_grad():
            items_emb_grad = self._center.items_emb_grad

            if self.train_all is None:
                target_items = self._center.target_items
                target_items_ = torch.Tensor(target_items).long().to(args.device)

                p = (items_emb_grad + self.noise(items_emb_grad.shape, std)).norm(2, dim=-1)
                p[target_items_] = 0.
                p = (p / p.sum()).cpu().numpy()
                rand_items = np.random.choice(
                    np.arange(len(p)), self._train_all_size - len(target_items), replace=False, p=p
                ).tolist()

                self.train_all = torch.Tensor(target_items + rand_items).long().to(args.device)

            items_emb_grad = items_emb_grad[self.train_all]
            items_emb_grad_norm = items_emb_grad.norm(2, dim=-1, keepdim=True)
            grad_max = args.grad_limit
            too_large = items_emb_grad_norm[:, 0] > grad_max
            items_emb_grad[too_large] /= (items_emb_grad_norm[too_large] / grad_max)
            items_emb_grad += self.noise(items_emb_grad.shape, std)

            self._center.items_emb_grad[self.train_all] -= items_emb_grad
        return self.train_all, items_emb_grad, None
