from FedRec.client import FedRecClient


class BaselineAttackClient(FedRecClient):
    def train_(self, items_emb, reg=0):
        a, b, _ = super().train_(items_emb, reg)
        return a, b, None

    def eval_(self, _items_emb):
        return None, None
