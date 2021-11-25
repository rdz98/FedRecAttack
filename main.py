import torch
import random
import numpy as np
from time import time
from parse import args
from data import load_dataset

from FedRec.server import FedRecServer
from FedRec.client import FedRecClient


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s " % args_str)

    t0 = time()
    m_item, all_train_ind, all_test_ind, part_train_ind, items_popularity = load_dataset(args.path + args.dataset)
    target_items = np.random.choice(m_item, 1, replace=False).tolist()

    server = FedRecServer(m_item, args.dim).to(args.device)
    clients = []
    for train_ind, test_ind in zip(all_train_ind, all_test_ind):
        clients.append(
            FedRecClient(train_ind, test_ind, target_items, m_item, args.dim).to(args.device)
        )

    malicious_clients_limit = int(len(clients) * args.clients_limit)
    if args.attack == 'FedRecAttack':
        from Attack.FedRecAttack.center import FedRecAttackCenter
        from Attack.FedRecAttack.client import FedRecAttackClient

        attack_center = FedRecAttackCenter(part_train_ind, target_items, m_item, args.dim).to(args.device)
        for _ in range(malicious_clients_limit):
            clients.append(FedRecAttackClient(attack_center, args.items_limit))

    else:
        from Attack.baseline import BaselineAttackClient

        if args.attack == 'Random':
            for _ in range(malicious_clients_limit):
                train_ind = [i for i in target_items]
                for __ in range(args.items_limit // 2 - len(target_items)):
                    item = np.random.randint(m_item)
                    while item in train_ind:
                        item = np.random.randint(m_item)
                    train_ind.append(item)
                clients.append(BaselineAttackClient(train_ind, [], [], m_item, args.dim).to(args.device))
        elif args.attack == 'Popular':
            for i in target_items:
                items_popularity[i] = 1e10
            _, train_ind = torch.Tensor(items_popularity).topk(args.items_limit // 2)
            train_ind = train_ind.numpy().tolist()
            for _ in range(malicious_clients_limit):
                clients.append(BaselineAttackClient(train_ind, [], [], m_item, args.dim).to(args.device))
        elif args.attack == 'Bandwagon':
            for i in target_items:
                items_popularity[i] = - 1e10
            items_limit = args.items_limit // 2
            _, popular_items = torch.Tensor(items_popularity).topk(m_item // 10)
            popular_items = popular_items.numpy().tolist()

            for _ in range(malicious_clients_limit):
                train_ind = [i for i in target_items]
                train_ind += np.random.choice(popular_items, int(items_limit * 0.1), replace=False).tolist()
                rest_items = []
                for i in range(m_item):
                    if i not in train_ind:
                        rest_items.append(i)
                train_ind += np.random.choice(rest_items, items_limit - len(train_ind), replace=False).tolist()
                clients.append(BaselineAttackClient(train_ind, [], [], m_item, args.dim).to(args.device))
        else:
            print('Unknown args --attack.')
            return

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time() - t0, len(clients), m_item,
           sum([len(i) for i in all_train_ind]),
           sum([len(i) for i in all_test_ind])))
    print("Target items: %s." % str(target_items))
    print("output format: ({Sampled HR@10}), ({ER@5},{ER@10},{NDCG@10})")

    # Init performance
    t1 = time()
    with torch.no_grad():
        test_result, target_result = server.eval_(clients)
    print("Iteration 0(init), (%.4f) on test" % tuple(test_result) +
          ", (%.4f, %.4f, %.4f) on target." % tuple(target_result) +
          " [%.1fs]" % (time() - t1))

    try:
        for epoch in range(1, args.epochs + 1):
            t1 = time()
            rand_clients = np.arange(len(clients))
            np.random.shuffle(rand_clients)

            total_loss = []
            for i in range(0, len(rand_clients), args.batch_size):
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                loss = server.train_(clients, batch_clients_idx)
                total_loss.extend(loss)
            total_loss = np.mean(total_loss).item()

            t2 = time()
            with torch.no_grad():
                test_result, target_result = server.eval_(clients)
            print("Iteration %d, loss = %.5f [%.1fs]" % (epoch, total_loss, t2 - t1) +
                  ", (%.4f) on test" % tuple(test_result) +
                  ", (%.4f, %.4f, %.4f) on target." % tuple(target_result) +
                  " [%.1fs]" % (time() - t2))
    except KeyboardInterrupt:
        pass


setup_seed(20211111)

if __name__ == "__main__":
    main()
