import math
import numpy as np
from parse import args


def load_file(file_path):
    m_item, all_pos = 0, []

    with open(file_path, "r") as f:
        for line in f.readlines():
            pos = list(map(int, line.rstrip().split(' ')))[1:]
            if pos:
                m_item = max(m_item, max(pos) + 1)
            all_pos.append(pos)

    return m_item, all_pos


def load_dataset(path):
    m_item = 0
    m_item_, all_train_ind = load_file(path + "train.txt")
    m_item = max(m_item, m_item_)
    m_item_, all_test_ind = load_file(path + "test.txt")
    m_item = max(m_item, m_item_)

    if args.part_percent > 0:
        _, part_train_ind = load_file(path + "train.part-{}%.txt".format(args.part_percent))
    else:
        part_train_ind = []

    items_popularity = np.zeros(m_item)
    for items in all_train_ind:
        for item in items:
            items_popularity[item] += 1
    for items in all_test_ind:
        for item in items:
            items_popularity[item] += 1

    return m_item, all_train_ind, all_test_ind, part_train_ind, items_popularity


def sample_part_of_dataset(path, ratio):
    _, all_pos = load_file(path + "train.txt")
    with open(path + "train.part-{}%.txt".format(int(ratio * 100)), "w") as f:
        for user, pos_items in enumerate(all_pos):
            part_pos_items = np.random.choice(pos_items, math.ceil(len(pos_items) * ratio), replace=False).tolist()
            f.write(str(user))
            for item in part_pos_items:
                f.write(' ' + str(item))
            f.write('\n')
