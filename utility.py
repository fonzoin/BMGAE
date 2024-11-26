import torch
import os
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader


def print_statistics(graph, string):
    print('=' * 10 + string + '=' * 10)
    print('Average interactions:', graph.sum(1).mean(0).item())
    nonzero_rows, nonzero_cols = graph.nonzero()
    unique_nonzero_rows = np.unique(nonzero_rows)
    unique_nonzero_cols = np.unique(nonzero_cols)
    print('Non-zero rows:', len(unique_nonzero_rows) / graph.shape[0])
    print('Non-zero columns:', len(unique_nonzero_cols) / graph.shape[1])
    print('Matrix density:', len(nonzero_rows) / (graph.shape[0] * graph.shape[1]))


class TrainDataset(Dataset):
    def __init__(self, u_b_pairs, u_b_graph, num_bundles, neg_sample=1):
        self.u_b_pairs = u_b_pairs
        self.u_b_graph = u_b_graph
        self.num_bundles = num_bundles
        self.neg_sample = neg_sample

    def __getitem__(self, index):
        user, pos_bundle = self.u_b_pairs[index]
        pos_and_neg_bundles = [pos_bundle]

        while True:
            i = np.random.randint(self.num_bundles)
            if self.u_b_graph[user, i] == 0 and i not in pos_and_neg_bundles:
                pos_and_neg_bundles.append(i)
                if len(pos_and_neg_bundles) == self.neg_sample + 1:
                    break

        return torch.LongTensor([user]), torch.LongTensor(pos_and_neg_bundles)

    def __len__(self):
        return len(self.u_b_pairs)


class TestDataset(Dataset):
    def __init__(self, u_b_graph, u_b_graph_train):
        self.u_b_graph = u_b_graph
        self.u_b_graph_train = u_b_graph_train

    def __getitem__(self, index):
        u_b_ground_truth = torch.from_numpy(self.u_b_graph[index].toarray()).squeeze()
        u_b_train_data = torch.from_numpy(self.u_b_graph_train[index].toarray()).squeeze()

        return index, u_b_ground_truth, u_b_train_data

    def __len__(self):
        return self.u_b_graph.shape[0]


class Datasets:
    def __init__(self, conf):
        self.path = conf['data_path']
        self.name = conf['dataset']
        batch_size_train = conf['batch_size_train']
        batch_size_test = conf['batch_size_test']

        self.num_users, self.num_bundles, self.num_items = self.get_data_size()

        b_i_pairs, b_i_graph = self.get_bi_graph()
        u_i_pairs, u_i_graph = self.get_ui_graph()
        u_b_pairs_train, u_b_graph_train = self.get_ub_graph("train")
        u_b_pairs_val, u_b_graph_val = self.get_ub_graph("val")
        u_b_pairs_test, u_b_graph_test = self.get_ub_graph("test")

        b_i_dict = dict()
        for i in b_i_pairs:
            if i[0] not in b_i_dict:
                b_i_dict[i[0]] = []
            if i[1] not in b_i_dict[i[0]]:
                b_i_dict[i[0]].append(i[1])
        self.b_i_dict = b_i_dict

        self.train_data = TrainDataset(u_b_pairs_train, u_b_graph_train, self.num_bundles, conf["neg_num"])
        self.val_data = TestDataset(u_b_graph_val, u_b_graph_train)
        self.test_data = TestDataset(u_b_graph_test, u_b_graph_train)

        self.graphs = [u_b_graph_train, u_i_graph, b_i_graph]

        self.train_loader = DataLoader(self.train_data, batch_size=batch_size_train, shuffle=True, num_workers=12,
                                       drop_last=True)
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size_test, shuffle=False, num_workers=12)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size_test, shuffle=False, num_workers=12)

    def get_data_size(self):
        name = self.name
        if "_" in name:
            name = name.split("_")[0]
        with open(os.path.join(self.path, self.name, '{}_data_size.txt'.format(name)), 'r') as f:
            return [int(s) for s in f.readline().split('\t')][:3]

    def get_bi_graph(self):
        with open(os.path.join(self.path, self.name, 'bundle_item.txt'), 'r') as f:
            b_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indices = np.array(b_i_pairs, dtype=np.int32)
        values = np.ones(len(b_i_pairs), dtype=np.float32)
        b_i_graph = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])),
                                  shape=(self.num_bundles, self.num_items)).tocsr()
        print_statistics(b_i_graph, 'B-I statistics')
        return b_i_pairs, b_i_graph

    def get_ui_graph(self):
        with open(os.path.join(self.path, self.name, 'user_item.txt'), 'r') as f:
            u_i_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indices = np.array(u_i_pairs, dtype=np.int32)
        values = np.ones(len(u_i_pairs), dtype=np.float32)
        u_i_graph = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])),
                                  shape=(self.num_users, self.num_items)).tocsr()
        print_statistics(u_i_graph, 'U-I statistics')
        return u_i_pairs, u_i_graph

    def get_ub_graph(self, task):
        with open(os.path.join(self.path, self.name, 'user_bundle_{}.txt'.format(task)), 'r') as f:
            u_b_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indices = np.array(u_b_pairs, dtype=np.int32)
        values = np.ones(len(u_b_pairs), dtype=np.float32)
        u_b_graph = sp.coo_matrix((values, (indices[:, 0], indices[:, 1])),
                                  shape=(self.num_users, self.num_bundles)).tocsr()
        print_statistics(u_b_graph, "U-B statistics in %s" % task)
        return u_b_pairs, u_b_graph
