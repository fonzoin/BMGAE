import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
import scipy.sparse as sp


def matrix_to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph


def laplace_transform(graph):
    sqrt_row_sum = sp.diags(1 / (np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    sqrt_col_sum = sp.diags(1 / (np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = sqrt_row_sum @ graph @ sqrt_col_sum
    return graph


def get_bpr_loss(y):
    if y.shape[1] > 2:
        negs = y[:, 1:]
        pos = y[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = y[:, 1].unsqueeze(1)
        pos = y[:, 0].unsqueeze(1)
    loss = - torch.log(torch.sigmoid(pos - negs))
    loss = torch.mean(loss)
    return loss


def random_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1 - dropout_ratio])
    values = mask * values
    return values


class BMGAE(nn.Module):
    def __init__(self, conf, raw_graph, b_i_dict):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device
        self.embedding_size = conf["embedding_size"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]

        self.init_emb()

        masking_ratio = conf["masking_ratio"]
        item_mask_num = max(int(self.init_items_rep.shape[0] * masking_ratio), 1)
        self.item_masker = NodeMask(self.conf, masking_ratio, self.embedding_size, item_mask_num)
        bundle_mask_num = max(int(self.init_bundles_rep.shape[0] * masking_ratio), 1)
        self.bundle_masker = NodeMask(self.conf, masking_ratio, self.embedding_size, bundle_mask_num)

        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        self.b_i_dict = b_i_dict

        self.get_item_level_graph_wod()
        self.get_bundle_level_graph_wod()
        self.get_bundling_strategy_level_graph_wom()
        self.get_item_level_graph()
        self.get_bundle_level_graph()

        self.init_message_dropout()

        self.layer_num = self.conf["layer_num"]
        self.c_temp = self.conf["c_temp"]
        self.neg_sample = conf["neg_num"]

    def init_message_dropout(self):
        self.item_level_dropout = nn.Dropout(self.conf["u_i_aug_ratio"], True)
        self.bundle_level_dropout = nn.Dropout(self.conf["u_b_aug_ratio"], True)
        self.bundling_strategy_level_dropout = nn.Dropout(self.conf["b_i_aug_ratio"], True)

    def init_emb(self):
        self.init_users_rep = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.init_users_rep)
        self.init_bundles_rep = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.init_bundles_rep)
        self.init_items_rep = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.init_items_rep)

    def item_mask(self):
        masked_embeds, seeds = self.item_masker(self.init_items_rep)
        return masked_embeds, seeds

    def bundle_mask(self):
        masked_embeds, seeds = self.bundle_masker(self.init_bundles_rep)
        return masked_embeds, seeds

    def get_item_level_graph(self):
        ui_graph = self.ui_graph
        device = self.device
        dropout_ratio = self.conf["u_i_aug_ratio"]
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph],
                                    [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        if dropout_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = item_level_graph.tocoo()
                values = random_edge_dropout(graph.data, dropout_ratio)
                item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        self.item_level_graph = matrix_to_tensor(laplace_transform(item_level_graph)).to(device)

    def get_item_level_graph_wod(self):
        ui_graph = self.ui_graph
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph],
                                    [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        self.item_level_graph_wod = matrix_to_tensor(laplace_transform(item_level_graph)).to(device)

    def get_bundle_level_graph(self):
        ub_graph = self.ub_graph
        device = self.device
        dropout_ratio = self.conf["u_b_aug_ratio"]
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph],
                                      [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        if dropout_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = random_edge_dropout(graph.data, dropout_ratio)
                bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        self.bundle_level_graph = matrix_to_tensor(laplace_transform(bundle_level_graph)).to(device)

    def get_bundle_level_graph_wod(self):
        ub_graph = self.ub_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph],
                                      [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.bundle_level_graph_wod = matrix_to_tensor(laplace_transform(bundle_level_graph)).to(device)

    def get_bundling_strategy_level_graph_wom(self):
        bi_graph = self.bi_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((bi_graph.shape[0], bi_graph.shape[0])), bi_graph],
                                      [bi_graph.T, sp.csr_matrix((bi_graph.shape[1], bi_graph.shape[1]))]])
        self.bundling_strategy_level_graph_wom = matrix_to_tensor(laplace_transform(bundle_level_graph)).to(
            device)

    def graph_propagate(self, graph, rep1, rep2, md, test):
        reps = torch.cat((rep1, rep2), 0)
        all_reps = [reps]

        for i in range(self.layer_num):
            reps = torch.spmm(graph, reps)
            if self.conf["aug_type"] == "MD" and not test:
                reps = md(reps)

            reps = reps / (i + 2)
            all_reps.append(functional.normalize(reps, p=2, dim=1))

        all_reps = torch.stack(all_reps, 1)
        all_reps = torch.sum(all_reps, dim=1).squeeze(1)

        rep1, rep2 = torch.split(all_reps, (rep1.shape[0], rep2.shape[0]), 0)

        return rep1, rep2

    def propagate_train(self, masked_item_rep, masked_bundle_rep, test=False):
        if test:
            item_level_users_rep, item_level_items_rep = self.graph_propagate(self.item_level_graph_wod,
                                                                              self.init_users_rep, self.init_items_rep,
                                                                              self.item_level_dropout, test)
        else:
            item_level_users_rep, item_level_items_rep = self.graph_propagate(self.item_level_graph,
                                                                              self.init_users_rep, self.init_items_rep,
                                                                              self.item_level_dropout, test)

        if test:
            bundle_level_users_rep, bundle_level_bundles_rep = self.graph_propagate(self.bundle_level_graph_wod,
                                                                                    self.init_users_rep,
                                                                                    self.init_bundles_rep,
                                                                                    self.bundle_level_dropout, test)
        else:
            bundle_level_users_rep, bundle_level_bundles_rep = self.graph_propagate(self.bundle_level_graph,
                                                                                    self.init_users_rep,
                                                                                    self.init_bundles_rep,
                                                                                    self.bundle_level_dropout, test)
        bundling_strategy_level_bundles_rep, bundling_strategy_level_items_rep = self.graph_propagate(
            self.bundling_strategy_level_graph_wom, masked_bundle_rep, masked_item_rep,
            self.bundling_strategy_level_dropout, test)

        users_rep = [item_level_users_rep, bundle_level_users_rep]
        bundles_rep = [bundle_level_bundles_rep, bundling_strategy_level_bundles_rep]
        items_rep = [item_level_items_rep, bundling_strategy_level_items_rep]

        return users_rep, bundles_rep, items_rep

    def propagate_test(self, test=True):
        if test:
            item_level_users_rep, item_level_items_rep = self.graph_propagate(self.item_level_graph_wod,
                                                                              self.init_users_rep, self.init_items_rep,
                                                                              self.item_level_dropout, test)
        else:
            item_level_users_rep, item_level_items_rep = self.graph_propagate(self.item_level_graph,
                                                                              self.init_users_rep, self.init_items_rep,
                                                                              self.item_level_dropout, test)

        if test:
            bundle_level_users_rep, bundle_level_bundles_rep = self.graph_propagate(self.bundle_level_graph_wod,
                                                                                    self.init_users_rep,
                                                                                    self.init_bundles_rep,
                                                                                    self.bundle_level_dropout, test)
        else:
            bundle_level_users_rep, bundle_level_bundles_rep = self.graph_propagate(self.bundle_level_graph,
                                                                                    self.init_users_rep,
                                                                                    self.init_bundles_rep,
                                                                                    self.bundle_level_dropout, test)

        bundling_strategy_level_bundles_rep, bundling_strategy_level_items_rep = self.graph_propagate(
            self.bundling_strategy_level_graph_wom, self.init_bundles_rep, self.init_items_rep,
            self.bundling_strategy_level_dropout, test)

        users_rep = [item_level_users_rep, bundle_level_users_rep]
        bundles_rep = [bundle_level_bundles_rep, bundling_strategy_level_bundles_rep]
        items_rep = [item_level_items_rep, bundling_strategy_level_items_rep]

        return users_rep, bundles_rep, items_rep

    def get_contrastive_loss(self, pos, neg):
        pos = pos[:, 0, :]
        neg = neg[:, 0, :]
        pos = functional.normalize(pos, p=2, dim=1)
        neg = functional.normalize(neg, p=2, dim=1)
        pos_score = torch.sum(pos * neg, dim=1)
        total_score = torch.matmul(pos, neg.permute(1, 0))
        pos_score = torch.exp(pos_score / self.c_temp)
        total_score = torch.sum(torch.exp(total_score / self.c_temp), axis=1)
        contrastive_loss = - torch.mean(torch.log(pos_score / total_score))
        return contrastive_loss

    def get_contrastive_loss_for_items(self, x, y):
        x = x[:, 0, :]
        y = y[:, 0, :]
        x = functional.normalize(x, p=2, dim=1)
        y = functional.normalize(y, p=2, dim=1)
        pos_score = torch.sum(x * y, dim=1)
        total_score = np.array([])

        for i in range(len(x)):
            temp = torch.matmul(x[i], y.permute(1, 0))
            temp = torch.exp(temp / self.c_temp)
            temp = torch.sum(temp)
            total_score = np.append(total_score, temp.detach().cpu().numpy())

        total_score = torch.tensor(total_score, device=self.device)
        pos_score = torch.exp(pos_score / self.c_temp)
        contrastive_loss = - torch.mean(torch.log(pos_score / total_score))
        return contrastive_loss

    def get_reconstruction_loss(self, x, y, temp=0.2):
        x = functional.normalize(x)
        y = functional.normalize(y)
        pos_score = torch.exp(torch.sum(x * y, dim=1) / temp)
        total_score = torch.sum(torch.exp(x @ y.T / temp), dim=1)
        return -torch.log(pos_score / (total_score + 1e-8) + 1e-8).mean()

    def reconstruction(self, masked_rep, seeds):
        mask_token_rep = masked_rep[seeds]
        init_rep = self.init_items_rep[seeds]
        reconstruction_loss = self.get_reconstruction_loss(mask_token_rep, init_rep, 0.2)
        return reconstruction_loss

    def get_loss(self, users_rep_prim, bundles_rep_prim, bundles_rep_aux, items_rep_aux, masked_item_rep, item_seeds,
                 masked_bundle_rep, bundle_seeds):
        item_level_users_rep_prim, bundle_level_users_rep_prim = users_rep_prim
        bundle_level_bundles_rep_prim, bundling_strategy_level_bundles_rep_prim = bundles_rep_prim
        bundle_level_bundles_rep_aux, bundling_strategy_level_bundles_rep_aux = bundles_rep_aux
        item_level_items_rep_aux, bundling_strategy_level_items_rep_aux = items_rep_aux

        yub = torch.sum(torch.cat((item_level_users_rep_prim, bundle_level_users_rep_prim), 2) * torch.cat(
            (bundle_level_bundles_rep_prim, bundling_strategy_level_bundles_rep_prim), 2), 2)
        rec_bpr_loss = get_bpr_loss(yub)

        ybi = torch.sum(
            torch.cat((bundle_level_bundles_rep_aux, bundling_strategy_level_bundles_rep_aux), 2) * torch.cat(
                (item_level_items_rep_aux, bundling_strategy_level_items_rep_aux), 2), 2)
        aux_bpr_loss = get_bpr_loss(ybi)

        user_contrastive_loss = self.get_contrastive_loss(item_level_users_rep_prim, bundle_level_users_rep_prim)
        bundle_contrastive_loss = self.get_contrastive_loss(bundle_level_bundles_rep_prim,
                                                            bundling_strategy_level_bundles_rep_prim)
        item_contrastive_loss = self.get_contrastive_loss_for_items(item_level_items_rep_aux,
                                                                    bundling_strategy_level_items_rep_aux)

        contrastive_losses = [user_contrastive_loss, bundle_contrastive_loss, item_contrastive_loss]
        contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)

        item_reconstruction_loss = self.reconstruction(masked_item_rep, item_seeds)
        bundle_reconstruction_loss = self.reconstruction(masked_bundle_rep, bundle_seeds)

        return rec_bpr_loss, aux_bpr_loss, contrastive_loss, item_reconstruction_loss, bundle_reconstruction_loss

    def forward(self, batch, drop=False):
        if drop:
            self.get_item_level_graph()
            self.get_bundle_level_graph()

        users, bundles = batch
        items = []
        for i in range(len(bundles)):
            for j in self.b_i_dict[int(bundles[i][0])]:
                temp_pair = [j]
                while True:
                    rd = np.random.randint(self.num_items)
                    if self.bi_graph[int(bundles[i][0]), rd] == 0 and rd not in temp_pair:
                        temp_pair.append(rd)
                        if len(temp_pair) == self.neg_sample + 1:
                            break
                temp_pair.append(bundles[i][0])
                items.append(temp_pair)
        items = torch.LongTensor(items)

        masked_item_rep, item_seeds = self.item_mask()
        masked_bundle_rep, bundle_seeds = self.bundle_mask()

        users_rep, bundles_rep, items_rep = self.propagate_train(masked_item_rep, masked_bundle_rep)

        users_rep_prim = [i[users].expand(-1, bundles.shape[1], -1) for i in users_rep]
        bundles_rep_prim = [i[bundles] for i in bundles_rep]
        bundles_rep_aux = [torch.unsqueeze(i[items[:, -1]], 1).expand(-1, items.shape[1] - 1, -1) for i in bundles_rep]
        items_rep_aux = [i[items[:, :-1]] for i in items_rep]

        rec_bpr_loss, aux_bpr_loss, contrastive_loss, item_reconstruction_loss, bundle_reconstruction_loss = self.get_loss(
            users_rep_prim, bundles_rep_prim, bundles_rep_aux, items_rep_aux, masked_item_rep, item_seeds,
            masked_bundle_rep, bundle_seeds)

        return rec_bpr_loss, aux_bpr_loss, contrastive_loss, item_reconstruction_loss, bundle_reconstruction_loss

    def evaluate(self, propagate_result, users):
        users_rep, bundles_rep, items_rep = propagate_result
        item_level_users_rep, bundle_level_users_rep = [i[users] for i in users_rep]
        item_level_bundles_rep, bundle_level_bundles_rep = bundles_rep

        scores = torch.mm(item_level_users_rep, item_level_bundles_rep.t()) + torch.mm(bundle_level_users_rep,
                                                                                       bundle_level_bundles_rep.t())
        return scores


class NodeMask(nn.Module):
    def __init__(self, conf, masking_ratio, embedding_size, mask_num):
        super(NodeMask, self).__init__()
        self.masking_ratio = masking_ratio
        self.mask_num = mask_num
        self.mask_token = nn.Parameter(torch.zeros(self.mask_num, embedding_size))
        self.device = conf["device"]

    def forward(self, embeds):
        seeds = np.random.choice(embeds.shape[0], size=self.mask_num, replace=False)
        seeds = torch.LongTensor(seeds).to(self.device)
        masked_embeds = embeds.clone()
        masked_embeds[seeds] = self.mask_token
        return masked_embeds, seeds
