import os
import yaml
import json
import argparse
from tqdm import tqdm
from itertools import product
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch

# Remove the comments below to make the model reproducible.
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)

import numpy as np

# Remove the comments below to make the model reproducible.
# np.random.seed(42)

import torch.optim as optim
from utility import Datasets
from models.BMGAE import BMGAE


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", default="0", type=str, help="The gpu to use.")
    parser.add_argument("-d", "--dataset", default="Youshu", type=str, help="The dataset to use.")
    parser.add_argument("-m", "--model", default="BMGAE", type=str, help="The model to train.")
    parser.add_argument("-i", "--info", default="", type=str,
                        help="The additional information that will be shown in the log file name.")
    args = parser.parse_args()
    return args


def main():
    conf = yaml.safe_load(open("./config.yaml"))
    print("Config file loaded.")
    paras = parse_arguments().__dict__
    dataset_name = paras["dataset"]
    conf = conf[dataset_name]

    conf["dataset"] = dataset_name
    conf["model"] = paras["model"]
    dataset = Datasets(conf)

    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]

    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    print(conf)

    for lr, lambda_4, u_i_aug_ratio, u_b_aug_ratio, b_i_aug_ratio, embedding_size, layer_num, lambda_2, c_temp, lambda_1, lambda_3, masking_ratio in \
            product(conf['lrs'], conf['lambda_4s'], conf['u_i_aug_ratios'], conf['u_b_aug_ratios'],
                    conf['b_i_aug_ratios'], conf["embedding_sizes"], conf["layer_nums"], conf["lambda_2s"],
                    conf["c_temps"], conf["lambda_1s"], conf["lambda_3s"], conf["masking_ratios"]):
        log_path = "./logs/%s/%s" % (conf["dataset"], conf["model"])
        run_path = "./runs/%s/%s" % (conf["dataset"], conf["model"])
        checkpoint_model_path = "./checkpoints/%s/%s/model" % (conf["dataset"], conf["model"])
        checkpoint_conf_path = "./checkpoints/%s/%s/conf" % (conf["dataset"], conf["model"])
        if not os.path.isdir(run_path):
            os.makedirs(run_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        if not os.path.isdir(checkpoint_model_path):
            os.makedirs(checkpoint_model_path)
        if not os.path.isdir(checkpoint_conf_path):
            os.makedirs(checkpoint_conf_path)

        conf["lambda_4"] = lambda_4
        conf["embedding_size"] = embedding_size

        settings = []
        if conf["info"] != "":
            settings += [conf["info"]]

        settings += [conf["aug_type"]]
        if conf["aug_type"] == "ED":
            settings += [str(conf["ed_interval"])]
        if conf["aug_type"] == "OG":
            assert u_i_aug_ratio == 0 and u_b_aug_ratio == 0 and b_i_aug_ratio == 0

        settings += ["Neg_%d" % (conf["neg_num"]), str(conf["batch_size_train"]), str(lr), str(lambda_4),
                     str(embedding_size)]

        conf["u_i_aug_ratio"] = u_i_aug_ratio
        conf["u_b_aug_ratio"] = u_b_aug_ratio
        conf["b_i_aug_ratio"] = b_i_aug_ratio
        conf["layer_num"] = layer_num
        settings += [str(u_i_aug_ratio), str(u_b_aug_ratio), str(b_i_aug_ratio), str(layer_num)]

        conf["lambda_2"] = lambda_2
        conf["c_temp"] = c_temp
        conf["lambda_1"] = lambda_1
        conf["lambda_3"] = lambda_3
        conf["masking_ratio"] = masking_ratio
        settings += [str(lambda_2), str(c_temp), str(lambda_1), str(lambda_3), str(masking_ratio)]

        setting = "_".join(settings)
        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting

        run = SummaryWriter(run_path)

        if conf['model'] == 'BMGAE':
            model = BMGAE(conf, dataset.graphs, dataset.b_i_dict).to(device)
        else:
            raise ValueError("Unimplemented model %s" % (conf["model"]))

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=conf["lambda_4"])

        batch_count = len(dataset.train_loader)
        test_interval_bs = int(batch_count * conf["test_interval"])
        dropout_interval_bs = int(batch_count * conf["ed_interval"])

        best_metrics, best_perform = init_best_metrics(conf)
        best_epoch = 0

        for name, param in model.named_parameters():
            print(f"{name}: {param.size()}")

        for epoch in range(conf['epochs']):
            epoch_anchor = epoch * batch_count
            model.train(True)
            process_bar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))

            for batch_i, batch in process_bar:
                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i

                drop = False
                if conf["aug_type"] == "ED" and (batch_anchor + 1) % dropout_interval_bs == 0:
                    drop = True
                rec_bpr_loss, aux_bpr_loss, contrastive_loss, item_reconstruction_loss, bundle_reconstruction_loss = model(
                    batch, drop=drop)
                loss = rec_bpr_loss + conf["lambda_1"] * aux_bpr_loss + conf["lambda_2"] * contrastive_loss + conf[
                    "lambda_3"] * item_reconstruction_loss + conf["lambda_3"] * bundle_reconstruction_loss
                loss.backward()
                optimizer.step()

                loss_scalar = loss.detach()
                rec_bpr_loss_scalar = rec_bpr_loss.detach()
                aux_bpr_loss_scalar = aux_bpr_loss.detach()
                contrastive_loss_scalar = contrastive_loss.detach()
                item_reconstruction_loss_scalar = item_reconstruction_loss.detach()
                bundle_reconstruction_loss = bundle_reconstruction_loss.detach()
                run.add_scalar("rec_bpr_loss", rec_bpr_loss_scalar, batch_anchor)
                run.add_scalar("aux_bpr_loss", aux_bpr_loss_scalar, batch_anchor)
                run.add_scalar("contrastive_loss", contrastive_loss_scalar, batch_anchor)
                run.add_scalar("item_reconstruction_loss)", item_reconstruction_loss_scalar, batch_anchor)
                run.add_scalar("bundle_reconstruction_loss)", bundle_reconstruction_loss, batch_anchor)
                run.add_scalar("loss", loss_scalar, batch_anchor)

                process_bar.set_description(
                    "epoch: %d, loss: %.4f, rec_bpr_loss: %.4f, aux_bpr_loss: %.4f, contrastive_loss: %.4f, item_reconstruction_loss: %.4f, bundle_reconstruction_loss: %.4f, GPU RAM: %.2f G/%.2f G" % (
                        epoch, loss_scalar, rec_bpr_loss_scalar, aux_bpr_loss_scalar, contrastive_loss_scalar,
                        item_reconstruction_loss_scalar,
                        bundle_reconstruction_loss, torch.cuda.max_memory_reserved(device) / 1024 ** 3,
                        torch.cuda.get_device_properties(device).total_memory / 1024 ** 3))

                if (batch_anchor + 1) % test_interval_bs == 0:
                    metrics = {"val": test(model, dataset.val_loader, conf),
                               "test": test(model, dataset.test_loader, conf)}
                    best_metrics, best_perform, best_epoch = log_metrics(conf, model, metrics, run, log_path,
                                                                         checkpoint_model_path, checkpoint_conf_path,
                                                                         epoch, batch_anchor, best_metrics,
                                                                         best_perform, best_epoch)


def init_best_metrics(conf):
    best_metrics = {"val": {}, "test": {}}
    for key in best_metrics:
        best_metrics[key]["Recall"] = {}
        best_metrics[key]["NDCG"] = {}
    for top_k in conf['top_k']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][top_k] = 0
    best_perform = {"val": {}, "test": {}}
    return best_metrics, best_perform


def write_log(run, log_path, top_k, step, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" % (m, top_k), val_score[top_k], step)
        run.add_scalar("%s_%d/Test" % (m, top_k), test_score[top_k], step)

    val_str = "%s, Val:  Recall@%d: %f, NDCG@%d: %f" % (
        curr_time, top_k, val_scores["Recall"][top_k], top_k, val_scores["NDCG"][top_k])
    test_str = "%s, Test:  Recall@%d: %f, NDCG@%d: %f" % (
        curr_time, top_k, test_scores["Recall"][top_k], top_k, test_scores["NDCG"][top_k])

    log = open(log_path, "a")
    log.write("%s\n" % val_str)
    log.write("%s\n" % test_str)
    log.close()

    print(val_str)
    print(test_str)


def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor,
                best_metrics, best_perform, best_epoch):
    for top_k in conf["top_k"]:
        write_log(run, log_path, top_k, batch_anchor, metrics)

    log = open(log_path, "a")

    top_k_judge = 20
    print("top%d as the final evaluation standard" % top_k_judge)
    if metrics["test"]["Recall"][top_k_judge] > best_metrics["test"]["Recall"][top_k_judge] and metrics["test"]["NDCG"][
        top_k_judge] > \
            best_metrics["test"]["NDCG"][top_k_judge]:
        torch.save(model.state_dict(), checkpoint_model_path)
        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for top_k in conf['top_k']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][top_k] = metrics[key][metric][top_k]

            best_perform["val"][top_k] = "%s, Best in epoch %d, Val:  Recall@%d=%.5f, NDCG@%d=%.5f" % (
                curr_time, best_epoch, top_k, best_metrics["val"]["Recall"][top_k], top_k,
                best_metrics["val"]["NDCG"][top_k])
            best_perform["test"][top_k] = "%s, Best in epoch %d, Test:  Recall@%d=%.5f, NDCG@%d=%.5f" % (
                curr_time, best_epoch, top_k, best_metrics["test"]["Recall"][top_k], top_k,
                best_metrics["test"]["NDCG"][top_k])
            print(best_perform["val"][top_k])
            print(best_perform["test"][top_k])
            log.write(best_perform["val"][top_k] + "\n")
            log.write(best_perform["test"][top_k] + "\n")

    log.close()

    return best_metrics, best_perform, best_epoch


def test(model, data_loader, conf):
    temp_metrics = {}
    for metric in ["Recall", "NDCG"]:
        temp_metrics[metric] = {}
        for top_k in conf["top_k"]:
            temp_metrics[metric][top_k] = [0, 0]

    device = conf["device"]
    model.eval()
    propagate_result = model.propagate_test(test=True)
    for users, u_b_ground_truth, u_b_train_data in data_loader:
        y = model.evaluate(propagate_result, users.to(device))
        y -= 1e8 * u_b_train_data.to(device)
        temp_metrics = get_metrics(temp_metrics, u_b_ground_truth.to(device), y, conf["top_k"])

    metrics = {}
    for metric, top_k_result in temp_metrics.items():
        metrics[metric] = {}
        for top_k, result in top_k_result.items():
            metrics[metric][top_k] = result[0] / result[1]

    return metrics


def get_metrics(metrics, u_b_ground_truth, y, top_ks):
    tmp = {"Recall": {}, "NDCG": {}}
    for top_k in top_ks:
        _, col_indices = torch.topk(y, top_k)
        row_indices = torch.zeros_like(col_indices) + torch.arange(y.shape[0], device=y.device, dtype=torch.long).view(
            -1, 1)
        hit = u_b_ground_truth[row_indices.view(-1), col_indices.view(-1)].view(-1, top_k)

        tmp["Recall"][top_k] = get_recall(y, u_b_ground_truth, hit)
        tmp["NDCG"][top_k] = get_ndcg(y, u_b_ground_truth, hit, top_k)

    for metric, top_k_result in tmp.items():
        for top_k, result in top_k_result.items():
            for i, x in enumerate(result):
                metrics[metric][top_k][i] += x

    return metrics


def get_recall(y, u_b_ground_truth, hit):
    epsilon = 1e-8
    hit_count = hit.sum(dim=1)
    pos_count = u_b_ground_truth.sum(dim=1)
    denorm = y.shape[0] - (pos_count == 0).sum().item()
    nomina = (hit_count / (pos_count + epsilon)).sum().item()
    return [nomina, denorm]


def get_ndcg(y, u_b_ground_truth, hit, top_k):
    def dcg(hit, top_k, device):
        hit = hit / torch.log2(torch.arange(2, top_k + 2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def idcg(pos_count, top_k, device):
        hit = torch.zeros(top_k, dtype=torch.float).to(device)
        hit[:pos_count] = 1
        return dcg(hit, top_k, device)

    device = u_b_ground_truth.device
    idcgs = torch.empty(1 + top_k, dtype=torch.float)
    idcgs[0] = 1
    for i in range(1, top_k + 1):
        idcgs[i] = idcg(i, top_k, device)

    pos_count = u_b_ground_truth.sum(dim=1).clamp(0, top_k).to(torch.long)
    dcg = dcg(hit, top_k, device)

    idcg = idcgs.to(device)[pos_count]
    ndcg = dcg / idcg.to(device)

    denorm = y.shape[0] - (pos_count == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()
