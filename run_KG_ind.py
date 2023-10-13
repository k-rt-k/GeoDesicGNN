import os
import torch
import argparse
import os.path as osp
import json
from datetime import datetime
from datasets import KGHorGDNegSampleDataset, KGVerGDNegSampleDataset

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from gnnfree.utils.evaluators import VarSizeRankingEvaluator
from gnnfree.utils.graph import construct_graph_from_edges
from gnnfree.utils.io import read_knowledge_graph
from gnnfree.utils.utils import (
    save_params,
    set_random_seed,
    hyperparameter_grid_search,
)
from gnnfree.nn.loss import FirstPosNegLoss
from gnnfree.utils.utils import write_res_to_file

from gnnfree.nn.models.GNN import RGCN

from models.link_predictor import GDLinkPredictor

from lightning_template import KGLinkPred


def main(params):
    # -------------------- Prepare Exp -------------------- #
    params = parser.parse_args()
    if not osp.exists("./saved_exp"):
        os.mkdir("./saved_exp")

    curtime = datetime.now()
    params.exp_dir = osp.join("./saved_exp", str(curtime))
    os.mkdir(params.exp_dir)

    save_params(osp.join(params.exp_dir, "command"), params)
    params.prefix = params.train_data_set + "_"
    params.model_name = osp.join(params.exp_dir, params.prefix)

    # -------------------- Data Preparation ---------------- #

    if params.gd_type == "HorGD":
        TrainSet = KGHorGDNegSampleDataset
    elif params.gd_type == "VerGD":
        TrainSet = KGVerGDNegSampleDataset
    elif params.gd_type == "":
        TrainSet = None

    database_path = osp.join(params.data_path, params.train_data_set)
    params.data_folder_path = osp.join(params.data_path, params.train_data_set)
    files = {}
    use_data = ["train", "test", "valid"]
    for f in use_data:
        files[f] = osp.join(database_path, f"{f}.txt")
    (
        trans_adj_list,
        trans_triplets,
        trans_entity2id,
        trans_relation2id,
        trans_id2entity,
        trans_id2relation,
    ) = read_knowledge_graph(files)

    trans_num_entities = len(trans_entity2id)
    trans_num_rel = len(trans_relation2id)

    params.test_data_set = params.train_data_set + "_ind"
    test_database_path = osp.join(params.data_path, params.test_data_set)
    files = {}
    for f in use_data:
        files[f] = osp.join(test_database_path, f"{f}.txt")

    (
        ind_adj_list,
        ind_triplets,
        ind_entity2id,
        ind_relation2id_test,
        ind_id2entity,
        ind_id2relation,
    ) = read_knowledge_graph(files, relation2id=trans_relation2id)
    ind_num_entities = len(ind_entity2id)

    params.num_rels = trans_num_rel
    params.aug_num_rels = trans_num_rel * 2

    trans_graph = construct_graph_from_edges(
        trans_triplets["train"].T[0],
        trans_triplets["train"].T[2],
        n_entities=trans_num_entities,
        inverse_edge=True,
        edge_type=trans_triplets["train"].T[1],
        num_rels=trans_num_rel,
    )
    ind_graph = construct_graph_from_edges(
        ind_triplets["train"].T[0],
        ind_triplets["train"].T[2],
        n_entities=ind_num_entities,
        inverse_edge=True,
        edge_type=ind_triplets["train"].T[1],
        num_rels=trans_num_rel,
    )

    trans_graph.ndata["feat"] = torch.ones(
        [trans_graph.num_nodes(), 1], dtype=torch.float32
    )
    ind_graph.ndata["feat"] = torch.ones(
        [ind_graph.num_nodes(), 1], dtype=torch.float32
    )
    params.inp_dim = 1

    train = TrainSet(
        trans_graph,
        trans_triplets["train"],
        params,
        trans_adj_list,
        mode="train",
        neg_link_per_sample=params.neg_sample,
    )

    test = TrainSet(
        ind_graph,
        ind_triplets["test"],
        params,
        ind_adj_list,
        mode="eval",
        neg_link_per_sample=params.neg_sample,
    )

    if params.val_method == "trans":
        val = TrainSet(
            trans_graph,
            trans_triplets["valid"],
            params,
            trans_adj_list,
            mode="eval",
            neg_link_per_sample=params.neg_sample,
        )
    else:
        val = TrainSet(
            ind_graph,
            ind_triplets["valid"],
            params,
            ind_adj_list,
            mode="eval",
            neg_link_per_sample=params.neg_sample,
        )

    params.train_edges = len(train)

    params.test_size = len(test)
    params.val_size = len(val)

    params.train_sample_size = None
    params.eval_sample_size = None

    print(
        f"Training graph has {params.train_edges} edges, validataion set has"
        f"{params.val_size} query edges, test set has {params.test_size} query"
        f"edges"
    )

    # ---------------- Data Preparation End ------------- #
    evlter = VarSizeRankingEvaluator("varsizeeval", params.num_workers)
    eval_metric = "h10"

    loss = FirstPosNegLoss(params.neg_sample)

    def run_exp(data, params):
        params.reach_dist = params.num_layers
        feature_list = ["head", "tail", "dist", "Rel"]
        if params.gd_type == "VerGD":
            feature_list.append("HeadVerGD" + ("Deg" if params.gd_deg else ""))
            feature_list.append("TailVerGD" + ("Deg" if params.gd_deg else ""))
        elif params.gd_type == "HorGD":
            feature_list.append("HorGD")
        params.feature_list = feature_list

        gnn = RGCN(
            params.reach_dist,
            params.aug_num_rels,
            params.inp_dim,
            params.emb_dim,
            params.num_bases,
        )

        link_predictor = GDLinkPredictor(
            params.emb_dim, gnn, feature_list, params.num_rels
        )

        link_pred = KGLinkPred(
            params.exp_dir,
            {"train": train, "val": val, "test": test},
            params,
            gnn,
            link_predictor,
            loss,
            evlter,
        )

        trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=params.num_layers,
            callbacks=[
                TQDMProgressBar(refresh_rate=20),
                ModelCheckpoint(monitor=eval_metric, mode="max"),
            ],
            logger=CSVLogger(save_dir=params.exp_dir),
        )
        trainer.fit(link_pred)
        valid_res = trainer.validate()[0]
        test_res = trainer.test()[0]
        return valid_res, test_res

    if params.psearch:
        hparams = {"num_layers": [2, 3, 4, 5],"train_data_set":["fb237_v1_ind","fb237_v2_ind","fb237_v3_ind","fb237_v4_ind", "WN18RR_v1_ind","WN18RR_v2_ind","WN18RR_v3_ind","WN18RR_v4_ind"]}
        if params.gd_type == "VerGD":
            hparams["gd_deg"] = [True, False]
    else :
        hparams = {"num_layers": [params.num_layers]}

    best_res = hyperparameter_grid_search(
        hparams, [train, test, val], run_exp, params, eval_metric, evlter
    )

    write_res_to_file(
        osp.join(params.exp_dir, "result"),
        params.train_data_set,
        eval_metric,
        best_res["test_mean"],
        best_res["val_mean"],
        best_res["test_std"],
        best_res["val_std"],
        best_res["params"],
        json.dumps(best_res),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gnn")

    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--train_data_set", type=str, default="brazil_airport")

    parser.add_argument(
        "--emb_dim", type=int, default=32, help="overall embedding dimension"
    )
    parser.add_argument(
        "--mol_emb_dim",
        type=int,
        default=32,
        help="embedding dimension for atom/bond",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of GNN layers"
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="jumping knowledge, should be 'last' or 'sum'",
    )

    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument(
        "--l2", type=float, default=0, help="l2 regularization strength"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="training batch size"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=64, help="evaluation batch size"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="number of workers in dataloading",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="number of epochs in one training routine",
    )

    parser.add_argument(
        "--reach_dist",
        type=int,
        default=3,
        help="max cutoff distance to find geodesics",
    )

    parser.add_argument(
        "--fold",
        type=int,
        default=10,
        help="number of fold for cross validation",
    )

    parser.add_argument(
        "--gd_type",
        type=str,
        default="HorGD",
        help="geodesic types, should be VerGD or HorGD",
    )

    parser.add_argument(
        "--gd_deg",
        type=bool,
        default=False,
        help="whether to use geodesic degrees for VerGD",
    )

    parser.add_argument(
        "--psearch",
        type=bool,
        default=False,
        help="perform hyperparameter search",
    )
    params = parser.parse_args()
    set_random_seed(1)
    main(params)
