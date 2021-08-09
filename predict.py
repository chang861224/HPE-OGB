import argparse
import json
import torch
import random
import numpy as np
from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser(description="OGB link prediction task")
parser.add_argument("--val_edges", type=str, default="valid_edges.json")
parser.add_argument("--test_edges", type=str, default="test_edges.json")
parser.add_argument("--val_percent", type=float, default=100)
parser.add_argument("--test_percent", type=float, default=100)
parser.add_argument("--embed", type=str, default="ogbl-citation2/node-feat.csv")
args = parser.parse_args()

embed_path = args.embed
val_percent = args.val_percent
test_percent = args.test_percent

# Load node embedding
if embed_path.startswith("hpe"):
    with open(embed_path, "r") as f:
        lines = f.readlines()

    embedding = dict()

    for line in lines:
        paper_id, paper_embed = line.strip("\n").split(" ", 1)
        paper_embed = paper_embed.split()

        if len(paper_embed) > 1:
            paper_id = int(paper_id)
            #paper_embed = np.array([float(e) for e in paper_embed])
            paper_embed = torch.tensor([float(e) for e in paper_embed])
            embedding[paper_id] = paper_embed
elif embed_path.startswith("ogbl-citation2/node-feat.csv"):
    embedding = np.genfromtxt(embed_path, delimiter=",", skip_header=False)
else:
    raise ValueError("Unknown embedding file")

# Setup and initial variables
num_pos = 1
num_neg = 1000

val_edges = json.load(open(args.val_edges))
val_num_sample = int(val_percent / 100 * len(val_edges))
val_sampled_keys = random.sample(val_edges.keys(), val_num_sample)
test_edges = json.load(open(args.test_edges))
test_num_sample = int(test_percent / 100 * len(test_edges))
test_sampled_keys = random.sample(test_edges.keys(), test_num_sample)

val_mrrs = list()
test_mrrs = list()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Predict validation set
print("Predicting validation set....")

for key in tqdm(val_sampled_keys):
    source = int(key)
    target_nodes = [val_edges[key]["pos_node"]] + val_edges[key]["neg_node"]
    
    y_true = [1] * num_pos + [0] * num_neg
    pred_scores = np.zeros(len(target_nodes), dtype=float)

    for idx, node in enumerate(target_nodes):
        try:
            #pred_scores[idx] += np.dot(embedding[source], embedding[node])
            pred_scores[idx] += torch.dot(embedding[source], embedding[node]).to(device).tolist()
        except:
            continue

    y_pred = ["N"] * len(pred_scores)
    sorted_indexes = np.argsort(-pred_scores)

    for rank, idx in enumerate(sorted_indexes):
        if idx < num_pos:
            y_pred[rank] = 1
        else:
            y_pred[rank] = 0

    rs = np.asarray(y_pred).nonzero()[0]
    val_mrrs.append(1. / (rs[0] + 1) if rs.size else 0.)

# Predict test set
print("Predicting testing set....")

for key in tqdm(test_sampled_keys):
    source = int(key)
    target_nodes = [test_edges[key]["pos_node"]] + test_edges[key]["neg_node"]
    
    y_true = [1] * num_pos + [0] * num_neg
    pred_scores = np.zeros(len(target_nodes), dtype=float)

    for idx, node in enumerate(target_nodes):
        try:
            #pred_scores[idx] += np.dot(embedding[source], embedding[node])
            pred_scores[idx] += torch.dot(embedding[source], embedding[node]).to(device).tolist()
        except:
            continue

    y_pred = ["N"] * len(pred_scores)
    sorted_indexes = np.argsort(-pred_scores)

    for rank, idx in enumerate(sorted_indexes):
        if idx < num_pos:
            y_pred[rank] = 1
        else:
            y_pred[rank] = 0

    rs = np.asarray(y_pred).nonzero()[0]
    test_mrrs.append(1. / (rs[0] + 1) if rs.size else 0.)

print("===== Prediction Result =====")
print("Valid MRR: {:.4f}".format(np.mean(val_mrrs)))
print("Test MRR: {:.4f}".format(np.mean(test_mrrs)))

