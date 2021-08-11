import argparse
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

# Parse arguments
parser = argparse.ArgumentParser(description="OGB link prediction task")
parser.add_argument("--dataset", type=str, default="ogbl-citation2")
parser.add_argument("--val_percent", type=float, default=100)
parser.add_argument("--test_percent", type=float, default=100)
parser.add_argument("--embed", type=str, default="ogbl-citation2/node-feat.csv")
args = parser.parse_args()

embed_path = args.embed
dataset_name = args.dataset
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
            paper_embed = np.array([float(e) for e in paper_embed])
            embedding[paper_id] = paper_embed
elif embed_path.startswith("ogbl-citation2/node-feat.csv"):
    embedding = np.genfromtxt(embed_path, delimiter=",", skip_header=False)
else:
    raise ValueError("Unknown embedding file")

# Load validation set and testing set
dataset = PygLinkPropPredDataset(name=dataset_name)
split_edge = dataset.get_edge_split()
valid_edge = split_edge["valid"]
test_edge = split_edge["test"]
evaluator = Evaluator(name=dataset_name)

# Setup and initial variables
val_size = valid_edge["source_node"].size()[0]
val_num_sample = int(val_percent / 100 * val_size)
val_sample_index = random.sample(range(val_size), val_num_sample)
test_size = test_edge["source_node"].size()[0]
test_num_sample = int(test_percent / 100 * test_size)
test_sample_index = random.sample(range(test_size), test_num_sample)

# Predict validation set
print("Predicting validation set....")
pos_scores = list()
neg_scores = list()

for idx in tqdm(val_sample_index):
    source = int(valid_edge["source_node"][idx])
    target = int(valid_edge["target_node"][idx])
    target_neg = valid_edge["target_node_neg"][idx].tolist()

    try:
        pos_score = np.dot(embedding[source], embedding[target])
    except:
        pos_score = 0.0
    pos_scores.append(pos_score)

    neg_score = list()
    for ix, node in enumerate(target_neg):
        try:
            score = np.dot(embedding[source], embedding[node])
        except:
            score = 0.0
        neg_score.append(score)
    neg_scores.append(neg_score)

val_result = evaluator.eval({"y_pred_pos": torch.tensor(pos_scores), "y_pred_neg": torch.tensor(neg_scores)})

# Predict test set
print("Predicting testing set....")
pos_scores = list()
neg_scores = list()

for idx in tqdm(test_sample_index):
    source = int(test_edge["source_node"][idx])
    target = int(test_edge["target_node"][idx])
    target_neg = test_edge["target_node_neg"][idx].tolist()

    try:
        pos_score = np.dot(embedding[source], embedding[target])
    except:
        pos_score = 0.0
    pos_scores.append(pos_score)

    neg_score = list()
    for ix, node in enumerate(target_neg):
        try:
            score = np.dot(embedding[source], embedding[node])
        except:
            score = 0.0
        neg_score.append(score)
    neg_scores.append(neg_score)

test_result = evaluator.eval({"y_pred_pos": torch.tensor(pos_scores), "y_pred_neg": torch.tensor(neg_scores)})

print("===== Prediction Result =====")
print("Valid MRR: {:.4f}".format(float(torch.mean(val_result["mrr_list"]))))
print("Test MRR: {:.4f}".format(float(torch.mean(test_result["mrr_list"]))))

