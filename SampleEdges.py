import argparse
import torch
import json
import random
import numpy as np
from tqdm import tqdm
from ogb.linkproppred import PygLinkPropPredDataset

# Arguments
parser = argparse.ArgumentParser(description="OGB Dataset")
parser.add_argument("--train_network", type=str, default="network.txt")
parser.add_argument("--train_percent", type=int, default=100)
args = parser.parse_args()

# Save arguments to variables
dataset_name = "ogbl-citation2"
train_path = "train_edges.json"
valid_path = "valid_edges.json"
test_path = "test_edges.json"
train_network = args.train_network
train_percent = args.train_percent

# Load dataset from OGB
dataset = PygLinkPropPredDataset(name=dataset_name)
split_edge = dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]

# Transform OGB dataset format to json
train_map = {
    "source": train_edge["source_node"].tolist(),
    "target": train_edge["target_node"].tolist()
}
valid_map = {
    "source": valid_edge["source_node"].tolist(),
    "pos_node": valid_edge["target_node"].tolist(),
    "neg_node": valid_edge["target_node_neg"].tolist()
}
test_map = {
    "source": test_edge["source_node"].tolist(),
    "pos_node": test_edge["target_node"].tolist(),
    "neg_node": test_edge["target_node_neg"].tolist()
}

train_dict = dict()
valid_dict = dict()
test_dict = dict()

print("Making training set....")

for i in tqdm(range(len(train_map["source"]))):
    source = train_map["source"][i]
    target = train_map["target"][i]

    try:
        train_dict[source].append(target)
    except:
        train_dict[source] = list()
        train_dict[source].append(target)

json.dump(train_dict, open(train_path, "w"))
print("Training set made!!")

print("Making validation set....")

for i in tqdm(range(len(valid_map["source"]))):
    source = valid_map["source"][i]
    pos_node = valid_map["pos_node"][i]
    neg_node = valid_map["neg_node"][i]

    valid_dict[source] = dict()
    valid_dict[source]["pos_node"] = pos_node
    valid_dict[source]["neg_node"] = neg_node

json.dump(valid_dict, open(valid_path, "w"))
print("Validation set made!!")

print("Making testing set....")

for i in tqdm(range(len(test_map["source"]))):
    source = test_map["source"][i]
    pos_node = test_map["pos_node"][i]
    neg_node = test_map["neg_node"][i]

    test_dict[source] = dict()
    test_dict[source]["pos_node"] = pos_node
    test_dict[source]["neg_node"] = neg_node

json.dump(test_dict, open(test_path, "w"))
print("Testing set made!!")

# Sample training network
print("Start making training network")
network = list()

for i in range(len(train_map["source"])):
    source = train_map["source"][i]
    target = train_map["target"][i]
    network.append([source, target])

num_sample = int(train_percent / 100 * len(network))
sample_edges = random.sample(network, num_sample)
print("Sample percent:", train_percent)
print("# of samle edges:", num_sample)

print("Sampling edges....")

with open(train_network, "w") as f:
    for edge in tqdm(sample_edges):
        source = edge[0]
        target = edge[1]
        f.write("{} {} 1\n".format(source, target))

print("Done!!")

