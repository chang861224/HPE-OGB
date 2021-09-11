import argparse
import torch
import json
import random
import numpy as np
from tqdm import tqdm
from ogb.linkproppred import PygLinkPropPredDataset

# Arguments
parser = argparse.ArgumentParser(description="OGB Dataset")
parser.add_argument("--dataset", type=str, default="ogbl-citation2")
parser.add_argument("--train_network", type=str, default="network.txt")
parser.add_argument("--train_percent", type=int, default=100)
parser.add_argument("--directed", type=bool, default=False)
args = parser.parse_args()

# Save arguments to variables
dataset_name = args.dataset
train_network = args.train_network
train_percent = args.train_percent
directed = args.directed

# Load dataset from OGB
dataset = PygLinkPropPredDataset(name=dataset_name)
feature = dataset[0]["x"]
split_edge = dataset.get_edge_split()
train_edge = split_edge["train"]

# Generate build-in features
print("Generate build-in features file")

with open("node-feat.txt", "w") as f:
    for row in tqdm(feature):
        for feature in row:
            f.write("{} ".format(round(float(feature), 6)))
        f.write("\n")

# Sample training network
print("Start generating training network")
network = list()

for idx in range(train_edge["source_node"].size()[0]):
    source = int(train_edge["source_node"][idx])
    target = int(train_edge["target_node"][idx])
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

        if directed == True:
            f.write("S{} T{} 1\n".format(source, target))
        else:
            f.write("{} {} 1\n".format(source, target))

print("Done!!")

