# An Implementation of HPE for OGB Link Prediction Task
This repository is that we use the mothod HPE[1] on [OGB Link Prediction Task](https://ogb.stanford.edu/docs/leader_linkprop/). Especially, we focus on the dataset [ogbl-citation2](https://ogb.stanford.edu/docs/linkprop/#ogbl-citation2). If you want to learn more about HPE, please click [here](https://github.com/cnclabs/smore).


## Requirements

In order to implement the codes in this repository, you have to install the Python package listed below first:

- ogb == 1.3.1
- torch >= 1.6

Or, you can install required Python packages via `requirements.txt`.

```bash
pip3 install -r requirements.txt
```

Furthermore, the main method HPE is contained in the repository `smore`, so you have to follow the steps in next section to get and compile the module.


## Get Submodule - SMORe

Our method HPE is contained in the repository `smore`, but when you clone this repository, there would be nothing in the `smore` repository. Because of this, we have to follow the commands shown below to get the submodue `smore`.

```bash
git submodule init
git submodule update
```

After the steps above, we can successfully get the whole codes in `smore`.


## Implement HPE for OGB Link Prediction Task

### Compile HPE
Because the module in `smore` is written in c++, we have to compile the program before using these codes. Therefore, we should go the the root of this repository and compile via the following command:

```bash
make
```

In this compilation process, we can decompose it into two parts:

- Native `smore` program; because we would like to use the HPE module in this project.
- Content-based initialized HPE program; because we would like to use the pre-trained features provided by dataset to initial node embedding to train the HPE model.

After compilation, there are execution files, `HPE`, generated. Moreover, there are `BPR`, `HOP-REC`, and `WARP` execution files generated also. You can try these methods which contains in SMORe project if you want.

### Network Format
To use HPE in `smore` module, first we have to generate the connection network of our dataset. The format of the network should be like this:

```
userA itemA 3
userA itemC 5
userB itemA 1
userB itemB 5
userC itemA 4
```

In each row, it means a user select an item, and the number is the weight of this relationship.

### Generate Training Network
In order to generate the network file, we have to run `SampleEdges.py`. There are three options:

- `--dataset`: The dataset we want to generate the network. Default dataset is `ogbl-citation2`.
- `--train_network`: The file name which we want to save this network. Default is `network.txt`.
- `--train_percent`: The number of percentage of edges we want to train. Default is 100.
- `--directed`: The edges is directed or not. If it is directed, the source node would be add a tag "S" and the target node would be add a tag "T". Default is `False`.

We set `--train_percent` as 100 and others as default, so we can run `SampleEdges.py` as following:

```bash
python3 SampleEdges.py --dataset ogbl-citation2 --train_network network.txt --train_percent 100
```

### Use HPE Method
We use cli constructed in smore to train the network with HPE method. The command is like this:

```bash
hpe -train network.txt -embed node-feature.txt -save hpe_rep.txt -dimensions 64 -undirected 0 -sample_times 1200 -walk_steps 5 -threads 8
```

The command above means that we train the network, which saved in `network.txt`, using HPE with 1200 sample times  and generate the 64-dimensional embedding for each node. Then we save these embeddings to the file `hpe_rep.txt`. The format in this file should be like this:

```
6 64
userA 0.0815412 0.0205459 0.288714 0.296497 0.394043 ...
itemA -0.207083 -0.258583 0.233185 0.0959801 0.258183 ...
itemC 0.0185886 0.138003 0.213609 0.276383 0.45732 ...
userB -0.0137994 -0.227462 0.103224 -0.456051 0.389858 ...
itemB -0.317921 -0.163652 0.103891 -0.449869 0.318225 ...
userC -0.156576 -0.3505 0.213454 0.10476 0.259673 ...
```

You can adjust the size of `-thread` according to the capability of you device.

### Predict
After we generate the node embedding file, we can use it on our prediction task. We would run `predict.py` to predict the results and get the evaluation score. There are four options:

- `--dataset`: The dataset of our prediction task. Default is `ogbl-citation2`.
- `--embed`: The file which save the embedding of each nodes. Default is `hpe_rep.txt`.
- `--val_percent`: The number of percentage of validation set you want to test for the evaluation score. Default is 100.
- `--test_percent`: The number of percentage of testing set you want to test for the evaluation score. Default is 100.

Suppose we set both `--val_percent` and `test_percent` as 20, the command we run `predict.py` should be like this:

```bash
python3 predict.py --dataset ogbl-citation2 --val_percent 20 --test_percent 20 --embed hpe_rep.txt
```

To get the true evaluation score, which means that you want to predict the whole validation set and testing set, be sure to set `--val_percent` and `--test_percent` as 100.

### Implement with Shell Script
To summarize the above, we construct a shell script file `run.sh`, which contain our experiment steps. You can run this file as following command to get the evaluation score with HPE method.

```bash
sh run.sh
```

Moreover, you can try to change the arguments we set to see whether the score is better or not.


## Reference
[1] Chih-Ming Chen, Ming-Feng Tsai, Yu-Ching Lin, and Yi-Hsuan Yang. [Query-based Music Recommendations via Preference Embedding](https://dl.acm.org/doi/10.1145/2959100.2959169)
