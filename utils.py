import numpy as np
import random
import torch

def SplitDataset(dataset, percent=10):
    dataset_path = {
        "cora": "dataset/Cora/cora_cit",
        "dblp_v10": "dataset/DBLP/dblp_v10_cit",
        "pubmed": "dataset/PubMed-Diabete/pubmed_diabetes_cit",
        "us_patent": "dataset/US-Patent/us_cit_patent",
        "arxiv_HepPh": "dataset/arXiv/cit-HepPh/arxiv_cit_HepPh",
        "arxiv_HepTh": "dataset/arXiv/cit-HepTh/arxiv_cit_HepTh",
    }[dataset]

    with open(dataset_path + ".edgelist", "r") as f:
        lines = f.readlines()

    edgelist = dict()
    all_nodes = set()

    for line in lines:
        source, target = line.split()
        source = int(source)
        target = int(target)
        
        all_nodes.add(source)
        all_nodes.add(target)

        try:
            edgelist[source].append(target)
        except:
            edgelist[source] = list()
            edgelist[source].append(target)

    print(len(edgelist))
    num_sample = int(len(edgelist) * percent / 100)
    sample_instances = random.sample(edgelist.keys(), num_sample)
    sample_set = {
        "source": list(),
        "pos_target": list(),
        "neg_target": list()
    }

    for node in sample_instances:
        sample_set["source"].append(node)
        sample_set["pos_target"].append(random.sample(edgelist[node], 1)[0])
        sample_set["neg_target"].append(random.sample(all_nodes - set(edgelist[node]) - {node}, 1000))

    return sample_set


def Evaluate(input_dict):
    y_pred_pos, y_pred_neg, type_info = ParseAndCheckInput(input_dict)
    return _eval_mrr(y_pred_pos, y_pred_neg, type_info)


def ParseAndCheckInput(input_dict):
    if not 'y_pred_pos' in input_dict:
        raise RuntimeError('Missing key of y_pred_pos')
    if not 'y_pred_neg' in input_dict:
        raise RuntimeError('Missing key of y_pred_neg')

    y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

    '''
        y_pred_pos: numpy ndarray or torch tensor of shape (num_edge, )
        y_pred_neg: numpy ndarray or torch tensor of shape (num_edge, num_node_negative)
    '''

    # convert y_pred_pos, y_pred_neg into either torch tensor or both numpy array
    # type_info stores information whether torch or numpy is used

    type_info = None

    # check the raw tyep of y_pred_pos
    if not (isinstance(y_pred_pos, np.ndarray) or (torch is not None and isinstance(y_pred_pos, torch.Tensor))):
        raise ValueError('y_pred_pos needs to be either numpy ndarray or torch tensor')

    # check the raw type of y_pred_neg
    if not (isinstance(y_pred_neg, np.ndarray) or (torch is not None and isinstance(y_pred_neg, torch.Tensor))):
        raise ValueError('y_pred_neg needs to be either numpy ndarray or torch tensor')

    # if either y_pred_pos or y_pred_neg is torch tensor, use torch tensor
    if torch is not None and (isinstance(y_pred_pos, torch.Tensor) or isinstance(y_pred_neg, torch.Tensor)):
        # converting to torch.Tensor to numpy on cpu
        if isinstance(y_pred_pos, np.ndarray):
            y_pred_pos = torch.from_numpy(y_pred_pos)

        if isinstance(y_pred_neg, np.ndarray):
            y_pred_neg = torch.from_numpy(y_pred_neg)

        # put both y_pred_pos and y_pred_neg on the same device
        y_pred_pos = y_pred_pos.to(y_pred_neg.device)

        type_info = 'torch'


    else:
        # both y_pred_pos and y_pred_neg are numpy ndarray

        type_info = 'numpy'


    if not y_pred_pos.ndim == 1:
        raise RuntimeError('y_pred_pos must to 1-dim arrray, {}-dim array given'.format(y_pred_pos.ndim))

    if not y_pred_neg.ndim == 2:
        raise RuntimeError('y_pred_neg must to 2-dim arrray, {}-dim array given'.format(y_pred_neg.ndim))

    return y_pred_pos, y_pred_neg, type_info


def _eval_mrr(y_pred_pos, y_pred_neg, type_info):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''

    if type_info == 'torch':
        y_pred = torch.cat([y_pred_pos.view(-1,1), y_pred_neg], dim = 1)
        argsort = torch.argsort(y_pred, dim = 1, descending = True)
        ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
        ranking_list = ranking_list[:, 1] + 1
        mrr_list = 1./ranking_list.to(torch.float)
        return mrr_list

    else:
        y_pred = np.concatenate([y_pred_pos.reshape(-1,1), y_pred_neg], axis = 1)
        argsort = np.argsort(-y_pred, axis = 1)
        ranking_list = (argsort == 0).nonzero()
        ranking_list = ranking_list[1] + 1
        mrr_list = 1./ranking_list.astype(np.float32)
        return mrr_list
    

if __name__ == "__main__":
    test_set = SplitDataset("pubmed", 20)
    print(len(test_set["source"]))

