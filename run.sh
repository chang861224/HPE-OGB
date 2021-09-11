python3 SampleEdges.py --dataset ogbl-citation2 --train_network network.txt --train_percent 100
#smore/cli/hpe -train network.txt -save hpe_rep.txt -dimensions 64 -undirected 0 -sample_times 1200 -walk_steps 5 -threads 8
./HPE -train network.txt -embed node-feat.txt -save hpe_rep.txt -dimensions 64 -undirected 0 -sample_times 1200 -walk_steps 5 -threads 8
#./BPR -train network.txt -directed 0 -embed node-feat.txt -save bpr_rep.txt -dimensions 64 -sample_times 1200 -threads 8
python3 predict.py --dataset ogbl-citation2 --val_percent 100 --test_percent 100 --embed hpe_rep.txt
#python3 predict.py --dataset ogbl-citation2 --val_percent 100 --test_percent 100 --embed bpr_rep.txt

