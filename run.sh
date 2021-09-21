python3 SampleEdges.py --dataset ogbl-citation2 --train_network network.txt --train_percent 100
#python3 SampleEdges.py --dataset ogbl-citation2 --train_network network.txt --train_percent 100 --tag True

./HPE -train network.txt -embed node-feat.txt -save hpe_rep.txt -dimensions 64 -undirected 0 -sample_times 1200 -walk_steps 5 -threads 8
#./BPR -train network.txt -embed node-feat.txt -save bpr_rep.txt -dimensions 64 -sample_times 1200 -threads 8 -directed 1
#./WARP -train network.txt -embed node-feat.txt -save warp_rep.txt -dimensions 64 -sample_times 1200 -threads 8 -directed 1
#./HOPREC -train network.txt -embed node-feat.txt -field node-year.txt -save hoprec_rep.txt -dimensions 64 -sample_times 1200 -threads 8 -directed 1
#./smore/cli/hpe -train network.txt -save hpe_rep.txt -dimensions 64 -undirected 0 -sample_times 1200 -walk_steps 5 -threads 8

python3 predict.py --dataset ogbl-citation2 --val_percent 100 --test_percent 100 --embed hpe_rep.txt

#./smore/cli/bpr -train network.txt -save bpr_rep.txt -dimensions 64 -sample_times 1200 -threads 8
#./smore/cli/warp -train network.txt -save warp_rep.txt -dimensions 64 -sample_times 1200 -threads 8
#./smore/cli/hoprec -train network.txt -field node-year.txt -save hoprec_rep.txt -dimensions 64 -sample_times 1200 -threads 8

#python3 predict.py --dataset ogbl-citation2 --val_percent 100 --test_percent 100 --embed hpe_rep.txt --tag True
