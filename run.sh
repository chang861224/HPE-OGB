python3 SampleEdges.py --dataset ogbl-citation2 --train_network network.txt --train_percent 100

./HPE -train network.txt -embed node-feat.txt -save hpe_rep.txt -dimensions 128 -undirected 1 -sample_times 1200 -walk_steps 2 -threads 8
#./BPR -train network.txt -embed node-feat.txt -save bpr_rep.txt -dimensions 64 -sample_times 1200 -threads 8 -directed 1
#./WARP -train network.txt -embed node-feat.txt -save warp_rep.txt -dimensions 64 -sample_times 1200 -threads 8 -directed 1
#./HOPREC -train network.txt -embed node-feat.txt -field node-year.txt -save hoprec_rep.txt -dimensions 64 -sample_times 1200 -threads 8 -directed 1
#./smore/cli/hpe -train network.txt -save hpe_rep.txt -dimensions 512 -undirected 1 -sample_times 1200 -walk_steps 5 -threads 8
#./smore/cli/bpr -train network.txt -save bpr_rep.txt -dimensions 64 -sample_times 1200 -threads 8
#./smore/cli/warp -train network.txt -save warp_rep.txt -dimensions 64 -sample_times 1200 -threads 8
#./smore/cli/hoprec -train network.txt -field node-year.txt -save hoprec_rep.txt -dimensions 64 -sample_times 1200 -threads 8

python3 predict.py --dataset ogbl-citation2 --embed hpe_rep.txt


