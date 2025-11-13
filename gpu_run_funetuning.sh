#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#!SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH -p swarm_a100
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --job-name=casrel
#SBATCH --output=./result_out/run_webnlg_t5_large_re_e100_bs64_lr_1e-4-%j.out

module add nvidia-cuda
nvidia-smi

echo "the job start "

# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2
python main.py --model bert-lstm-crf  --data_path ./data/ --save_dir ./save_model --word_embedding_model_path /home/mh1f25/scratch/llms_model/bert-base-cased  --do_train --do_test  --num_epochs 3



echo "the job end "




