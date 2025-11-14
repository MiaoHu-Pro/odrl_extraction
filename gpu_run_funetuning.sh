#!/bin/bash
#SBATCH --mail-user=miao.hu@soton.ac.uk
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#!SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH -p a100
#SBATCH --gres=gpu:1
#SBATCH --time=60:00:00
#SBATCH --job-name=odrl_extraction
#SBATCH --output=./result_out/roberta-%j.out

module add nvidia-cuda
nvidia-smi

echo "the job start "

# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2
#python main.py --model bert-lstm-crf  --data_path ./data/ --save_dir ./save_model --word_embedding_model_path /home/mh1f25/scratch/llms_model/bert-base-cased  --do_train --do_test  --num_epochs  20  --rnn_layers 5

python main.py --model roberta-lstm-crf  --data_path ./data/ --save_dir ./save_model --word_embedding_model_path /home/mh1f25/scratch/llms_model/roberta-large  --do_train --do_test  --num_epochs  20   --rnn_layers 5


#python main.py --model bert-lstm-crf  --data_path ./data/ --save_dir ./save_model --word_embedding_model_path /home/mh1f25/scratch/llms_model/bert-base-cased  --do_train --do_test  --num_epochs 3
#python main.py --model t5-lstm-crf  --data_path ./data/ --save_dir ./save_model --word_embedding_model_path /home/mh1f25/scratch/llms_model/t5-base  --do_train --do_test  --num_epochs 3
#python main.py --model llama-lstm-crf  --data_path ./data/ --save_dir ./save_model --word_embedding_model_path /home/mh1f25/scratch/llms_model/Llama-3.2-1B  --do_train --do_test  --num_epochs 3
#python main.py --model mixtral-lstm-crf  --data_path ./data/ --save_dir ./save_model --word_embedding_model_path /home/mh1f25/scratch/llms_model/Mistral7B  --do_train --do_test  --num_epochs 3

#python main.py --model deberta-lstm-crf  --data_path ./data/ --save_dir ./save_model --word_embedding_model_path /home/mh1f25/scratch/llms_model/deberta-v3-base  --do_train --do_test  --num_epochs  20



echo "the job end "




