
import logging
import os
import argparse
from xml.dom.minidom import Entity

from model.config import Config
import torch
from utils import print_and_save_params, get_data
from train import run_train
from predict import run_test
# from predict_hybrid_bias import run_test
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)  # Only show errors, not warnings


def main():
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument("--model", choices=["bert-idcnn-crf", "bert-lstm-crf", "deberta-crf"], required=True)
    parser.add_argument("--data_path", required=True, default="./data/")
    parser.add_argument("--save_dir", required=True, default="./save_models/")
    parser.add_argument("--word_embedding_model_path", required=True,
                        choices=["/home/mh1f25/scratch/llms_model/bert-base-cased",
                        "/home/mh1f25/scratch/llms_model/deberta-base"])  # e.g., "/home/.../bert-base-cased"

    # Training settings
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_scheduler", type=bool, default=True)


    # IDCNN hyperparams
    parser.add_argument("--num_filters", type=int, default=64)
    parser.add_argument("--cnn_dropout", type=float, default=0.4)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--num_block", type=int, default=4)

    # LSTM hyperparams
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--rnn_dropout", type=float, default=0.1)
    parser.add_argument("--proj_dropout", type=float, default=0.4)

    parser.add_argument("--max_length", type=int, default=400)

    # Labels handling
    parser.add_argument("--include_special_labels", action="store_true", default=True)


    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")


    args = parser.parse_args()

    cfg = Config(args)

    print_and_save_params(cfg, args.save_dir)

    model = cfg.build_model()


    # will be making changes
    train_dataloader, dev_dataloader, test_dataloader, dev_features, test_features = get_data(cfg)

    # Train
    if args.do_train:
        run_train(model, cfg, train_dataloader, dev_dataloader, dev_features)  #to call run_train in train.py

    # # Test
    if args.do_test:

        logger.info('do testing ...')


        best_model_path = cfg.save_path + '_best_f1.pth'
        last_model_path = cfg.save_path + '_last_model.pth'

        # Check if best_f1.pth exists, otherwise load last_model.pth
        if os.path.isfile(best_model_path):
            print   (f"Loading best model from {best_model_path}")
            logger.info(f"Loading best model from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, weights_only=True))
        elif os.path.isfile(last_model_path):
            print  (f"Loading last model from {last_model_path}")
            logger.info(f"Loading last model from {last_model_path}")
            model.load_state_dict(torch.load(last_model_path))
        else:
            raise FileNotFoundError(f"Neither {best_model_path} nor {last_model_path} exists.")

        # 模型训练
        metrics_df = run_test(model, cfg,
                              test_dataloader, test_features=test_features)  # or metrics_df, details_df = run_test(..., return_details=True)
        print(metrics_df)
        # Save where you like (your snippet used config.trained_model_path; if not set, use cfg.save_dir)

        logger.info('------------------test result------------------')
        logger.info(metrics_df)

        metrics_df.to_csv(cfg.save_path + '_test_result.txt', index=True, sep=',', mode='a')



if __name__ == "__main__":

    #  python main.py --model bert-lstm-crf  --data_path ./data/ --save_dir ./save_model --word_embedding_model_path /home/mh1f25/scratch/llms_model/bert-base-cased  --do_train --do_test  --num_epochs 3
    #  python main.py --model deberta-crf  --data_path ./data/ --save_dir ./save_model --word_embedding_model_path /home/mh1f25/scratch/llms_model/deberta-base  --do_train --do_test  --num_epochs 3

    # nohup python main.py --model bert-lstm-crf  --data_path ./data/ --save_dir ./save_model --word_embedding_model_path /home/mh1f25/scratch/llms_model/bert-base-cased  --do_train --do_test  --num_epochs 3  > main.log  2>&1 &



    main()

# one model:
# BERT + LSTM + CRF
# BERT is used to embeddings each token
# LSTM + CRF sequence labeling

# Could be adding rule-based model, create a ACTREE to identify ODRL Entity


#
# third
# T5 + CRF