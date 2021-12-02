"""
Train BERT on NLI dataset
"""

import random
import numpy as np
import torch

from sentence_transformers.readers import NLIReader
from sentence_transformers.datasets import NLIDataset
from sentence_transformers.util import create_logger, bool_flag

from torch.utils.data import DataLoader

from transformers import BertTokenizer,XLMRobertaTokenizer
import configparser
import os
import argparse
import uuid
from training import NLIBert
from transformers.data.data_collator import DataCollatorWithPadding
from training.optimization import get_optimizer, get_scheduler



def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # check if output dir exists. if so, assign a new one
    if os.path.isdir(args.outdir):
        # create new output dir
        outdir = os.path.join(args.outdir, str(uuid.uuid4()))
    else:
        outdir = args.outdir

    # make the output dir
    os.makedirs(outdir)
    if args.save_best:
        os.makedirs(os.path.join(outdir, 'best_model'))

    # find out device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # create a logger
    logger = create_logger(__name__, to_disk=True, log_file='{}/{}'.format(outdir, args.logfile))
    logger.info('Created new output dir {}'.format(outdir))
    logger.info('Running experiments on {}'.format(device))

    # get config with all data locations
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config)

    # load train data
    if args.bert_model == 'small_bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif args.bert_model == 'xlm-roberta-base':
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.bert_model)
    else: tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    logger.info('Tokenizing with {} tokenizer'.format(tokenizer.name_or_path))
    reader = NLIReader.NLIReader(ds=args.ds)
    label_map = reader.get_labels()
    train_data = NLIDataset(reader.get_examples(config.get('Files', '{}_train'.format(args.ds)), max_examples=args.max_examples), tokenizer=tokenizer,
                            max_seq_len=args.max_seq_len, hypo_only=args.hypo_only)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.bs, collate_fn=collator)
    logger.info('Loaded {} train examples from {}'.format(len(train_data), args.ds))

    # load dev data
    dev_data = NLIDataset(reader.get_examples(config.get('Files', '{}_dev'.format(args.ds)), max_examples=args.max_examples), tokenizer=tokenizer,
                            max_seq_len=args.max_seq_len, hypo_only=args.hypo_only)

    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.bs, collate_fn=collator)
    logger.info('Loaded {} dev examples from {}'.format(len(dev_data), args.ds))

    # load test data
    if args.predict:
        test_data = NLIDataset(reader.get_examples(config.get('Files', '{}_test'.format(args.ds))),
                              tokenizer=tokenizer,
                              max_seq_len=args.max_seq_len, hypo_only=args.hypo_only)

        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.bs, collate_fn=collator)
        logger.info('Loaded {} test examples from {}'.format(len(test_data), args.ds))

    else: test_dataloader=None


    #load model
    model = NLIBert(checkpoint=args.bert_model, label_map=label_map, device=device, num_labels=len(reader.get_labels()))

    # get optimizer
    optimizer = get_optimizer(model, lr=args.lr, eps=args.eps, decay=args.decay)

    # get schedule
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = args.warmup_frac * total_steps
    logger.info('Scheduler: {} with {} warmup steps'.format(args.scheduler, warmup_steps))
    scheduler = get_scheduler(optimizer, scheduler=args.scheduler, warmup_steps=warmup_steps, t_total=total_steps)

    model.fit(optimizer=optimizer,
              scheduler=scheduler,
              train_dataloader=train_dataloader,
              dev_dataloader=dev_dataloader,
              test_dataloader=test_dataloader,
              epochs=args.epochs,
              evaluation_step=args.evaluation_steps,
              save_best = args.save_best,
              outdir=outdir,
              grad_accumulation_steps=args.grad_accumulation_steps,
              predict=args.predict)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
          description='Train SentenceBert with analogy data')
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")

    parser.add_argument('--bert_model', type=str,
                        default='small_bert',
                        choices=['bert-base-multilingual-cased', 'bert-base-uncased', 'small_bert'],
                        help="The pre-trained encoder used to encode the entities of the analogy")
    parser.add_argument('--config', default='./config.cfg')
    parser.add_argument("--hypo_only", type=bool_flag, default=False)
    parser.add_argument('--ds', type=str,
                        help="Type of input data. Mnlitoy can be used for debugging", default='mnlitoy', choices = ['mnli', 'mnlitoy'])
    parser.add_argument('--outdir', type=str,
                        help="output path", default='out')
    parser.add_argument('--logfile', type=str,
                        help="name of log file", default='nli_model.log')
    parser.add_argument('--bs', type=int, default=2,
                        help="Batch size")
    parser.add_argument('--grad_accumulation_steps', type=int, default=4,
                        help="Steps over which the gradient is accumulated")
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help="Max seq length")
    parser.add_argument('--max_examples', type=int, default=-1,
                        help="Number of examples read from the datasets (useful for debugging). Set to -1 to read all data")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument('--evaluation_steps', type=int, default=2,
                        help="Evaluate every n training steps")
    parser.add_argument("--save_best", type=bool_flag, default=True)
    parser.add_argument("--predict", type=bool_flag, default=False)

    # Optimization parameters
    parser.add_argument("--decay", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--scheduler", type=str, default='warmuplinear')
    parser.add_argument("--lr", type=float, default=5e-5)



    args = parser.parse_args()

    main(args)
