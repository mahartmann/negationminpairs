"""
Predict NLI dataset using a fine-tuned BERT model
"""

import random
import numpy as np
import torch
import json
import configparser
import os
import argparse


from sentence_transformers.readers import NLIReader
from sentence_transformers.datasets import NLIDataset
from sentence_transformers.util import create_logger, dump_json, bool_flag

from torch.utils.data import DataLoader
from transformers import BertTokenizer,XLMRobertaTokenizer
from training import NLIBert
from transformers.data.data_collator import DataCollatorWithPadding


def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # check if output dir exists. if not, make a new one
    outdir = args.outdir
    if not os.path.isdir(args.outdir):
        # create new output dir
        os.makedirs(outdir)


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
    if args.tokenizer == 'small_bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif args.tokenizer == 'xlm-roberta-base':
        tokenizer = XLMRobertaTokenizer.from_pretrained(args.tokenizer)
    else: tokenizer = BertTokenizer.from_pretrained(args.tokenizer)
    logger.info('Tokenizing with {} tokenizer'.format(tokenizer.name_or_path))
    reader = NLIReader.NLIReader()
    label_map = reader.get_labels()


    # load test data
    test_dataloaders = []
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    test_dataset_names = args.test_data.split(',')
    test_dataset_ids = []
    for ds in test_dataset_names:
        test_data = NLIDataset(reader.get_examples(os.path.join(args.data_path, ds)),
                              tokenizer=tokenizer,
                              max_seq_len=args.max_seq_len, hypo_only=args.hypo_only)

        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=args.bs, collate_fn=collator)
        test_dataset_ids.append(test_data.pairIds)
        logger.info('Loaded {} test examples from {}'.format(len(test_data), ds))

        test_dataloaders.append(test_dataloader)



    #load model

    model = NLIBert(checkpoint=os.path.join(args.model_dir, args.model_name), label_map=label_map, device=device, num_labels=len(reader.get_labels()))
    for ds, pairIds, test_dataloader in zip(test_dataset_names, test_dataset_ids, test_dataloaders):
        test_score, test_results, test_predictions = model.evaluate_on_dev(data_loader=test_dataloader)

        # dump to file
        model_name = args.model_name.replace('/', '_')
        ds = parse_ds_name(ds)

        # get pairIds and map predictions to pairIds

        mapped_predictions = {}
        for pairId, pred in zip(pairIds, test_predictions):
            mapped_predictions[pairId] = model.id2label[pred]


        logger.info('Results for {} on {}'.format(model_name, ds))
        logger.info('Logging to {}'.format(os.path.join(outdir, '{}_{}.json'.format(model_name, ds))))
        logger.info('Score:  {} '.format(test_score))
        logger.info('Results:  {}'.format(json.dumps(test_results, indent=4)))


        dump_json(fname=os.path.join(outdir, '{}_{}.json'.format(model_name, ds)),
                  data={'score': test_score, 'results': test_results, 'predictions': test_predictions}, indent=4)
        dump_json(fname=os.path.join(outdir, '{}_{}_predictions.json'.format(model_name, ds)),
                  data=mapped_predictions, indent=4)

def parse_ds_name(ds):
    ds = ds.split('/')[-1]
    splt = ds.split('.')
    ds = '.'.join([elm for elm in splt[:-1]])
    return ds

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
          description='Predict NLI data using a fine-tuned BERT model')
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")
    parser.add_argument('--tokenizer', type=str,
                        help="Name of tokenizer used to tokenize the test data",
                        default='bert-base-cased')
    parser.add_argument('--model_name', type=str,
                        help="Name of the model used for loading and logging results",
                        default='ce572990-e68f-4b8b-87de-5d7f130febc4/best_model')
    parser.add_argument('--model_dir', type=str,
                        help="Model directory, used as prefix for loading the model specified in model_name",
                        default='/home/mareike/PycharmProjects/negation/code/negationsensitiveLM/training/out/')
    parser.add_argument("--hypo_only", type=bool_flag, default=False)
    parser.add_argument('--config', default='/home/mareike/PycharmProjects/negation/code/negationsensitiveLM/training/config.cfg')
    parser.add_argument('--data_path', type=str,
                        help="data directory used as prefix for test data sets", default='/home/mareike/PycharmProjects/negation/eval_data')
    parser.add_argument('--test_data', type=str,
                        help="list of test data sets", default='xnli_bg.neg.test.jsonl')
    parser.add_argument('--outdir', type=str,
                        help="output path", default='results')
    parser.add_argument('--logfile', type=str,
                        help="name of log file", default='nli_model.log')
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help="Max seq length")
    parser.add_argument('--bs', type=int, default=8,
                        help="Batch size")
    parser.add_argument('--max_examples', type=int, default=-1,
                        help="Number of examples read from the datasets (useful for debugging). Set to -1 to read all data")


    args = parser.parse_args()

    main(args)
