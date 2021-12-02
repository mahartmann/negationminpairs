from torch.utils.data import Dataset
from typing import List
import torch
import logging
from tqdm import tqdm
from sentence_transformers.readers.InputExample import NLIInputExample
from transformers import PreTrainedTokenizer


class NLIDataset(Dataset):
    """
    """
    def __init__(self, examples: List[NLIInputExample], tokenizer, max_seq_len,  show_progress_bar: bool = None, hypo_only=False):
        """
        Create a new SentencesDataset with the tokenized texts and the labels as Tensor
        """
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.hypo_only = hypo_only
        self.pairIds = []
        self.convert_input_examples(examples, tokenizer)


    def convert_input_examples(self, examples: List[NLIInputExample], tokenizer: PreTrainedTokenizer):
        """
        """

        inputs = []
        labels = []

        label_type = None
        iterator = examples
        max_seq_length = self.max_seq_len

        if self.show_progress_bar:
            iterator = tqdm(iterator, desc="Convert dataset")
        logging.info('Considering hypothesis only: {}'.format(self.hypo_only))
        for ex_index, example in enumerate(iterator):

            if label_type is None:
                if isinstance(example.label, int):
                    label_type = torch.long
                elif isinstance(example.label, float):
                    label_type = torch.float

            labels.append(example.label)

            if self.hypo_only is False:
                inputs.append(tokenizer.encode_plus(text = example.premise, text_pair = example.hypo,  padding='longest'))
            elif self.hypo_only is True:
                inputs.append(tokenizer.encode_plus(text=example.hypo,  padding='longest'))

            self.pairIds.append(example.guid)
        tensor_labels = torch.tensor(labels, dtype=label_type)

        logging.info("Num sentences: %d" % (len(examples)))

        self.tokens = inputs
        self.labels = tensor_labels

    def __getitem__(self, i):
        elm = dict(self.tokens[i])
        elm.update({ 'label': self.labels[i]})
        return elm

    def __len__(self):
        return len(self.tokens)



