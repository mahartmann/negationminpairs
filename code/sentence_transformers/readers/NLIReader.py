from sentence_transformers.readers import InputExample, NLIInputExample
import json

class NLIReader(object):
    """
    Reads in a dataset of negated analogies A is to not A as B is to not B
    """

    def __init__(self, ds='mnli'):
        if ds == 'snli' or ds=='mnli' or ds=='mnlitoy' or ds=='mnlifiltered':
            self.idstring = 'pairID'

    def get_examples(self, filename,  max_examples=0):
        """
        """
        examples = []

        for elm in load_data(filename):
            eid = elm[self.idstring]
            guid = "{}-{}".format(filename.split('/')[-1], eid)

            premise = elm['sentence1']
            hypo = elm['sentence2']
            if elm['gold_label'] == '-':
                continue

            label = self.map_label(elm['gold_label'])
            examples.append(NLIInputExample(guid=guid,  premise=premise, hypo=hypo, label=label))
            if max_examples > 0:
                if len(examples) >= max_examples:break
        return examples


    @staticmethod
    def get_labels():
        return {"contradiction": 0, "entailment": 1, "neutral": 2}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]

def load_data(fname):
    with open(fname) as f:
        for line in f:
            elm = json.loads(line)
            yield elm



