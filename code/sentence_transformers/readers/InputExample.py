from typing import Union, List


class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str, texts: List[str], label: Union[int, float]):
        """
        Creates one InputExample with the given texts, guid and label

        str.strip() is called on both texts.

        :param guid
            id for the example
        :param texts
            the texts for the example
        :param label
            the label for the example
        """
        self.guid = guid
        self.pairId = pairId
        self.texts = [text.strip() for text in texts]
        self.label = label

class NLIInputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str, premise: str, hypo: str, label: str):
        """
        Creates one InputExample with the given texts, guid and label

        str.strip() is called on both texts.

        :param guid
            id for the example
        :param texts
            the texts for the example
        :param label
            the label for the example
        """
        self.guid = guid
        self.premise = premise
        self.hypo = hypo
        self.label = label