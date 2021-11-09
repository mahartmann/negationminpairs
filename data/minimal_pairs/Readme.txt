This folder contains 6 datasets of minimal pairs:
- 5 datasets with our minimal pairs created by removing negation markers from original XNLI examples in en,de,fr,bg,zh
- 1 dataset of minimal pairs derived from data of Hossain et al. (2020) (https://github.com/mosharafhossain/negation-and-nli) based on inserting 
  negation markers into original NLI examples

The format of each file is one .json dict per line representing an NLI example. 
The example is either an original NLI example as present in the XNLI or MNLI datasets (*Orig*), 
or a modified version thereof (created by removing a negation cue) from either premise (*Mod1*) or hypothesis (*Mod2*). 
For the Hossain dataset, the modified versions are created by negation cue insertion instead of removal.
Each json dict contains the following information:

- "pairID":    The original id of the NLI example plus an indicator if the example is original, derived by modifying the first sentence, or derived by modifying 
               the second sentence
- "version":   either "Orig", "Mod1", or "Mod2", same as the indicator mentioned above
- "sentence1": NLI premise sentence
- "sentence2": NLI hypothesis sentence

For *Orig* examples:
- "gold_label":         Gold label (as assigned in XNLI dataset)
- "annotator_labels":   Annotator labels (as assigned in XNLI dataset), with an indication of how many annotators (out of 5) assigned the label, 
                        e.g. "contradiction:4,neutral:1" --> 4 annotators assigned a contradiction label, 1 annotator assigned a neutral label
- "sentence1_matches":  parts of the sentence that matched with a negation cue, if any
- "sentence2_matches":  parts of the sentence that matched with a negation cues, if any

For *Mod1* or *Mod2* examples:
- "orig_label":             Gold label (as assigned in XNLI dataset) of the original version of this NLI example
- "orig_annotator_labels":  Annotator labels (as assigned in XNLI dataset) of the original version of this NLI example
- "gold_label":             Manually assigned new gold label for the modified version

"orig_label" and "gold_label" indicate if the example belongs to a minimal pair with important or unimportant negation:
- orig_label == gold_label --> unimportant negation
- orig_label != gold_label --> important negation




