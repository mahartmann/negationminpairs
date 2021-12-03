# A Multilingual Benchmark for Probing Negation-Awareness <br/> with Minimal Pairs
This repository contains the data for our Multilingual Benchmark for Probing Negation-Awareness with Minimal Pairs (CoNLL 2021). The idea of our work is to provide multilingual datasets that allow to investigate the extent to which pre-trained language models are aware of the semantics of negation markers. The datasets are manually derived from the multilingual [XNLI](https://github.com/facebookresearch/XNLI) datasets, and consist of minimal pairs of NLI examples that only differ in the presence/absence of a negation marker. For more information, checkout the associated [video](https://drive.google.com/file/d/1TaF1vYWRyedG-uNJzdBak7lHls6up06l/view?usp=sharing) and [poster](./poster_negation_conll.pdf).

The repository contains the following data:
- the [lists of negation cues](./data/negation_cues) used to select NLI examples for the minimal pairs
- the [datasets of minimal pairs](./data/minimal_pairs) resulting from negation removal


## Model Evaluation
We suggest to evaluate negation awareness on minimal pairs by comparing the fraction of correct predictions on the original NLI example (correct prediction for first element of the minimal pair) with the fraction of correct predictions on the original NLI example AND the corresponding modified NLI example (correct predictions for both elements of the minimal pair). To replicate the results in our paper:
1. Download the BERT model fine-tuned on MNLI data from [here](https://drive.google.com/file/d/1iY7Es6jxjnWGn5ntGXxrZ7U2Nrq3lUa0/view?usp=sharing). Put it in the `./trained_models` folder and unzip the file. 
2. Get predictions for minimal pairs by running `./eval_scripts/run_predict_multilingual.sh`, which calls the python code for model evaluation in `code/training/predict_nli.py`. The predictions are written to the `./results` folder.
3. Compute difference in performance on `orig` and `orig AND modified` examples using `./eval_scripts/compute_performance_loss.py`. By uncommenting the last two lines in the script, you can generate the barplots shown in Figure 4 in the paper.



## Citing and Contact 
If you have questions or comments, please contact the corresponding author at `mrkhartmann4@gmail.com` (Mareike Hartmann).

