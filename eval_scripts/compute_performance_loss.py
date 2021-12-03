import json
from generate_bar_plots import generate_bar_plot_unimportant_negations, generate_bar_plot_important_negations


def load_json(fname):
    with open(fname) as f:
        return json.load(f)


def get_minimal_pair_prediction_info(ds, test_data, pid2pred, pos_only):
    # compute the percentage of correctly solved minimal pairs
    # gold_orig gold_mod pred_orig pred_mod
    pid2idx = {}
    idx2pid = {}

    def gen_key(elm1, elm2):
        return ':'.join([elm1, elm2])

    info = [[0, 0, 0, 0] for i, elm in test_data.items() if not elm['pairID'].endswith('Orig')]
    if pos_only is True:
        # Identify the subset of test examples that have exactly one negation in premise XOR exactly one negation in hypothesis
        test_data_pairs = {}
        for pid, elm in test_data.items():
            test_data_pairs.setdefault(pid.split('_')[0], []).append(pid)
        test_data_pos = {}
        for p, pids in test_data_pairs.items():
            if len(pids) == 2:
                for pid in pids:
                    test_data_pos[pid] = test_data[pid]
        target_data = test_data_pos
    else:
        target_data = test_data

    for elm in target_data.values():
        pid_ = '{}.jsonl-{}'.format(ds, elm['pairID'])
        pid_stem = '_'.join(elm['pairID'].split('_')[:-1])
        pid = elm['pairID']
        version = elm['pairID'].split('_')[-1]
        if version != 'Orig':
            pid_orig = pid_stem + '_Orig'
            pid_orig_ = '{}.jsonl-{}'.format(ds, pid_orig)
            orig = test_data[pid_orig]

            key = gen_key(pid_orig, pid)

            pid2idx[key] = len(pid2idx)
            idx2pid[len(pid2idx)] = key
            info[pid2idx[key]][0] = orig['gold_label']
            info[pid2idx[key]][2] = pid2pred[pid_orig_]

            info[pid2idx[key]][1] = elm['gold_label']
            info[pid2idx[key]][3] = pid2pred[pid_]

    label2idx = {label: idx for idx, label in
                 enumerate(['entailment-entailment', 'neutral-neutral', 'contradiction-contradiction',
                            'contradiction-entailment', 'neutral-entailment',
                            'entailment-neutral', 'contradiction-neutral',
                            'neutral-contradiction', 'entailment-contradiction'
                            ])}

    idx2label = {idx: label for label, idx in label2idx.items()}
    pred2idx = {pred: idx for idx, pred in enumerate(['entailment', 'neutral', 'contradiction'])}
    idx2pred = {idx: pred for pred, idx in pred2idx.items()}
    return info, label2idx, idx2label, pred2idx, idx2pred


def compute_performance_loss(lang2info, target_labels):
    # find all that have l1-l2 as orig-mod gold labels
    # record true positives, false positives, false negatives
    langs = lang2info.keys()
    for lang in langs:
        print('\nLanguage: {}'.format(lang))
        info = lang2info[lang]
        for l1, l2 in target_labels:
            total_l1l2 = 0.
            correct1 = 0.
            correct2 = 0.
            for scores in info:
                gold_orig = scores[0]
                gold_mod = scores[1]
                pred_orig = scores[2]
                pred_mod = scores[3]
                if gold_orig == l1 and gold_mod == l2:
                    total_l1l2 += 1
                    # check if first element of minimal pair is predicted correctly
                    if pred_orig == gold_orig:
                        correct1 += 1
                        # check if both elements of minimal pair is predicted correctly
                        if pred_mod == gold_mod:
                            correct2 += 1
            p1 = correct1 / total_l1l2
            p2 = correct2 / total_l1l2
            performance_loss = (((p1 - p2) / p1) * 100)
            print('\n{} -> {}:\nCorrect M1:\t{:.2f}\nCorrect M1 AND M2:\t{:.2f}\nPerformance loss:\t{:.2f}'.format(l1, l2, p1, p2, performance_loss))






if __name__=="__main__":

    langs = ['en', 'bg', 'de', 'fr', 'zh']
    lang2info_allminpairs= {}
    lang2info_posonlysubset = {}
    result_dir = '../results'
    data_dir = '../data/minimal_pairs/eval_splits'
    model = 'bert-base-multilingual-cased#mnli_bs8_gradacc4_lr2e-5_maxlen128split_warmup0_epochs3'
    run = 1
    for lang in langs:
        test_data = {}
        pid2pred = {}
        ds = 'minimal_pairs_All_xnli_{}.test'.format(lang)
        with open('{}/{}.jsonl'.format(data_dir, ds)) as f:
            for line in f:
                d = json.loads(line.strip())
                test_data[d['pairID']] = d
        f.close()

        # load predictions
        with open('{}/{}_run{}_best_model_{}_predictions.json'.format(result_dir, model, run, ds)) as f:
            pid2pred = json.load(f)
        f.close()

        info, label2idx, idx2label, pred2idx, idx2pred = get_minimal_pair_prediction_info(ds, test_data, pid2pred, pos_only=False)
        lang2info_allminpairs[lang] = info

        info, label2idx, idx2label, pred2idx, idx2pred = get_minimal_pair_prediction_info(ds, test_data, pid2pred, pos_only=True)
        lang2info_posonlysubset[lang] = info



    # Evaluate on all minimal pairs
    # performance on unimportant negations
    print('# UNIMPORTANT NEGATIONS')
    compute_performance_loss(lang2info_allminpairs, target_labels=[('entailment', 'entailment'),
                                                        ('neutral', 'neutral'),
                                                        ('contradiction', 'contradiction')])
    # performance on important negations
    print('\n\n# IMPORTANT NEGATIONS')
    compute_performance_loss(lang2info_allminpairs, target_labels=[('entailment','contradiction'),
                                                        ('contradiction','entailment'),
                                                        ('neutral','contradiction'),
                                                        ('contradiction','neutral')])

    # Evaluate on the subset of minimal pairs that contain exactly one negation in premise XOR exactly one negation in hypothesis (dotted bars in Figure 4)
    # performance on unimportant negations
    print('\n\n# M- --> M+ subset')
    print('# UNIMPORTANT NEGATIONS')
    compute_performance_loss(lang2info_posonlysubset, target_labels=[('entailment', 'entailment'),
                                                                   ('neutral', 'neutral'),
                                                                   ('contradiction', 'contradiction')])
    # performance on important negations
    print('\n\n# IMPORTANT NEGATIONS')
    compute_performance_loss(lang2info_posonlysubset, target_labels=[('entailment', 'contradiction'),
                                                                   ('contradiction', 'entailment'),
                                                                   ('neutral', 'contradiction'),
                                                                   ('contradiction', 'neutral')])

    # Uncomment the following code to produce the barplots in Figure 4
    #generate_bar_plot_unimportant_negations(lang2info_posonlysubset, lang2info_allminpairs, target_langs=langs)
    #generate_bar_plot_important_negations(lang2info_posonlysubset, lang2info_allminpairs, target_langs=langs)