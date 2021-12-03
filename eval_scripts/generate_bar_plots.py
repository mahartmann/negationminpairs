import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import style as style


def generate_bar_plot_unimportant_negations(lang2info1, lang2info2, target_langs):
    labels = [('entailment', 'entailment'), ('neutral', 'neutral'), ('contradiction', 'contradiction')]

    sns.set_style("white", {'font.family': [u'serif'], 'font.serif': [u'Times New Roman']})
    matplotlib.rcParams.update({'font.size': 12})
    style.use('seaborn-colorblind')
    current_palette = sns.color_palette()
    colors = [current_palette[0], current_palette[1], current_palette[2], current_palette[4], current_palette[5]]
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)  #
    ax.spines['bottom'].set_position("zero")

    ax.grid(axis='y')

    plt.ylim((0, 100))

    x_ticks = []
    pos = 0
    for l1, l2 in labels:
        heights = []
        for lang in target_langs:
            info = lang2info2[lang]
            total_l1l2 = 0.
            correct1 = 0.
            correct2 = 0.
            bar_width = 0.8
            for scores in info:
                gold_orig = scores[0]
                gold_mod = scores[1]
                pred_orig = scores[2]
                pred_mod = scores[3]
                if gold_orig == l1 and gold_mod == l2:
                    total_l1l2 += 1
                    if pred_orig == gold_orig:
                        correct1 += 1
                        if pred_mod == gold_mod:
                            correct2 += 1
            p1 = correct1 / total_l1l2
            p2 = correct2 / total_l1l2
            heights.append(((p1 - p2) / p1) * 100)
        x_pos = [pos + elm * bar_width for elm in range(len(heights))]
        bar = plt.bar(x=x_pos, height=heights, width=bar_width, color=colors)
        x_ticks.append((x_pos[0] + x_pos[-1]) / 2)
        pos += len(heights) * bar_width + 2

    pos = 0

    for l1, l2 in labels:
        heights = []
        for lang in target_langs:
            info = lang2info1[lang]
            total_l1l2 = 0.
            correct1 = 0.
            correct2 = 0.
            bar_width = 0.8
            for scores in info:
                gold_orig = scores[0]
                gold_mod = scores[1]
                pred_orig = scores[2]
                pred_mod = scores[3]
                if gold_orig == l1 and gold_mod == l2:
                    total_l1l2 += 1
                    if pred_orig == gold_orig:
                        correct1 += 1
                        if pred_mod == gold_mod:
                            correct2 += 1
            p1 = correct1 / total_l1l2
            p2 = correct2 / total_l1l2
            heights.append(((p1 - p2) / p1) * 100)
        x_pos = [pos + elm * bar_width for elm in range(len(heights))]
        bar = plt.bar(x=x_pos, height=heights, width=bar_width, facecolor='none')
        # x_ticks.append((x_pos[0] + x_pos[-1])/2)
        pos += len(heights) * bar_width + 2
        # Loop over the bars
        for i, thisbar in enumerate(bar.patches):
            # Set a different hatch for each bar
            # if i == 2:
            thisbar.set_hatch('..')
            thisbar.set_edgecolor('k')

    labels = target_langs

    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i, _ in enumerate(labels)]
    add_label = 'M$^-$→M$^+_O$'
    patches.append(mpatches.Patch(facecolor='none', label=add_label, hatch='..', edgecolor='k'))
    plt.legend(handles=patches, labels=labels + [add_label], loc=2)
    plt.xticks(x_ticks, ['E→E', 'N→N', 'C→C'])
    plt.ylabel('Change in performance in %')
    plt.title('Unimportant negations on M$^-$ → M$^\pm_O$')
    #plt.savefig('./plots/performance_loss_unimp_pos.pdf', format='pdf')
    plt.show()


def generate_bar_plot_important_negations(lang2info1, lang2info2, target_langs):
    labels = [('entailment', 'contradiction'), ('contradiction', 'entailment'),
              ('neutral', 'contradiction'),
              ('contradiction', 'neutral')
              ]
    # find all that have l1-l2 as orig-mod gold labels
    # record true positives, false positives, false negatives

    sns.set_style("white", {'font.family': [u'serif'], 'font.serif': [u'Times New Roman']})
    style.use('seaborn-colorblind')
    current_palette = sns.color_palette()
    colors = [current_palette[0], current_palette[1], current_palette[2], current_palette[4], current_palette[5]]
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)  #
    ax.spines['bottom'].set_position("zero")

    ax.grid(axis='y')
    plt.ylim((0, 100))

    pos = 0
    x_ticks = []

    for l1, l2 in labels:
        heights = []
        for lang in target_langs:
            info = lang2info2[lang]
            total_l1l2 = 0.
            correct1 = 0.
            correct2 = 0.
            bar_width = 0.8
            for scores in info:
                gold_orig = scores[0]
                gold_mod = scores[1]
                pred_orig = scores[2]
                pred_mod = scores[3]
                if gold_orig == l1 and gold_mod == l2:
                    total_l1l2 += 1
                    if pred_orig == gold_orig:
                        correct1 += 1
                        if pred_mod == gold_mod:
                            correct2 += 1
            p1 = correct1 / total_l1l2
            p2 = correct2 / total_l1l2
            heights.append(((p1 - p2) / p1) * 100)
        x_pos = [pos + elm * bar_width for elm in range(len(heights))]
        bar = plt.bar(x=x_pos, height=heights, width=bar_width, color=colors)

        x_ticks.append((x_pos[0] + x_pos[-1]) / 2)
        pos += len(heights) * bar_width + 2
    pos = 0
    for l1, l2 in labels:
        heights = []
        for lang in target_langs:
            info = lang2info1[lang]
            total_l1l2 = 0.
            correct1 = 0.
            correct2 = 0.
            bar_width = 0.8
            for scores in info:
                gold_orig = scores[0]
                gold_mod = scores[1]
                pred_orig = scores[2]
                pred_mod = scores[3]
                if gold_orig == l1 and gold_mod == l2:
                    total_l1l2 += 1
                    if pred_orig == gold_orig:
                        correct1 += 1
                        if pred_mod == gold_mod:
                            correct2 += 1
            p1 = correct1 / total_l1l2
            p2 = correct2 / total_l1l2
            heights.append(((p1 - p2) / p1) * 100)
        x_pos = [pos + elm * bar_width for elm in range(len(heights))]
        bar = plt.bar(x=x_pos, height=heights, width=bar_width, facecolor='none')

        # x_ticks.append((x_pos[0] + x_pos[-1])/2)
        pos += len(heights) * bar_width + 2
        # Loop over the bars
        for i, thisbar in enumerate(bar.patches):
            # Set a different hatch for each bar
            # if i == 2:
            thisbar.set_hatch('..')
            thisbar.set_edgecolor('k')

    labels = target_langs
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i, _ in enumerate(labels)]
    # plt.legend(handles=patches, labels=labels,ncol=5)
    plt.title('Important negations on M$^-$ → M$^\pm_O$')
    plt.xticks(x_ticks, ['E→C', 'C→E', 'N→C', 'C→N'])
    #plt.savefig('./plots/performance_loss_imp_pos.pdf', format='pdf')
    plt.show()

