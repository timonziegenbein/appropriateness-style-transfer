import json
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon
from scipy.stats import ttest_ind

RESULTDIMS = [
    'eval_Appropriateness_macroF1',
    'eval_Appropriateness_precision',
    'eval_Appropriateness_recall',
    'eval_Intelligibility_macroF1',
    'eval_Intelligibility_precision',
    'eval_Intelligibility_recall',
    'eval_Intelligible Organization_macroF1',
    'eval_Intelligible Organization_precision',
    'eval_Intelligible Organization_recall',
    'eval_Intelligible Position_macroF1',
    'eval_Intelligible Position_precision',
    'eval_Intelligible Position_recall',
    'eval_Intelligible Relevance_macroF1',
    'eval_Intelligible Relevance_precision',
    'eval_Intelligible Relevance_recall',
    'eval_Commitment_macroF1',
    'eval_Commitment_precision',
    'eval_Commitment_recall',
    'eval_Committed Openness_macroF1',
    'eval_Committed Openness_precision',
    'eval_Committed Openness_recall',
    'eval_Committed Seriousness_macroF1',
    'eval_Committed Seriousness_precision',
    'eval_Committed Seriousness_recall',
    'eval_Emotional Intensity_macroF1',
    'eval_Emotional Intensity_precision',
    'eval_Emotional Intensity_recall',
    'eval_Emotional Typology_macroF1',
    'eval_Emotional Typology_precision',
    'eval_Emotional Typology_recall',
    'eval_Emotions_macroF1',
    'eval_Emotions_precision',
    'eval_Emotions_recall',
    'eval_Not classified_macroF1',
    'eval_Not classified_precision',
    'eval_Not classified_recall',
    'eval_Orthography_macroF1',
    'eval_Orthography_precision',
    'eval_Orthography_recall',
    'eval_Other_macroF1',
    'eval_Other_precision',
    'eval_Other_recall',
    'eval_mean_F1',
    'eval_mean_precision',
    'eval_mean_recall'
]

model_dir = '../../data/models/'

approaches = [
#    'multilabel-roberta-conservative',
#    'multilabel-roberta-majority',
#    'multilabel-roberta-full',

#    'multilabel-debertav3-majority',
    'multilabel-debertav3-conservative',
    'multilabel-debertav3-conservative-wo-issue',
    'multilabel-debertav3-conservative-shuffle',
]


test_dict = {x: [] for x in RESULTDIMS}
for approach in approaches:
    tmp_results = []
    for repeat in range(5):
        for k in range(5):
            with open(model_dir+approach+'/fold{}/fold{}.{}/test_results.json'.format(repeat,k,repeat), 'r') as f:
                tmp_result = json.load(f)
            tmp_results.append(tmp_result)
    d = {}
    for k, _ in tmp_results[0].items():
        d[k] = np.mean([d[k] for d in tmp_results])
    for dim in RESULTDIMS:
        test_dict[dim].append(d[dim])

test_dict['approach'] = approaches

df = pd.DataFrame(data=test_dict)

### Print F1-scores (table 4 in the paper)
print(df[[
   'approach',
    'eval_mean_F1',
    'eval_Appropriateness_macroF1',
    'eval_Emotions_macroF1',
    'eval_Emotional Intensity_macroF1',
    'eval_Emotional Typology_macroF1',
    'eval_Commitment_macroF1',
    'eval_Committed Seriousness_macroF1',
    'eval_Committed Openness_macroF1',
    'eval_Intelligibility_macroF1',
    'eval_Intelligible Position_macroF1',
    'eval_Intelligible Relevance_macroF1',
    'eval_Intelligible Organization_macroF1',
    'eval_Other_macroF1',
    'eval_Orthography_macroF1',
    'eval_Not classified_macroF1',
    ]].sort_values('eval_mean_F1', ascending=False).round(4))


with open('mytable.tex', 'w') as tf:
    tf.write(df[[
       'approach',
        'eval_mean_F1',
        'eval_Appropriateness_macroF1',
        'eval_Emotions_macroF1',
        'eval_Emotional Intensity_macroF1',
        'eval_Emotional Typology_macroF1',
        'eval_Commitment_macroF1',
        'eval_Committed Seriousness_macroF1',
        'eval_Committed Openness_macroF1',
        'eval_Intelligibility_macroF1',
        'eval_Intelligible Position_macroF1',
        'eval_Intelligible Relevance_macroF1',
        'eval_Intelligible Organization_macroF1',
        'eval_Other_macroF1',
        'eval_Orthography_macroF1',
        'eval_Not classified_macroF1',
        ]].sort_values('eval_mean_F1', ascending=False).to_latex(index=False, float_format="{:0.2f}".format))
test_dict = {x: [] for x in RESULTDIMS}
for approach in approaches:
    tmp_results = []
    for repeat in range(5):
        for k in range(5):
            with open(model_dir+approach+'/fold{}/fold{}.{}/test_results.json'.format(repeat,k,repeat), 'r') as f:
                tmp_result = json.load(f)
            tmp_results.append(tmp_result)
    for dim in RESULTDIMS:
        test_dict[dim].append([x[dim] for x in tmp_results])

### Check significance of all approaches
for dim in RESULTDIMS:
    if 'F1' in dim:
        for i, approach1 in enumerate(approaches):
            for j, approach2 in enumerate(approaches):
                if i<j:
                    w, p = wilcoxon(test_dict[dim][i], test_dict[dim][j], mode='exact')
                    print((dim, approach1, approach2))
                    print(p<=0.5)
