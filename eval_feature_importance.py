import argparse
import json
import pickle as pkl
import pandas as pd
import numpy as np
from rac import RAClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# CMD LINE ARGS PARSING

parser = argparse.ArgumentParser(description='Feature importance evaluation')
parser.add_argument(
    '--dataset',
    action='store',
    type=str,
    required=True,
    help="Path to the .pkl file with the dataset."
)
parser.add_argument(
    '--rac-params',
    action='store',
    type=str,
    required=True,
    help="Path to the .json file with RAC parameters."
)
parser.add_argument(
    '--svm-params',
    action='store',
    type=str,
    required=True,
    help="Path to the .json file with SVM parameters."
)
parser.add_argument(
    '--rf-params',
    action='store',
    type=str,
    required=True,
    help="Path to the .json file with RF parameters."
)
parser.add_argument(
    '--path',
    action='store',
    type=str,
    required=True,
    help="The path where to save results."
)
parser.add_argument(
    '--name',
    action='store',
    type=str,
    required=True,
    help="The name of the dataset (plot title)."
)

args = parser.parse_args()
dataset = args.dataset
rac_params_path = args.rac_params
svm_params_path = args.svm_params
rf_params_path = args.rf_params
path = args.path
name = args.name

with open(dataset, 'rb') as fp :
    dict = pkl.load(fp)
X = dict['X']
y = dict['y']
feature_ids = dict['features_ids']

with open(rac_params_path) as rac_params_fp:
    rac_params = json.load(rac_params_fp)

with open(svm_params_path) as svm_params_fp:
    svm_params = json.load(svm_params_fp)

with open(rf_params_path) as rf_params_fp:
    rf_params = json.load(rf_params_fp)

# FITTING CLASSIFIERS

rac = RAClassifier(**rac_params)
rac.fit(X, y)

svm = SVC(**svm_params)
svm.fit(X, y)

rf = RandomForestClassifier(**rf_params)
rf.fit(X, y)

# FEATURE IMPORTANCE HISTOGRAM

N_TOP_FEATURES = 20

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7))

rac_idx = np.flip(np.argsort(rac.feature_importances_))
rac_feature_rank = [feature_ids[i] for i in rac_idx]
ax1.bar(range(N_TOP_FEATURES), rac.feature_importances_[rac_idx[:N_TOP_FEATURES]]/X.shape[1], align='center')
ax1.set_title('RAC')
ax1.set_ylabel('Feature importance')
ax1.set_axisbelow(True)
ax1.grid(axis='y')
ax1.set_xticks(range(N_TOP_FEATURES), rac_feature_rank[:N_TOP_FEATURES], rotation = 90)

svm_idx = np.flip(np.argsort(svm.coef_[0]))
svm_feature_rank = [feature_ids[i] for i in svm_idx]
ax2.bar(range(N_TOP_FEATURES), svm.coef_[0][svm_idx[:N_TOP_FEATURES]], align='center')
ax2.set_title('SVM')
ax2.set_ylabel('Feature importance')
ax2.set_axisbelow(True)
ax2.grid(axis='y')
ax2.set_xticks(range(N_TOP_FEATURES), svm_feature_rank[:N_TOP_FEATURES], rotation = 90)

rf_idx = np.flip(np.argsort(rf.feature_importances_))
rf_feature_rank = [feature_ids[i] for i in rf_idx]
ax3.bar(range(N_TOP_FEATURES), rf.feature_importances_[rf_idx[:N_TOP_FEATURES]], align='center')
ax3.set_title('RF')
ax3.set_ylabel('Feature importance')
ax3.set_axisbelow(True)
ax3.grid(axis='y')
ax3.set_xticks(range(N_TOP_FEATURES), rf_feature_rank[:N_TOP_FEATURES], rotation = 90)

fig.subplots_adjust(top=0.88)
fig.suptitle(name)
plt.tight_layout()
plt.savefig(path+name+'_feature_hist.pdf')

# SAVE RANKS

rac_rank_svm = []
rac_rank_rf = []
svm_rank_rac = []
svm_rank_rf = []
rf_rank_rac = []
rf_rank_svm = []

for rac_f, svm_f, rf_f in zip(rac_feature_rank, svm_feature_rank, rf_feature_rank):
    rac_rank_svm.append(rac_feature_rank.index(svm_f))
    rac_rank_rf.append(rac_feature_rank.index(rf_f))
    svm_rank_rac.append(svm_feature_rank.index(rac_f))
    svm_rank_rf.append(svm_feature_rank.index(rf_f))
    rf_rank_rac.append(rf_feature_rank.index(rac_f))
    rf_rank_svm.append(rf_feature_rank.index(svm_f))

ranks = {}

ranks['rac_feature_ranked'] = rac_feature_rank
ranks['rac_rank_svm'] = rac_rank_svm
ranks['rac_rank_rf'] = rac_rank_rf

ranks['svm_feature_ranked'] = svm_feature_rank
ranks['svm_rank_rac'] = svm_rank_rac
ranks['svm_rank_rf'] = svm_rank_rf

ranks['rf_feature_ranked'] = rf_feature_rank
ranks['rf_rank_rac'] = rf_rank_rac
ranks['rf_rank_svm'] = rf_rank_svm

df = pd.DataFrame(ranks)
df.to_csv(path+'rank.csv')

# VENN DIAGRAM

plt.figure()
A = set(rac_feature_rank[:N_TOP_FEATURES])
B = set(rf_feature_rank[:N_TOP_FEATURES])
C = set(svm_feature_rank[:N_TOP_FEATURES])
diagram = venn3([A, B, C], ("RAC features", "RF features", "SVM features"))

A=set(map(str,A))
B=set(map(str,B))
C=set(map(str,C))

if diagram.get_label_by_id('100'): diagram.get_label_by_id('100').set_text('\n'.join(A - B - C))
if diagram.get_label_by_id('110'): diagram.get_label_by_id('110').set_text('\n'.join(A & B - C))
if diagram.get_label_by_id('010'): diagram.get_label_by_id('010').set_text('\n'.join(B - C - A))
if diagram.get_label_by_id('101'): diagram.get_label_by_id('101').set_text('\n'.join(A & C - B))
if diagram.get_label_by_id('111'): diagram.get_label_by_id('111').set_text('\n'.join(A & B & C))
if diagram.get_label_by_id('011'): diagram.get_label_by_id('011').set_text('\n'.join(B & C - A))
if diagram.get_label_by_id('001'): diagram.get_label_by_id('001').set_text('\n'.join(C - B - A))

for text in diagram.subset_labels:
    if text:
        text.set_fontsize(6)

plt.tight_layout()
plt.title(name)
plt.savefig(path+name+'_venn.pdf')