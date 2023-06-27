import argparse
import json
import pickle as pkl
import pandas as pd
import numpy as np
from rac import RAClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# CMD LINE ARGS PARSING

parser = argparse.ArgumentParser(description='Feature importance')
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
    '--xgb-params',
    action='store',
    type=str,
    required=True,
    help="Path to the .json file with XGB parameters."
)
parser.add_argument(
    '--output-path',
    action='store',
    type=str,
    required=True,
    help="The path where to save results."
)

args = parser.parse_args()
dataset = args.dataset
rac_params_path = args.rac_params
svm_params_path = args.svm_params
rf_params_path = args.rf_params
xgb_params_path = args.xgb_params
output_path = args.output_path

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

with open(xgb_params_path) as xgb_params_fp:
    xgb_params = json.load(xgb_params_fp)

# FITTING CLASSIFIERS

rac = RAClassifier(**rac_params)
rac.fit(X, y)

svm = SVC(**svm_params)
svm.fit(X, y)

rf = RandomForestClassifier(**rf_params)
rf.fit(X, y)

xgb = XGBClassifier(**xgb_params)
xgb.fit(X, y)

# SAVE FEATURE RANKS

rac_idx = np.flip(np.argsort(rac.feature_importances_))
rac_feature_rank = [feature_ids[i] for i in rac_idx]

svm_idx = np.flip(np.argsort(svm.coef_[0]))
svm_feature_rank = [feature_ids[i] for i in svm_idx]

rf_idx = np.flip(np.argsort(rf.feature_importances_))
rf_feature_rank = [feature_ids[i] for i in rf_idx]

xgb_idx = np.flip(np.argsort(xgb.feature_importances_))
xgb_feature_rank = [feature_ids[i] for i in xgb_idx]

ranks = {}
ranks['rac_feature_ranked'] = rac_feature_rank
ranks['svm_feature_ranked'] = svm_feature_rank
ranks['rf_feature_ranked'] = rf_feature_rank
ranks['xgb_feature_ranked'] = xgb_feature_rank

df = pd.DataFrame(ranks)
df.to_csv(output_path+'feature_rank.csv')