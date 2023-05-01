import numpy as np
import pickle as pkl
import pandas as pd
import json
import os
import argparse
from time import time
from enum import Enum
from statistics import mean
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_selection import RFE
from rac import RAClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import auc
from utils import Scorer
from utils import compute_scores, compute_roc_auc

# ------------------- Classifiers available for comparison ---------------------
class Clfs(Enum):
    RAC = 'RAC'
    SVM = 'SVM'
    RF = 'RF'
    XGB = 'XGB'

# ----------------------------- Scores evaluated -------------------------------
scores = {
    'accuracy': Scorer(accuracy_score),
    'f1 micro': Scorer(f1_score, average='micro'),
    'f1 macro': Scorer(f1_score, average='macro'),
    'f1 weighted': Scorer(f1_score, average='weighted')
}

# -------------------------- Cmd line argument parsing -------------------------
parser = argparse.ArgumentParser(description='Recursive Feature Elimination')
parser.add_argument(
    '--dataset',
    action='store',
    type=str,
    required=True,
    help="Path to the dataset."
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
    help="Path to the .json file with SVM parameters (kernel must be 'linear')."
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
    '--pos-lbl',
    action='store',
    type=str,
    default=None,
    help="Label of the 'positive' class in binary classification."
)
parser.add_argument(
    '--output-path',
    action='store',
    type=str,
    required=True,
    help="Path where to save results."
)
parser.add_argument(
    '--n-splits',
    action='store',
    default=5,
    type=int,
    help="Number of folds to use."
)
parser.add_argument(
    '--random-state',
    action='store',
    type=int,
    default=0,
    help="Seed to get reproducible results."
)
parser.add_argument(
    '--standardize',
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to standardize features."
)
parser.add_argument(
    '--n-features-to-select',
    action='store',
    type=int,
    default=20,
    help="RFE parameter."
)
parser.add_argument(
    '--step',
    action='store',
    type=float,
    default=0.5,
    help="RFE parameter."
)
parser.add_argument(
    '--verbose',
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to print verbose output."
)

args = parser.parse_args()
dataset = args.dataset
rac_params_path = args.rac_params
svm_params_path = args.svm_params
rf_params_path = args.rf_params
xgb_params_path = args.xgb_params
pos_lbl = args.pos_lbl
output_path = args.output_path
n_splits = args.n_splits
random_state = args.random_state
standardize = args.standardize
n_features_to_select = args.n_features_to_select
step = args.step
verbose = args.verbose

# ------------------------- Cmd line argument validation -----------------------

with open(dataset, 'rb') as fp :
    dataset_dict = pkl.load(fp)
X = dataset_dict['X']
y = dataset_dict['y']
features_names = np.array(dataset_dict['features_ids'])

labels = unique_labels(y)
binary = len(labels) <= 2

if binary and pos_lbl and labels[1] != pos_lbl:
    if labels[0] != pos_lbl:
        raise ValueError("Invalid poslbl (allowed values are " + 
            str(labels) + ")")
    else:
        # swap labels
        tmp = labels[0]
        labels[0] = labels[1]
        labels[1] = tmp

with open(rac_params_path) as rac_params_fp:
    rac_params = json.load(rac_params_fp)

with open(svm_params_path) as svm_params_fp:
    svm_params = json.load(svm_params_fp)

with open(rf_params_path) as rf_params_fp:
    rf_params = json.load(rf_params_fp)

with open(xgb_params_path) as xgb_params_fp:
    xgb_params = json.load(xgb_params_fp)

# Dict to store classifiers parameters
clfs_params = {}
clfs_params['RAC']=[]
clfs_params['SVM']=[]
clfs_params['RF']=[]
clfs_params['XGB']=[]
for i in range(n_splits):
    clfs_params['RAC'].append(rac_params)
    clfs_params['SVM'].append(svm_params)
    clfs_params['RF'].append(rf_params)
    clfs_params['XGB'].append(xgb_params)
print(clfs_params['RAC'])
print(clfs_params['RF'])
print(clfs_params['SVM'])
print(clfs_params['XGB'])

if not os.path.isdir(output_path):
    raise ValueError(
        "Invalid resfoder ('" + output_path + "' is not an existing directory)."
    )
if not output_path.endswith('/'):
    raise ValueError("Invalid resfoder (must end with /.)")

# Dict to store classifiers classes
clfs_classes = {}
clfs_classes[Clfs.RAC.name] = RAClassifier
clfs_classes[Clfs.SVM.name] = SVC
clfs_classes[Clfs.RF.name] = RandomForestClassifier
clfs_classes[Clfs.XGB.name] = XGBClassifier

if n_splits < 2:
    raise ValueError("Invalid nsplits (must be an int >= 2).")

multiclass = "ovr"

scaler = False
if standardize:
    scaler = StandardScaler()

# -------------------------------  Cross validation ----------------------------
# Dict to store the concatenation of predictions, the values of the scores 
# and the true targets from cross validation
predictions = {}
for clf_name in clfs_classes:
    predictions[clf_name] = {}

# Dict to store cross validation results
results = {}
for i in range(n_splits):
    results['split%d params' % i] = []
for i in range(n_splits):
    results['split%d fit times' % i] = []
for i in range(n_splits):
    results['split%d score times' % i] = []
for score in scores:
    for i in range(n_splits):
        results['split%d %s' % (i, score)] = []
    results['mean '+score] = []
    results['concat '+score] = []
for i in range(n_splits):
    results['split%d auroc' % i] = []
results['mean auroc'] = []
results['concat auroc'] = []
results['concat sklearn auroc'] = []

# Iterate on classifiers
for clf_name, base_clf in clfs_classes.items():
    if verbose: print(clf_name+" classifier")

    # Splitting strategy to use
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state)

    # Concatenated predictions, true targets and scores from test folds
    y_pred_concat = np.array([])
    y_true_concat = np.array([])
    y_score_concat = None

    # Dict to store scores from test folds
    cv_scores = {}
    for score_name in scores:
        cv_scores[score_name] = []
    cv_scores['auroc'] = []

    # Dict to store selected feature with this classifier
    selected_features = {}

    # Cross validation
    for split, (train_index , test_index) in enumerate(cv.split(X, y)):

        if verbose: print('[CV %d/%d]' % (split+1, n_splits))

        X_train, X_val = X[train_index, :], X[test_index, :]
        y_train, y_val = y[train_index] , y[test_index]

        # Data scaling
        if standardize:
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
    
        # Fit
        clf_params = clfs_params[clf_name][split]
        if base_clf==SVC:
            print(clf_params)
            clf_params['probability'] = True
        selector = RFE(
            base_clf(**clf_params),
            n_features_to_select=n_features_to_select,
            step=step
        )
        fit_start = time()
        selector = selector.fit(X_train, y_train)
        fit_time = time()-fit_start

        # Store selected features
        for f in features_names[selector.support_]:
            if f not in selected_features:
                selected_features[f] = [False for i in range(n_splits)]
            selected_features[f][split] = True

        results['split%d params' % split].append(clf_params)
        results['split%d fit times' % split].append(fit_time)

        # Test the model on the validation set
        score_start = time()
        y_pred = selector.predict(X_val)
        results['split%d score times' % split].append(time()-score_start)

        # Get probabilities associated to predictions
        y_score = selector.predict_proba(X_val)

        # Compute and store AUROC score on this test fold
        if binary:
            auroc_s = roc_auc_score(
                y_val,
                y_score[:, 1]
            )
        else:
            auroc_s = roc_auc_score(
                y_val,
                y_score,
                labels=clf.classes_,
                average="weighted",
                multi_class=multiclass
            )
        results['split%d auroc' % split].append(auroc_s)
        cv_scores['auroc'].append(auroc_s)

        # Concatenate predictions made on this fold
        y_pred_concat = np.concatenate([y_pred_concat, y_pred])
        y_true_concat = np.concatenate([y_true_concat, y_val])
        y_score_concat = np.concatenate([y_score_concat, y_score]) \
            if y_score_concat is not None else y_score

        # Compute and store scores on this fold
        test_scores = compute_scores(
            scores,
            y_pred=y_pred,
            y_true=y_val,
            y_score=y_score
        )
        for score_name, value in test_scores.items():
            results['split%d %s' % (split, score_name)].append(value)
            cv_scores[score_name].append(value)
            if verbose: print('%s=%f ' % (score_name, value), end='')
        if verbose: print()

    # Save selected features into a csv file
    feature_count = []
    for f, splits in selected_features.items():
        feature_count.append(splits.count(True))
    df = pd.DataFrame.from_dict(
        selected_features,
        orient='index',
        columns=['split%d'%i for i in range(n_splits)]
    )
    df['count']= feature_count
    df.to_csv(output_path+'%s_features.csv'%clf_name)

    # Compute and store mean scores on test folds
    for score_name, value in cv_scores.items():
        results['mean '+score_name].append(mean(value))

    # Compute and store scores on the concatenation of the predictions on test folds
    concat_scores = compute_scores(
        scores,
        y_pred=y_pred_concat,
        y_true=y_true_concat,
        y_score=y_score_concat
    )
    for score_name, value in concat_scores.items():
        results['concat '+score_name].append(value)

    # Save predictions, true targets and scores
    predictions[clf_name]['preds'] = y_pred_concat
    predictions[clf_name]['true'] = y_true_concat
    predictions[clf_name]['probs'] = y_score_concat

    # Compute auroc
    fpr, tpr, auroc = compute_roc_auc(
        y_true=y_true_concat,
        y_score=y_score_concat,
        weighted=True,
        labels=labels
    )
    results['concat auroc'].append(auroc)
    results['concat sklearn auroc'].append(auc(fpr, tpr))

# Rank models
aggr_rank = np.zeros((len(clfs_classes)))
scores['auroc'] = Scorer(None)
for score_name, scorer in scores.items():
    mean_score = np.array(results['mean '+score_name])
    if scorer.greater_better: mean_score = 1-mean_score
    rank_mean = rankdata(mean_score, method='min').tolist()
    results['rank mean '+score_name] = rank_mean
    aggr_rank += np.array(rank_mean)
    concat_score = np.array(results['concat '+score_name])
    if scorer.greater_better: concat_score = 1-concat_score
    rank_concat = rankdata(concat_score, method='min').tolist()
    results['rank concat '+score_name] = rank_concat
    aggr_rank += np.array(rank_concat)
results['aggregated rank'] = rankdata(aggr_rank, method='min').tolist()

# Save test results on csv file
df = pd.DataFrame.from_dict(results, orient='index', columns=clfs_classes.keys())
df.to_csv(output_path+'rfe_cv.csv')

# Save predictions, true targets and scores from test folds
with open(output_path + "rfe_predictions.pkl", "wb") as fp:
    pkl.dump(predictions, fp)