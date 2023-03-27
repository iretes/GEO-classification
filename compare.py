import numpy as np
import pickle as pkl
import pandas as pd
import argparse
import math
import os
import json
from time import time
from enum import Enum
from statistics import mean
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from rac import RAClassifier
from nc import NearestCentroidProba
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import auc
from utils import Scorer
from utils import compute_scores, compute_roc_auc, grid_search_cv

# ------------------- Classifiers available for comparison ---------------------
class Clfs(Enum):
    RAC = 'RAC'
    NC = 'NC'
    KNN = 'KNN'
    SVM = 'SVC'
    GNB = 'GNB'
    RF = 'RF'

classifiers_available = [name for name in Clfs.__members__]

# ----------------------------- Scores evaluated -------------------------------
scores = {
    'accuracy': Scorer(accuracy_score),
    'f1 micro': Scorer(f1_score, average='micro'),
    'f1 macro': Scorer(f1_score, average='macro'),
    'f1 weighted': Scorer(f1_score, average='weighted')
}

best_scores_available = []
for score_name in scores:
    best_scores_available.append(score_name)
    best_scores_available.append('mean '+score_name)
    best_scores_available.append('concat '+score_name)
best_scores_available.append('aggregated rank')

# -------------------------- Cmd line argument parsing -------------------------
parser = argparse.ArgumentParser(description='Classifiers comparison')
parser.add_argument(
    '--nested-cv',
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to perform nested cross validation."
)
parser.add_argument(
    '--dev-dataset',
    action='store',
    type=str,
    required=True,
    help="Path to the development dataset."
)
parser.add_argument(
    '--test-dataset',
    action='store',
    type=str,
    default=None,
    help="Path to the test dataset."
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
    '--classifiers',
    nargs='+',
    type=str,
    default=[],
    help="Classifiers to compare, must be in %s. If not specified all the " + 
    "classifiers will be compared." % (classifiers_available)
)
parser.add_argument(
    '--best-score',
    action='store',
    type=str,
    default='concat f1 weighted',
    help="Scorer function to use to determine the best parameters, "+
    "must be in %s. If not specified all the scores will be evaluated." 
    % best_scores_available
)
parser.add_argument(
    '--n-splits',
    action='store',
    default=5,
    type=int,
    help="Number of folds to use; with --nested-cv refers to the number of " + 
    "internal folds; ignored with --loo (default: 5)."
)
parser.add_argument(
    '--ext-n-splits',
    action='store',
    default=5,
    type=int,
    help="Number of folds to use in the external cross validation; " +
    "ignored with --ext-loo (default: 5)."
)
parser.add_argument(
    '--loo',
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to perform leave one out cross validation; " +
    "with --nested-cv leave one out cross validation will be done in the " +
    "internal cross validation"
)
parser.add_argument(
    '--ext-loo',
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to perform leave one out in the external cross validation."
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
    '--verbose',
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to print verbose output."
)

args = parser.parse_args()
nested_cv = args.nested_cv
dev_dataset = args.dev_dataset
test_dataset = args.test_dataset
pos_lbl = args.pos_lbl
output_path = args.output_path
classifiers = args.classifiers
best_score = args.best_score
n_splits = args.n_splits
ext_n_splits = args.ext_n_splits
loo = args.loo
ext_loo = args.ext_loo
random_state = args.random_state
standardize = args.standardize
verbose = args.verbose

# ------------------------- Cmd line argument validation -----------------------

with open(dev_dataset, 'rb') as fp :
    dev_dict = pkl.load(fp)
X = dev_dict['X']
y = dev_dict['y']

if not nested_cv and not test_dataset:
    raise ValueError("Test set needed.")

if test_dataset:
    with open(test_dataset, 'rb') as fp :
        test_dict = pkl.load(fp)
    X_test = test_dict['X']
    y_test = test_dict['y']

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

if not os.path.isdir(output_path):
    raise ValueError(
        "Invalid resfoder ('" + output_path + "' is not an existing directory)."
    )
if not output_path.endswith('/'):
    raise ValueError("Invalid resfoder (must end with /.)")

# Dict to store classifiers classes
clfs = {}

if not classifiers:
    classifiers = classifiers_available

for clf in classifiers:
    if clf==Clfs.RAC.name:
        clfs[Clfs.RAC.name] = RAClassifier
    elif clf==Clfs.NC.name:
        clfs[Clfs.NC.name] = NearestCentroidProba
    elif clf==Clfs.KNN.name:
        clfs[Clfs.KNN.name] = KNeighborsClassifier
    elif clf==Clfs.SVM.name:
        clfs[Clfs.SVM.name] = SVC
    elif clf==Clfs.GNB.name:
        clfs[Clfs.GNB.name] = GaussianNB
    elif clf==Clfs.RF.name:
        clfs[Clfs.RF.name] = RandomForestClassifier
    else:
        raise ValueError(
            "Invalid classifier (must be in %s)." % classifiers_available
        )

if best_score not in best_scores_available:
    raise ValueError(
        "Invalid best score (must be in %s)." % best_scores_available
    )

if n_splits < 2:
    raise ValueError("Invalid nsplits (must be an int >= 2).")

if ext_n_splits < 2:
    raise ValueError("Invalid ext_nsplits (must be an int >= 2).")

multiclass = "ovr"

scaler = False
if standardize:
    scaler = StandardScaler()

# ---------------------------- Parameter grids ---------------------------------

n_features = X.shape[1]
third = math.floor(1/3*n_features)
RAC_grid = ParameterGrid([
    {
        'ra_method': ['borda', 'borda_median', 'borda_gmean', 'borda_l2'],
        'weighted': [True],
        'r_method': ['min', 'max', 'average'],
        'p': [1, 2, 3/4]
    },
    {
        'ra_method': ['borda', 'borda_median', 'borda_gmean', 'borda_l2'],
        'weighted': [False, (third, third)],
        'r_method': ['min', 'max', 'average'],
    },
    {
        'metric': ['kendall'],
        'ra_method': ['borda', 'borda_median', 'borda_gmean', 'borda_l2'],
        'r_method': ['min', 'max', 'average']
    }
])

NC_grid = ParameterGrid({
    'metric': ['euclidean', 'manhattan']
})

# Estimates max_neighbors according to min train size
n_classes = np.unique(y).shape[0]
ext_train_fract = (n_splits-1)/n_splits
if nested_cv:
    int_train_fract = (ext_n_splits-1)/ext_n_splits
else:
    int_train_fract = 1
max_neighbors = math.floor((int_train_fract*ext_train_fract*X.shape[0])/n_classes)
if max_neighbors == 0:
    max_neighbors = 1
if max_neighbors > 20:
    max_neighbors = 20

KNN_grid = ParameterGrid([
    {
        'weights': ['uniform', 'distance'],
        'n_neighbors': [*range(3, max_neighbors+1, 2)],
        'metric': ['euclidean', 'manhattan']
    },
    {
        'n_neighbors': [1],
        'metric': ['euclidean', 'manhattan']
    }
])

SVC_grid = ParameterGrid([
    {
        'C': [0.001, 0.1, 1, 100],
        'kernel': ['linear']
    },
    {
        'C': [0.001, 0.1, 1, 100],
        'kernel': ['poly'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto', 0.001, 0.1, 1, 100]
    },
    {
        'C': [0.001, 0.1, 1, 100],
        'kernel': ['rbf', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.001, 0.1, 1, 100]
    }
])

GNB_grid = ParameterGrid({
    'var_smoothing': [1e-1, 1e-9, 1e-3, 0.5]
})

RF_grid = ParameterGrid({
    'n_estimators': [50, 100, 500],
    'min_samples_split': [2, 0.1, 0.25, 0.33],
})

grids = {
    Clfs.RAC.name: RAC_grid,
    Clfs.NC.name: NC_grid,
    Clfs.KNN.name: KNN_grid,
    Clfs.SVM.name: SVC_grid,
    Clfs.GNB.name: GNB_grid,
    Clfs.RF.name: RF_grid
}

# ----------------------------  Nested cross validation ------------------------
if nested_cv:
    # Dict to store the concatenation of predictions, the values of the scores 
    # and the true targets from external cross validation (for each classifier)
    predictions = {}

    for clf_name in clfs:
        predictions[clf_name] = {}

    # Dict of scores to evaluate in the internal cross validation
    internal_scores = {}
    for score_name, scorer in scores.items():
        if score_name in best_score:
            internal_scores[score_name] = scorer
            break

    # Dict to store nested cross validation results
    results = {}

    if ext_loo: ext_n_splits = X.shape[0]
    for i in range(ext_n_splits):
        results['split%d params' % i] = []
    for i in range(ext_n_splits):
        results['split%d fit times' % i] = []
    for i in range(ext_n_splits):
        results['split%d score times' % i] = []
    for score in scores:
        for i in range(ext_n_splits):
            results['split%d %s' % (i, score)] = []
        results['mean '+score] = []
        results['concat '+score] = []
    if not loo:
        for i in range(n_splits):
            results['split%d auroc' % i] = []
        results['mean auroc'] = []
    results['concat auroc'] = []
    results['concat sklearn auroc'] = []

    # Iterate on classifiers
    for clf_name, base_clf in clfs.items():
        if verbose: print(clf_name+" classifier")

        # External splitting strategy to use
        if ext_loo: 
            outer_cv = LeaveOneOut()
        else: 
            outer_cv = StratifiedKFold(
                n_splits=ext_n_splits,
                shuffle=True,
                random_state=random_state)

        # Concatenated predictions, true targets and scores from external test folds
        y_pred_concat = np.array([])
        y_true_concat = np.array([])
        y_score_concat = None

        # Dict to store scores from external test folds
        external_scores = {}
        for score_name in scores:
            external_scores[score_name] = []
        if not loo:
            external_scores['auroc'] = []

        # External cross validation
        for split, (train_index , test_index) in enumerate(outer_cv.split(X, y)):

            if verbose: print('[EXTERNAL CV %d/%d]' % (split+1, n_splits))

            X_train, X_val = X[train_index, :], X[test_index, :]
            y_train, y_val = y[train_index] , y[test_index]

            # Internal splitting strategy to use
            if loo: 
                inner_cv = LeaveOneOut()
                n_splits = X_train.shape[0]
            else: 
                inner_cv = StratifiedKFold(n_splits=n_splits)

            # Internal cross validation for each parameter
            internal_results, best_param = grid_search_cv(
                base_clf=base_clf, 
                param_grid=grids[clf_name],
                scores=internal_scores,
                X=X_train,
                y=y_train,
                cv=inner_cv,
                scaler=scaler,
                best_score=best_score,
                probability=False,
                verbose=verbose
            )

            if verbose: print("Best params %s" % best_param)

            # Save internal results on csv file
            df = pd.DataFrame.from_dict(internal_results)
            df.to_csv('%ssplit%d_%s_cv.csv' % (output_path, split, clf_name))

            if base_clf==SVC:
                best_param['probability'] = True

            # Data scaling
            if standardize:
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
        
            # Refit the best model with the best params
            fit_start = time()
            clf = base_clf(**best_param).fit(X_train, y_train)
            fit_time = time()-fit_start

            results['split%d params' % split].append(best_param)
            results['split%d fit times' % split].append(fit_time)

            # Test the model on the external validation set
            score_start = time()
            y_pred = clf.predict(X_val)
            results['split%d score times' % split].append(time()-score_start)

            # Get probabilities associated to predictions
            y_score = clf.predict_proba(X_val)

            # Compute and store AUROC score on this external test fold
            if not ext_loo:
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
                external_scores['auroc'].append(auroc_s)

            # Concatenate predictions made on this external fold
            y_pred_concat = np.concatenate([y_pred_concat, y_pred])
            y_true_concat = np.concatenate([y_true_concat, y_val])
            y_score_concat = np.concatenate([y_score_concat, y_score]) \
                if y_score_concat is not None else y_score

            # Compute and store scores on this external fold
            test_scores = compute_scores(
                scores,
                y_pred=y_pred,
                y_true=y_val,
                y_score=y_score
            )
            for score_name, value in test_scores.items():
                results['split%d %s' % (split, score_name)].append(value)
                external_scores[score_name].append(value)
                if verbose: print('%s=%f ' % (score_name, value), end='')
            if verbose: print()

        # Compute and store mean scores on external folds
        for score_name, value in external_scores.items():
            results['mean '+score_name].append(mean(value))

        # Compute and store scores on the concatenation of the predictions on external folds
        concat_scores = compute_scores(
            scores,
            y_pred=y_pred_concat,
            y_true=y_true_concat,
            y_score=y_score_concat
        )
        for score_name, value in concat_scores.items():
            results['concat '+score_name].append(value)

        # Save external predictions, true targets and scores
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
    aggr_rank = np.zeros((len(clfs)))
    if not loo:
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
    df = pd.DataFrame.from_dict(results, orient='index', columns=clfs.keys())
    df.to_csv(output_path+'nestedcv.csv')

    # Save predictions, true targets and scores from external test folds
    with open(output_path + "predictions.pkl", "wb") as fp:
        pkl.dump(predictions, fp)
    # --------------------------------------------------------------------------
else: # ---------- Cross validation on development set & hold out test ---------
    X_train = X
    y_train = y

    # Dict to store predictions, scores and true targets on test data
    predictions = {}
    for clf_name in clfs:
        predictions[clf_name] = {}

    # Dictionary to store results
    results = {}
    results['params'] = []
    results['fit time'] = []
    results['score time'] = []
    for score in scores:
        results['%s'%score] = []
    results['auroc'] = []

    # Iterate on classifiers
    for clf_name, base_clf in clfs.items():
        if verbose: print(clf_name+" classifier")

        # Splitting strategy to use
        if loo: 
            cv = LeaveOneOut()
        else: 
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state)

        # Grid search with cross validation 
        cv_results, best_param = grid_search_cv(
            base_clf=base_clf, 
            param_grid=grids[clf_name],
            scores=scores,
            X=X_train,
            y=y_train,
            cv=cv,
            scaler=scaler,
            best_score=best_score,
            probability=True,
            verbose=verbose
        )

        if verbose: print("Best params %s" % best_param)

        # Save best params on json file
        with open(output_path + clf_name + "_params.json", "w") as outfile:
            json.dump(best_param, outfile)

        # Save cv results on csv file
        df = pd.DataFrame.from_dict(cv_results)
        df.to_csv('%s%s_cv.csv' % (output_path, clf_name))

        if base_clf==SVC:
            best_param['probability'] = True

        # Data scaling
        if standardize:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Refit the best model with the best params
        fit_start = time()
        clf = base_clf(**best_param).fit(X_train, y_train)
        fit_time = time()-fit_start

        results['params'].append(best_param)
        results['fit time'].append(fit_time)

        # Test the model on the test set
        score_start = time()
        y_pred = clf.predict(X_test)
        results['score time'].append(time()-score_start)

        # Get probabilities associated to predictions
        y_score = clf.predict_proba(X_test)

        # Compute and store AUROC score on test set
        if not loo:
            if binary:
                auroc_s = roc_auc_score(
                    y_test,
                    y_score[:, 1]
                )
            else:
                auroc_s = roc_auc_score(
                    y_test,
                    y_score,
                    labels=clf.classes_,
                    average="weighted",
                    multi_class=multiclass
                )
            results['auroc'].append(auroc_s)

        # Compute and store scores on test set
        test_scores = compute_scores(
            scores,
            y_pred=y_pred,
            y_true=y_test,
            y_score=y_score
        )
        for score_name, value in test_scores.items():
            results['%s' % (score_name)].append(value)
            if verbose: print('%s=%f ' % (score_name, value), end='')
        if verbose: print()

        predictions[clf_name]['preds'] = y_pred
        predictions[clf_name]['true'] = y_test
        predictions[clf_name]['probs'] = y_score

    # Save test results on csv file
    df = pd.DataFrame.from_dict(results, orient='index', columns=clfs.keys())
    df.to_csv(output_path+'test_scores.csv')

    # Save predictions, true targets and scores
    with open(output_path + "predictions.pkl", "wb") as fp:
        pkl.dump(predictions, fp)