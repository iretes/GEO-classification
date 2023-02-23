import numpy as np
from time import time
from statistics import mean
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

class Scorer():
    def __init__(self, score_func, greater_better=True, prob=False, **kwargs):
        self.score_func_ = score_func
        self.greater_better = greater_better
        self.prob = prob
        self.kwargs_ = kwargs

    def score(self, y_true, y_pred):
        return self.score_func_(y_true, y_pred, **self.kwargs_)

def compute_scores(scores, y_true, y_pred, y_score=None):
    scores_val = {}
    for score_name, scorer in scores.items():
        if scorer.prob:
            scores_val[score_name] = scorer.score(y_true, y_score)
        else:
            scores_val[score_name] = scorer.score(y_true, y_pred)
    return scores_val

def grid_search_cv(
    base_clf,
    param_grid,
    scores,
    X,
    y,
    cv,
    scaler=False,
    best_score='mean accuracy',
    probability=False,
    verbose=False
):

    n_splits = cv.get_n_splits(X)
    n_params = len(list(param_grid))

    if verbose: 
        print('Fitting %d folds for each of %d candidates, totalling %d fits'
        % (n_splits, n_params, n_splits*n_params))

    results = {}
    results['params'] = []
    results['mean fit time'] = []
    results['mean score time'] = []
    for score_name in scores:
        for i in range(n_splits):
            results['split%d %s' % (i, score_name)] = []
        results['mean '+score_name] = []
        results['concat '+score_name] = []
    for param_comb in param_grid:
        for param in param_comb:
            results['param_'+param] = [None for _ in range(n_params)]
    
    # Iterate on parameters
    for p, params in enumerate(param_grid):

        y_pred_concat = np.array([])
        y_true_concat = np.array([])
        y_score_concat = None
        fit_times = []
        score_times = []
        folds_scores = {}
        for score_name in scores:
            folds_scores[score_name] = []

        for param_name, param_val in params.items():
            results['param_'+param_name][p] = param_val
        results['params'].append(params)
        
        # Cross validation
        for split, (train_index , test_index) in enumerate(cv.split(X, y)):
            if verbose:
                print('[CV %d/%d; %d/%d] %s '
                % (split+1, n_splits, p+1, n_params, params), end='')

            X_train, X_val = X[train_index, :], X[test_index, :]
            y_train, y_val = y[train_index] , y[test_index]

            # Data scaling
            if scaler:
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            if probability and base_clf==SVC:
                params['probability'] = True

            # Fit the model on training set
            fit_start = time()
            clf = base_clf(**params).fit(X_train, y_train)
            fit_times.append(time()-fit_start)

            # Test the model on validation set
            score_start = time()
            y_pred = clf.predict(X_val)
            score_times.append(time()-score_start)

            # Compute the probability associated to predictions
            y_score = None
            if probability:
                y_score = clf.predict_proba(X_val)

            # Concatenate predictions on this test fold
            y_pred_concat = np.concatenate([y_pred_concat, y_pred])
            y_true_concat = np.concatenate([y_true_concat, y_val])
            if probability:
                y_score_concat = np.concatenate([y_score_concat, y_score]) \
                    if y_score_concat is not None else y_score

            # Compute and store scores on this test fold
            curr_fold_scores = compute_scores(
                scores,
                y_pred=y_pred,
                y_true=y_val,
                y_score=y_score
            )
            
            for score_name, value in curr_fold_scores.items():
                results['split%d %s' % (split, score_name)].append(value)
                folds_scores[score_name].append(value)
                if verbose: print('%s=%f ' % (score_name, value), end='')
            if verbose: print()

        # Compute and store mean scores over folds
        for score_name, value in folds_scores.items():
            results['mean '+score_name].append(mean(value))
        results['mean fit time'].append(mean(fit_times))
        results['mean score time'].append(mean(score_times))

        # Compute and store scores on the concatenation of the predictions for each test fold
        concat_scores = compute_scores(
            scores,
            y_pred=y_pred_concat,
            y_true=y_true_concat,
            y_score=y_score_concat
        )
        
        for score_name, value in concat_scores.items():
            results['concat '+score_name].append(value)

    # Rank models
    aggr_rank = np.zeros((n_params))
    for score_name, scorer in scores.items():
        mean_score = np.array(results['mean '+score_name])
        if scorer.greater_better: 
            mean_score = 1-mean_score
        rank_mean = rankdata(mean_score, method='min')
        aggr_rank += rank_mean
        results['rank mean '+score_name] = rank_mean.tolist()
        concat_score = np.array(results['concat '+score_name])
        if scorer.greater_better: 
            concat_score = 1-concat_score
        rank_concat = rankdata(concat_score, method='min')
        aggr_rank += rank_concat
        results['rank concat '+score_name] = rank_concat.tolist()
    results['aggregated rank'] = rankdata(aggr_rank, method="min").tolist()

    # Find the index of the best params
    if best_score=='aggregated rank':
        idx_best_params = results[best_score].index(1)
    else:
        # If more than one parameter combination has rank 1
        # select the one with lower index whose aggregated rank is 1
        indeces_best_params = [
            i for i, x in enumerate(results['rank '+best_score]) if x == 1
        ]
        bests_ranks = np.array(results['aggregated rank'])[indeces_best_params]
        idx_best_params = indeces_best_params[np.argmin(bests_ranks)]
    
    best_params = results['params'][idx_best_params]
    
    return results, best_params

def compute_roc_auc(y_true, y_score, weighted, labels):
    n_classes = len(labels) 

    if n_classes <= 2:
        roc_auc = roc_auc_score(y_true, y_score[:, 1])
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1], pos_label=labels[1])
        return fpr, tpr, roc_auc
    
    y_true = label_binarize(y_true, classes=labels)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute weights
    w = np.ones((n_classes))
    tot = n_classes
    if weighted:
        tot = y_true.shape[0]
        for i in range(n_classes):
            w[i] = np.count_nonzero(y_true[:, i] == 1)

    # Compute ROC curve points and AUC for each class and average them
    roc_auc_mean = 0
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        roc_auc_mean += roc_auc[i] * w[i]
    roc_auc_mean /= tot

    # Concatenate unique fpr of all classes
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Compute mean tpr with interpolation
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i]) * w[i]
    mean_tpr /= tot

    return all_fpr, mean_tpr, roc_auc_mean