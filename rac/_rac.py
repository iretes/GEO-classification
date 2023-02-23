"""
Implementation of the Rank Aggregation Classifier
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import rankdata, gmean, kendalltau
from numpy.linalg import norm

class RAClassifier(BaseEstimator, ClassifierMixin):
    """RA Classifier class.
    Parameters
    ----------
    r_method: {'min', 'max', 'average'}, default='min'
        Parameter that specifies the method used to assign ranks to tied elements.
    ra_method : {'borda', 'borda_median', 'borda_gmean', 'borda_l2'}, default='borda'
        Parameter that specifies the method used to aggregate the rankings.
    metric : {'spearman', 'kendall'}, default='spearman'
        Parameter that specifies the metric used to compute the distance to the signatures.
    weighted : Boolean or int, default=False
        Parameter that specifies whether the distance to the signatures is weighted by the rank or not.
        Ignored if the parameter metric is 'kendall'.
        If True weights depends on to the rank: the weights for the candidates at top and bottom of the ranking
        are higher and decrease towards the center of the ranking.
        If tuple it must have 2 int values (n1, n2). In that case only the n1 features at the top and 
        n2 features at the bottom of the ranking will be taken into account.
    p : float
        Parameter that specifies the exponent to use in the weighting function when weighted is True and metric is 'kendall'.
    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    n_features_in_ : int
        The number of features seen during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    class_signatures_ : ndarray, shape (n_classes, n_features)
        The signatures for each class, computed during :meth:`fit`.
    class_centroids_ : ndarray, shape (n_classes, n_features)
        The centroids for each class, computed during :meth:`fit`.
    weights_ : ndarray, shape (n_classes, n_features)
        The weights associated with each class signature, computed during :meth:`fit`.
    """
    
    def __init__(
        self,
        r_method='min',
        ra_method='borda',
        metric='spearman',
        weighted=False,
        p=1):
        self.r_method = r_method
        self.ra_method = ra_method
        self.metric = metric
        self.weighted = weighted
        self.p = p

    def fit(self, X, y):
        """Computes class signatures and centroids.
        To compute class signatures the features are ranked for each training sample
        and a Borda based rank aggregation method is applied to all samples from each class.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the features seen during fit
        self.n_features_in_ = X.shape[1]
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        # Store samples and targets
        self.X_ = X
        self.y_ = y

        # Check rank method, rank aggregation method, metric and the exponent of the weighting function
        if self.r_method not in ['min', 'max', 'average']:
            raise ValueError("Unrecognized r_method.")
        if self.ra_method not in ['borda', 'borda_median', 'borda_gmean', 'borda_l2']:
            raise ValueError("Unrecognized ra_method.")
        if self.metric not in ['spearman', 'kendall']:
            raise ValueError("Unrecognized metric.")
        if not isinstance(self.p, (int, float)):
            raise TypeError("Invalid p (must be int or float).")
        if self.p <= 0:
            raise ValueError("Invalid p (must be greater than 0).")

        # Check weighted
        if self.metric in ['kendall']:
            self.weighted = False
        else:
            if isinstance(self.weighted, tuple):
                if (not isinstance(self.weighted[0], int) or not isinstance(self.weighted[1], int)):
                    raise TypeError("Invalid weighted (values must be int).")
                if (self.weighted[0] < 0 or self.weighted[1] < 0):
                    raise ValueError("Invalid weighted (values must be >= 0).")
                if (self.weighted[0]+self.weighted[1] >= self.n_features_in_):
                    raise ValueError("Invalid weighted (the sum of the values must be less than the number of features).")
                if (self.weighted[0]+self.weighted[1] == 0):
                    raise ValueError("Invalid weighted (cannot set all weights to 0).")
            elif not isinstance(self.weighted, bool):
                raise TypeError("Invalid weighted (must be bool or tuple).")

        # Compute signature, centroid and weights for each class
        self.class_signatures_ = np.empty((len(self.classes_), self.n_features_in_))
        self.class_centroids_ = np.empty((len(self.classes_), self.n_features_in_))
        self.weights_ = np.ones((len(self.classes_), self.n_features_in_))
        for i in range(len(self.classes_)):
            self.class_signatures_[i] = self.aggregate(self.X_[self.y_ == self.classes_[i]])
            self.class_centroids_[i]= self.compute_centroid(self.X_[self.y_ == self.classes_[i]])
            if self.weighted: 
                self.weights_[i] = self.compute_weights(self.class_signatures_[i])

        # Return the classifier
        return self

    def predict(self, X):
        """Predicts the class for each test sample. 
        The prediction is based on the distance between the sample and the class signatures 
        (or the euclidean distance to the class centroids if nearest class_sigatures are equidistant). 
        The distance between rankings is computed using the metric specified, 
        weighted by the rank according to the weighting mode required.
        Parameters.
        ----------
        X : array-like, shape (n_samples, n_features)
            The test samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted label for each sample.
        """
        # Check if fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X, ensure_min_features=self.n_features_in_)

        # Predict label of samples according to the distances to signatures
        d = self.distances_to_signatures(X)
        closest = np.empty((X.shape[0]), dtype=int)
        for i in range(X.shape[0]):
            closers = np.where(d[i]==d[i].min())[0]
            closest[i] = closers[0]
            if closers.size > 1: # Determine nearest centroid (using euclidean distance)
                closest[i] = closers[
                    np.argmin(np.linalg.norm(X[i]-self.class_centroids_[closers], axis=1))
                ]
        return self.classes_[closest]

    def predict_proba(self, X):
        """Return probability estimates for the test data X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test samples.
        Returns
        -------
        p : array-like of shape (n_samples, n_classes)
            The class probabilities of the test samples,
            where classes are ordered as they are in `self.classes_`.
        """
        # Check if fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X, ensure_min_features=self.n_features_in_)

        p = np.ones((X.shape[0], len(self.classes_)))
        if len(self.classes_) == 1:
            return p
        
        d = self.distances_to_signatures(X)

        # Convert distances to probabilities
        for i in range(X.shape[0]):
            p[i] = self.convert_distances_to_probas(d[i])
            closers = np.where(d[i]==d[i].min())[0]
            if closers.size > 1: # Compute euclidean distances to centroids
                pl = 0
                p2l = 0
                for j in range(p.shape[1]):
                    if p[i][j] > pl:
                        p2l = pl
                        pl = p[i][j]
                leftp = closers.size*(pl-p2l)
                dc = np.linalg.norm(X[i]-self.class_centroids_[closers], axis=1)
                pc = self.convert_distances_to_probas(dc)
                for j in range(closers.size): # Adjust probabilities
                    p[i][closers[j]] = p2l + pc[j]*leftp
        
        return p

    def distances_to_signatures(self, X):
        """Computes the matrix of pairwise distances between the ranking of features in the two sample sets.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples to compare to signatures.
        Returns
        -------
        d : array-like, shape (n_samples, n_classes)
            The distance matrix between all samples from X to all class signatures.
        """
        d = np.zeros((X.shape[0], len(self.classes_)))
        if X.shape[1] == 1:
            return d 

        # Compute distances to signatures according to the metric
        for i in range(X.shape[0]):
            ranked_sample = rankdata(X[i], method=self.r_method)
            for j in range(len(self.classes_)):
                if self.metric == 'spearman':
                    d[i][j] = np.sum(np.multiply(np.abs(ranked_sample-self.class_signatures_[j]), self.weights_[j]))
                elif self.metric == 'kendall':
                    d[i][j] = 1-kendalltau(ranked_sample, self.class_signatures_[j])[0]

        return d

    def aggregate(self, X):
        """Ranks features and aggregates them into one signature.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples from one class only.
        Returns
        -------
        a : ndarray, shape (n_features,)
            The signature determined by the input samples. 
            Smallest rank corresponds to smallest feature value.
        """
        # Rank features
        r = rankdata(X, self.r_method, axis=1)

        # Aggregate ranks
        a = np.empty(X.shape[1])
        if self.ra_method == 'borda':
            a = rankdata(np.sum(r, axis=0), method=self.r_method)
        elif self.ra_method == 'borda_median':
            a = rankdata(np.median(r, axis=0), method=self.r_method)
        elif self.ra_method == 'borda_gmean':
            a = rankdata(gmean(r, axis=0), method=self.r_method)
        else: # self.ra_method == 'borda_l2'
            a = rankdata(norm(r, 2, axis=0), method=self.r_method)
        return a

    def compute_centroid(self, X):
        """Computes the centroid of the class.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples from one class only.
        Returns
        -------
        m : ndarray, shape (n_features,)
            The centroid of the class.
        """
        return X.mean(axis=0)

    def compute_weights(self, signature):
        """Computes the weights associated with a class signature.
        Parameters
        ----------
        signature : array-like of shape (n_features)
            The class signature.
        Returns
        -------
        w : array-like of shape (n_features)
            The weights associated with the class signature,
            computed according to the weighting mode required.
        """
        max_rank = np.max(signature)
        if isinstance(self.weighted, tuple):
            return ((signature <= self.weighted[0]) | \
                    (signature > max_rank-self.weighted[1])).astype(int)
        else:
            return np.power(np.abs(max_rank+1-2*signature), self.p)

    def convert_distances_to_probas(self, d):
        """Converts distances into probabilities estimates.
           Lower distances correspond to higher probabilities.
           Parameters
           ----------
           d : array-like, shape (n_classes)
               The distances between a sample and some classes.
           Returns
           -------
           p : array-like of shape (n_classes)
               The probabilities that the sample belongs to each class.
        """
        p = np.empty((d.shape[0]))
        sumd = np.sum(d)
        if sumd == 0:
            p.fill(1/d.shape[0])
            return p
        for i in range(d.shape[0]):
            p[i] = (1-(d[i]/sumd))/(d.shape[0]-1)
        return p

    def get_params(self, deep=True):
        return {
            "r_method": self.r_method,
            "ra_method": self.ra_method,
            "metric": self.metric,
            "weighted": self.weighted,
            "p": self.p
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self