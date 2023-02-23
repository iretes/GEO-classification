"""
Implementation of Nearest Centroid Classifier that returns probability estimates
"""
import numpy as np
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted

class NearestCentroidProba(NearestCentroid):
    
    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.

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
        p = np.empty((X.shape[0], len(self.classes_)))
        if len(self.classes_) == 1:
            p[:] = 1
            return p
        d = self.distances_to_centroids(X)
        for i in range(X.shape[0]):
            p[i] = self.convert_distances_to_probas(d[i])
        return p

    def distances_to_centroids(self, X):
        """Computes the distances between each test sample and class centroids.

            Parameters
            ----------
            X : array-like of shape (n_samples, n_features)
                The test samples.
            Returns
            -------
            d : array-like of shape (n_samples, n_classes)
                The test samples' distances to classes,
                where classes are ordered as they are in `self.classes_`.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        X = self._validate_data(X, accept_sparse="csr", reset=False)

        return pairwise_distances(X, self.centroids_, metric=self.metric)

    def convert_distances_to_probas(self, d):
        """Converts distances into probabilities estimates.
            Lower distances correspond to higher probabilities.

            Parameters
            ----------
            d : array-like, shape (n_classes)
                The distances between a sample and the classes.
            Returns
            -------
            p : array-like of shape (n_classes)
                The probabilities that the sample belongs to the classes.
        """
        p = np.empty((d.shape[0]))
        sumd = np.sum(d)
        if sumd == 0:
            p.fill(1/d.shape[0])
            return p
        for i in range(d.shape[0]):
            p[i] = (1-(d[i]/sumd))/(d.shape[0]-1)
        return p