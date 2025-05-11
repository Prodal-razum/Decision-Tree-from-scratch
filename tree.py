import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import _num_samples


def entropy(y):
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    probs = y.mean(axis=0)
    output = -np.sum(probs * np.log2(probs + EPS))  # YOUR CODE HERE
    return output


def gini(y):
    """
    Computes the Gini impurity of the provided distribution

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    probs = y.mean(axis=0)
    output = 1 - (probs ** 2).sum()  # YOUR CODE HERE
    return output


def variance(y):
    """
    Computes the variance the provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Variance of the provided target vector
    """

    # YOUR CODE HERE
    return y.var()


def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    mad_med = np.abs(y - y.median()).mean()
    return mad_med


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """

    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None


class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True),  # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2,
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(
            self.all_criterions.keys())

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None  # Use the Node class to initialize it later
        self.debug = debug

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        left_indeces = X_subset[:, feature_index] < threshold

        X_left, X_right = X_subset[left_indeces], X_subset[~left_indeces]
        y_left, y_right = y_subset[left_indeces], y_subset[~left_indeces]

        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE

        left_indeces = X_subset[:, feature_index] < threshold

        y_left, y_right = y_subset[left_indeces], y_subset[~left_indeces]

        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        best_score = np.inf
        best_feature_index, best_threshold = None, None
        sample_size = X_subset.shape[0]

        for feature_index in np.arange(X_subset.shape[1]):
            thresholds = np.unique(X_subset[:, feature_index])
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2

            for threshold in thresholds:
                y_left, y_right = self.make_split_only_y(feature_index, threshold, X_subset, y_subset)

                left_weight = y_left.shape[0] / sample_size
                right_weight = y_right.shape[0] / sample_size

                current_score = left_weight * self.criterion(y_left) + right_weight * self.criterion(y_right)

                if best_score > current_score:
                    best_score = current_score
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        if X_subset.shape[0] < self.min_samples_split or self.depth > self.max_depth:  # or \
            # (self.classification and len(np.unique(y_subset)) == 1):

            new_node = None

            if self.classification:
                class_probas = y_subset.mean(axis=0)
                predicted_class = class_probas.argmax()
                new_node = Node(None, predicted_class, class_probas)

            else:
                predicted_value = y_subset.mean()
                new_node = Node(None, predicted_value)

            return new_node

        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        new_node = Node(feature_index, threshold)

        # if self.classification:
        #   new_node.proba = y_subset.mean(axis = 0)
        #   new_node.value = new_node.proba.argmax()
        # else:
        #   new_node.value = y_subset.mean()

        (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)

        self.depth += 1
        left_child = self.make_tree(X_left, y_left)
        right_child = self.make_tree(X_right, y_right)
        self.depth -= 1

        new_node.left_child = left_child
        new_node.right_child = right_child

        return new_node

    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification
                   of type float with shape (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification
                   (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """

        # YOUR CODE HERE
        n_samples = X.shape[0]
        y_predicted = np.zeros(n_samples)

        for i, sample in enumerate(X):
            node = self.root
            while node.left_child:
                if sample[node.feature_index] < node.value:
                    node = node.left_child
                else:
                    node = node.right_child
                y_predicted[i] = node.value

        return y_predicted.reshape(-1, 1)

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        n_samples = X.shape[0]
        y_predicted_probs = np.zeros((n_samples, self.n_classes))

        for i, sample in enumerate(X):
            node = self.root
            while node.left_child:
                if sample[node.feature_index] < node.value:
                    node = node.left_child
                else:
                    node = node.right_child
                y_predicted_probs[i] = node.proba

        return y_predicted_probs
