from sklearn.ensemble import RandomForestClassifier
import logging
import numpy as np
from random import randint


class RandomForest:
    def __init__(self):  # Minimal constructor
        pass

    def initialize(self, exp_params):
        self.params = exp_params

        num_trees = exp_params['num_trees']
        min_samples_for_split = exp_params['min_samples_for_split']
        randomize_state = exp_params['randomize_state']

        if randomize_state != 0:
            randomize_state = randint(0, 1000000)

        self.rf = RandomForestClassifier(n_estimators=num_trees, min_samples_split=min_samples_for_split,
                                         n_jobs=4, random_state=randomize_state, verbose=1)

    # To be used after the model has been initialized for first time
    def set_params(self, exp_params):
        assert False    # Does not support continued training (need to use for that warm_start; verify)

    # def fit(self, X, y):
    #     return

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None, parent=None):
        self.rf.fit(X_train, y_train)
        return None

    def predict(self, X):
        # return self.predict_classes(X)
        return self.rf.predict_proba(X)

    def predict_classes(self, X):
        label_arr = self.rf.predict(X)  # Array with 1 for predicted class, 0 others
        label = np.argmax(label_arr, axis=1)
        return label

    def save(self, filename, **kwargs):
        assert False


if __name__ == "__main__":
    assert False  # Not supposed to be run as a separate script
