import logging
import models.ann
import models.ae
import utility
import hashlib


class AE_ANN:
    def __init__(self): # Minimal constructor
        pass

    def initialize(self, exp_params):
        self.params = exp_params

        self.ae = models.ae.AE()
        self.ann = models.ann.ANN()

        ae_exp_params = dict(exp_params)
        ae_exp_params['epochs'] = exp_params['ae_epochs']
        ae_exp_params['goal_metric'] = -1   # Disable for AE training
        self.ae.initialize(ae_exp_params)

        ann_exp_params = dict(exp_params)
        ann_exp_params['input_nodes'] = ae_exp_params['ae_encoder_units'][-1]
        ann_exp_params['epochs'] = exp_params['ann_epochs']
        # ann_exp_params['output_nodes'] = num_classes  # Already set to num_classes in  exp_params
        self.ann.initialize(ann_exp_params)

    # To be used after the model has been initialized for first time
    def set_params(self, exp_params):
        self.params['epochs'] = exp_params['epochs']
        self.params['early_stop_patience'] = exp_params['early_stop_patience']

    def fit(self, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None, parent=None):
        # Use a portion of the training set for unsupervised AE training
        unsupervised_ratio = self.params['unsupervised_ratio']

        if unsupervised_ratio == -1:
            logging.info('Dataset splitting disabled. Use the full dataset for both AE and ANN training')
            (X_train_unsupervised, y_train_unsupervised) = X_train, y_train
            (X_train_supervised, y_train_supervised) = X_train, y_train

        else:
            random_seed = None
            if 'split_random_seed' in self.params:
                random_seed = self.params['split_random_seed']

            logging.info('Splitting train set into 2 sets (unsupervised, supervised), random_seed = {}'
                         .format(random_seed))
            splits = utility.split_dataset(X_train, y_train, [unsupervised_ratio, 1 - unsupervised_ratio], random_seed)
            (X_train_unsupervised, y_train_unsupervised), (X_train_supervised, y_train_supervised) = splits
            dataset_hash = hashlib.sha1(str(X_train_unsupervised).encode('utf-8')).hexdigest()
            logging.info('Split sizes (instances). total = {}, unsupervised = {}, supervised = {}, unsupervised dataset hash = {}'
                .format(X_train.shape[0], X_train_unsupervised.shape[0], X_train_supervised.shape[0], dataset_hash))

        logging.info("Training autoencoder")
        self.ae.fit(X_train_unsupervised, None, X_valid, None)
        logging.info("Training autoencoder complete")

        logging.info("Encoding data for supervised training")
        X_train_sup_encoded = self.ae.transform(X_train_supervised)
        X_val_sup_encoded = None
        if X_valid is not None:
            X_val_sup_encoded = self.ae.transform(X_valid)
        logging.info("Encoding complete")

        logging.info("Training neural network layers (after autoencoder)")
        history = self.ann.fit(X_train_sup_encoded, y_train_supervised, X_val_sup_encoded, y_valid, X_test, y_test, self)

        return history

    def predict(self, X):
        X_encoded = self.ae.transform(X)
        return self.ann.predict(X_encoded)

    def predict_classes(self, X):
        X_encoded = self.ae.transform(X)
        return self.ann.predict_classes(X_encoded)

    # Return encoded (transformed) features
    def transform(self, X):
        return self.ae.transform(X)

    # Return decoded (inverse transformed) features
    def inverse_transform(self, X_encoded):
        return self.ae.inverse_transform(X_encoded)

    def save(self, filename, **kwargs):
        self.ae.save(filename, kwargs)


if __name__ == "__main__":
    assert False    # Not supposed to be run as a separate script
