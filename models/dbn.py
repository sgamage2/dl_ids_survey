from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
import models.ann
import logging, hashlib
import utility


class DBN:
    def __init__(self): # Minimal constructor
        pass

    def initialize(self, exp_params):
        self.params = exp_params

        layer_nodes = exp_params['dbn_layer_units']
        batch_size = exp_params['batch_size']
        pretrain_epochs = self.params['pretrain_epochs']

        # Pretraining params
        rbm_learning_rate = exp_params['dbn_learning_rate']

        self.rbms = []
        steps = []
        for i, nodes in enumerate(layer_nodes):
            rbm = BernoulliRBM(random_state=0, verbose=True, learning_rate=rbm_learning_rate,
                               n_components=nodes, n_iter=pretrain_epochs, batch_size=batch_size)
            self.rbms.append(rbm)
            rbm_name = 'rbm_' + str(i)
            steps.append((rbm_name, rbm))

        self.dbn = Pipeline(steps=steps)

        self.ann = None # The final feed forward net to be fine tuned (to be created after DBN is pretrained)

        # logger = logging.getLogger(__name__)
        # self.dbn.summary(print_fn=logger.info)

    # To be used after the model has been initialized for first time
    def set_params(self, exp_params):
        self.params['epochs'] = exp_params['epochs']
        self.params['early_stop_patience'] = exp_params['early_stop_patience']

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None, parent=None):
        # Use a portion of the training set for unsupervised DB pre-training
        unsupervised_ratio = self.params['unsupervised_ratio']
        pretrain_epochs = self.params['pretrain_epochs']

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
            logging.info(
                'Split sizes (instances). total = {}, unsupervised = {}, supervised = {}, unsupervised dataset hash = {}'
                .format(X_train.shape[0], X_train_unsupervised.shape[0], X_train_supervised.shape[0], dataset_hash))

        if pretrain_epochs > 0:
            logging.info("Pretraining Deep Belief Network")
            self.dbn.fit(X_train_unsupervised)
            logging.info("Pretraining Complete")
            logging.info("Getting pretrained weights")
            all_layers_weights = self.get_dbn_params()
        else:
            logging.info("Pretraining is turned off. Skipping")
            all_layers_weights = None

        self.ann = self.create_ffw_ann(all_layers_weights)

        logging.info("Fine-tuning final neural network")
        history = self.ann.fit(X_train_supervised, y_train_supervised, X_valid, y_valid, X_test, y_test, self)    # Fine tune only

        return history

    def create_ffw_ann(self, all_layers_weights):
        logging.info('Creating and initializing feed forward neural network')
        params = dict(self.params)
        params['ann_layer_units'] = self.params['dbn_layer_units']
        params['epochs'] = self.params['fine_tune_epochs']

        ann = models.ann.ANN()
        ann.initialize(params)

        if all_layers_weights:
            ann.set_layers_weights(all_layers_weights)

        return ann

    # def predict(self, X):
    #     return self.dbn.predict(X)
    #
    # def predict_classes(self, X):
    #     class_probs = self.dbn.predict(X)
    #     classes = np.argmax(class_probs, axis=1)
    #     return classes

    def get_dbn_params(self):
        dbn_weights = []
        for rbm in self.rbms:
            rbm_weights = [rbm.components_.T, rbm.intercept_hidden_]
            dbn_weights.append(rbm_weights)

        return dbn_weights

    def predict(self, X):
        return self.ann.predict(X)

    def predict_classes(self, X):
        return self.ann.predict_classes(X)

    def save(self, filename, **kwargs):
        self.dbn.save(filename, kwargs)


if __name__ == "__main__":
    assert False    # Not supposed to be run as a separate script
