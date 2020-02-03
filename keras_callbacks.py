from keras.callbacks import Callback, EarlyStopping
import utility
import logging
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# A Keras callback to save an ML model (of our custom type) at intervals of epochs during training
class ModelCheckpointer(Callback):
    def __init__(self, model_wrapper, checkpoint_epoch_interval, save_directory, filename_prefix=''):
        self.model_wrapper = model_wrapper
        self.checkpoint_epoch_interval = checkpoint_epoch_interval
        self.last_checkpoint_epoch = 0
        self.save_directory = save_directory
        self.filename_prefix = filename_prefix

    def on_epoch_end(self, epoch, logs=None):
        if epoch - self.last_checkpoint_epoch >= self.checkpoint_epoch_interval:
            self.last_checkpoint_epoch = epoch

            model_filepath = self.save_directory + '/' + self.filename_prefix + '_model_epoch_' + str(epoch) + '.pickle'

            utility.save_obj_to_disk(self.model_wrapper, model_filepath)
            logging.info('epoch = {}. Intermediate model saved to {}'.format(epoch, model_filepath))


# Track and save validation metrics other than loss
class Metrics(Callback):
    def __init__(self):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
        val_data = self.validation_data

        if len(val_data) > 0:
            val_predict = (np.asarray(self.model.predict(val_data[0])).round())
            val_targ = val_data[1]

            if len(val_predict.shape) == 3:   # LSTM output --> flatten the first 2 dimensions
                (num_seqs, seq_length, num_labels) = val_predict.shape
                val_predict = val_predict.reshape(num_seqs * seq_length, num_labels)
                val_targ = val_targ.reshape(num_seqs * seq_length, num_labels)

            val_f1 = f1_score(val_targ, val_predict, average='weighted')
            # val_recall = recall_score(val_targ, val_predict)
            # val_precision = precision_score(val_targ, val_predict)

            self.val_f1s.append(val_f1)
            # self.val_recalls.append(val_recall)
            # self.val_precisions.append(val_precision)

            print(" - val_f1: {:.4f}".format(val_f1))


# Keep track of the best weights, and restore them at the end of training
class WeightRestorer(EarlyStopping):
    def __init__(self, epochs):
        # The EarlyStopping super class tracks val_loss and the best weights
        # With patience=epochs+1, the early stopping will not triggered
        super(WeightRestorer, self).__init__(monitor='val_loss', patience=epochs+1, verbose=2,
                                             mode='auto', restore_best_weights=True)

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            logging.info('WeightRestorer::on_train_end(): restoring best weights')
            self.model.set_weights(self.best_weights)


class StopperOnGoal(Callback):
    def __init__(self, model_wrapper, out_of_sample_X, out_of_sample_y, goal_metric, metric_type):
        self.model_wrapper = model_wrapper
        self.out_of_sample_X = out_of_sample_X
        self.out_of_sample_y = out_of_sample_y
        self.goal_metric = goal_metric
        self.metric_type = metric_type

        assert metric_type in ["accuracy", "f1"]

        self.out_of_sample_f1s = []
        self.num_epochs = 0
        self.goal_reached = False


    def on_epoch_end(self, epoch, logs=None):
        X = self.out_of_sample_X
        y = self.out_of_sample_y
        self.num_epochs += 1

        predictions = (np.asarray(self.model_wrapper.predict(X)).round())

        if len(predictions.shape) == 3:   # LSTM output --> flatten the first 2 dimensions
            (num_seqs, seq_length, num_labels) = predictions.shape
            predictions = predictions.reshape(num_seqs * seq_length, num_labels)
            y = y.reshape(num_seqs * seq_length, num_labels)

        if self.metric_type == 'accuracy':
            metric = accuracy_score(y, predictions)
        elif self.metric_type == 'f1':
            metric = f1_score(y, predictions, average='weighted')

        self.out_of_sample_f1s.append(metric)

        print(" - out_of_sample_{}: {:.4f}".format(self.metric_type, metric))

        if metric >= self.goal_metric:
            logging.info('StopperOnGoal: reached goal_metric ({}). Stopping training. '
                         'goal_metric = {:.4f}, current_metric = {:.4f}, num_epochs = {}'.
                         format(self.metric_type, self.goal_metric, metric, self.num_epochs))
            self.model.stop_training = True
            self.goal_reached = True

    def on_train_end(self, logs=None):
        if self.goal_reached is False:
            logging.info('StopperOnGoal: did not reach goal, num_epochs = {}'.format(self.num_epochs))
