from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
import logging
from keras_callbacks import ModelCheckpointer, Metrics, WeightRestorer, StopperOnGoal


class LSTMClassifer:
    def __init__(self): # Minimal constructor
        pass

    def initialize(self, exp_params):
        self.params = exp_params
        self.lstm = Sequential()

        time_steps = exp_params['lstm_time_steps']
        input_features = exp_params['input_nodes']
        num_classes = exp_params['output_nodes']
        layer_nodes = exp_params['lstm_layer_units']
        activations = exp_params['lstm_layer_activations']
        dropouts = exp_params['lstm_layer_dropout_rates']

        # First hidden layer (need to specify input layer nodes here)
        self.lstm.add(LSTM(units=layer_nodes[0], return_sequences=True, input_shape=(time_steps, input_features), activation=activations[0]))
        self.lstm.add(BatchNormalization())
        self.lstm.add(Dropout(dropouts[0]))

        # Other hidden layers
        for i in range(1, len(layer_nodes)):
            self.lstm.add(LSTM(units=layer_nodes[i], return_sequences=True, activation=activations[i]))
            self.lstm.add(BatchNormalization())
            self.lstm.add(Dropout(dropouts[i]))

        # Output layer (Synced to sequence: each time step has an output label)
        self.lstm.add(TimeDistributed(Dense(num_classes, activation='softmax')))

        self.lstm.compile(optimizer='adam', loss='binary_crossentropy')

        # Following is also valid: need to explore more. But when using binary_crossentropy loss, set metrics=[categorical_accuracy]
        # self.ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])

        logger = logging.getLogger(__name__)
        self.lstm.summary(print_fn=logger.info)

    # To be used after the model has been initialized for first time
    def set_params(self, exp_params):
        self.params['epochs'] = exp_params['epochs']
        self.params['early_stop_patience'] = exp_params['early_stop_patience']

    def fit(self, X_train, y_train, X_valid, y_valid, X_test=None, y_test=None, parent=None):
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']
        early_stop_patience = self.params['early_stop_patience']

        # --------------------------------
        # Create callbacks

        tensorboard_cb = TensorBoard(log_dir=self.params['tensorboard_log_dir'], batch_size=batch_size)

        interval = max(epochs // 10, 10)
        checkpointer_cb = ModelCheckpointer(model_wrapper=self, checkpoint_epoch_interval=interval,
                                            save_directory=self.params['results_dir'])

        metrics_cb = Metrics()

        callbacks = [tensorboard_cb, checkpointer_cb, metrics_cb]

        # Stop training when a goal metric value is reached
        if 'goal_metric' in self.params and self.params['goal_metric'] > 0:
            assert X_test is not None
            assert y_test is not None
            goal_stopper_cb = StopperOnGoal(self, X_test, y_test, self.params['goal_metric'], "accuracy")
            callbacks.append(goal_stopper_cb)

        if X_valid is not None:
            weight_restorer_cb = WeightRestorer(epochs)
            callbacks.append(weight_restorer_cb)

        # Early stopping is enabled
        if X_valid is not None and early_stop_patience > 0 and early_stop_patience < epochs:
            early_stopping_cb = EarlyStopping(monitor='val_loss', patience=early_stop_patience,
                                              verbose=2, mode='auto', restore_best_weights=False)
            callbacks.append(early_stopping_cb)

        # --------------------------------

        if X_valid is not None and y_valid is not None:
            validation_data = (X_valid, y_valid)
        else:
            validation_data = None

        history = self.lstm.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                validation_data=validation_data, callbacks=callbacks, verbose=2)

        history.history['val_f1'] = metrics_cb.val_f1s

        # --------------------------------
        # Evaluate loss of final (best-weights-restored) model
        # These values can be different from the values seen during training (due to batch norm and dropout)

        train_loss = self.lstm.evaluate(X_train, y_train, verbose=0)
        val_loss = None
        if X_valid is not None and y_valid is not None:
            val_loss = self.lstm.evaluate(X_valid, y_valid, verbose=0)

        logging.info('Last epoch loss evaluation: train_loss = {:.6f}, val_loss = {:.6f}'.format(train_loss, val_loss))

        return history

    def predict(self, X):
        return self.lstm.predict(X)

    def predict_classes(self, X):
        return self.lstm.predict_classes(X)

    def save(self, filename, **kwargs):
        self.lstm.save(filename, kwargs)


if __name__ == "__main__":
    assert False    # Not supposed to be run as a separate script
