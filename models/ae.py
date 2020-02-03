import logging

from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras import regularizers
from keras.callbacks import EarlyStopping, TensorBoard
from keras_callbacks import ModelCheckpointer, Metrics, WeightRestorer


class AE:
    def __init__(self): # Minimal constructor
        pass

    def initialize(self, exp_params):
        self.params = exp_params
        self.ae = Sequential()

        input_nodes = exp_params['input_nodes']
        output_nodes = input_nodes

        encoder_nodes = exp_params['ae_encoder_units']
        encoder_activations = exp_params['ae_encoder_activations']
        encoder_dropouts = exp_params['ae_encoder_dropout_rates']

        decoder_nodes = exp_params['ae_decoder_units']
        decoder_activations = exp_params['ae_decoder_activations']
        decoder_dropouts = exp_params['ae_decoder_dropout_rates']

        encoder_l1_param = exp_params['ae_encoder_l1_param']
        loss_function = exp_params['loss_function']
        output_activation = exp_params['output_activation']

        # List with an L1 activity_regularizer only for the endoded layer (last encoder hidden layer)
        regs = [None] * len(encoder_nodes)
        # regs[-1] = regularizers.l1(encoder_l1_param)

        # First encoder hidden layer (need to specify input layer nodes here)
        self.ae.add(Dense(encoder_nodes[0], input_dim=input_nodes, activation=encoder_activations[0],
                          activity_regularizer=regs[0], kernel_initializer='he_uniform', bias_initializer='zeros'))
        self.ae.add(BatchNormalization())
        self.ae.add(Dropout(encoder_dropouts[0]))

        last_encoder_dense_layer = self.ae.layers[-1]

        # Other encoder hidden layers before the encoded layer
        for i in range(1, len(encoder_nodes)):  #
            self.ae.add(Dense(encoder_nodes[i], activation=encoder_activations[i],
                              activity_regularizer=regs[i], kernel_initializer='he_uniform', bias_initializer='zeros'))
            last_encoder_dense_layer = self.ae.layers[-1]
            self.ae.add(BatchNormalization())
            self.ae.add(Dropout(encoder_dropouts[i]))

        # Create and store the encoder model (for encoding in transform() function)
        # self.encoder = Model(self.ae.input, self.ae.layers[-1].output)
        self.encoder = Model(self.ae.input, last_encoder_dense_layer.output)

        # Placeholder for encoded values
        encoded_dim = encoder_nodes[-1]
        encoded_input = Input(shape=(encoded_dim,))
        decoded = encoded_input # See loop below

        # Decoder hidden layers
        for i in range(0, len(decoder_nodes)):
            self.ae.add(Dense(decoder_nodes[i], activation=decoder_activations[i],
                              kernel_initializer='he_uniform', bias_initializer='zeros'))
            decoded = self.ae.layers[-1](decoded)   # String the decoder layers for creating the decoder model
            self.ae.add(BatchNormalization())
            self.ae.add(Dropout(decoder_dropouts[i]))

        if output_activation == 'sigmoid' or output_activation == 'tanh':
            initializer = 'glorot_uniform'
        elif output_activation == 'relu':
            initializer = 'he_uniform'

        # Output layer
        self.ae.add(Dense(output_nodes, activation=output_activation,
                          kernel_initializer=initializer, bias_initializer='zeros'))
        decoded = self.ae.layers[-1](decoded)   # String the final decoder layer

        # Create and store the decoder model (for decoding in inverse_transform() function)
        self.decoder = Model(encoded_input, decoded)

        self.ae.compile(optimizer='adam', loss=loss_function)

        logger = logging.getLogger(__name__)
        self.ae.summary(print_fn=logger.info)

    # To be used after the model has been initialized for first time
    def set_params(self, exp_params):
        self.params['epochs'] = exp_params['epochs']
        self.params['early_stop_patience'] = exp_params['early_stop_patience']

    def fit(self, X_train, y_train=None, X_valid=None, y_valid=None):
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']
        early_stop_patience = self.params['early_stop_patience']

        # --------------------------------
        # Create callbacks

        tensorboard_cb = TensorBoard(log_dir=self.params['tensorboard_log_dir'], batch_size=batch_size)

        interval = max(epochs // 10, 10)
        checkpointer_cb = ModelCheckpointer(model_wrapper=self, checkpoint_epoch_interval=interval,
                                            save_directory=self.params['results_dir'], filename_prefix='ae')

        callbacks = [tensorboard_cb, checkpointer_cb]

        if X_valid is not None:
            weight_restorer_cb = WeightRestorer(epochs)
            callbacks.append(weight_restorer_cb)

        # Early stopping is enabled
        if X_valid is not None and early_stop_patience > 0 and early_stop_patience < epochs:
            early_stopping_cb = EarlyStopping(monitor='val_loss', patience=early_stop_patience,
                                          verbose=2, mode='auto', restore_best_weights=False)
            callbacks.append(early_stopping_cb)

        # --------------------------------

        if X_valid is not None:
            validation_data = (X_valid, X_valid)
        else:
            validation_data = None

        history = self.ae.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                                validation_data=validation_data, callbacks=callbacks, shuffle=True, verbose=2)

        # --------------------------------
        # Evaluate loss of final (best-weights-restored) model
        # These values can be different from the values seen during training (due to batch norm and dropout)

        train_loss = self.ae.evaluate(X_train, X_train, verbose=0)
        val_loss = None
        if X_valid is not None:
            val_loss = self.ae.evaluate(X_valid, X_valid, verbose=0)

        logging.info('Last epoch loss evaluation: train_loss = {:.6f}, val_loss = {:.6f}'.format(train_loss, val_loss))

        return history

    def predict(self, X):
        # return self.decoder.predict(self.encoder.predict(X))  # In two steps: encode, then decode
        return self.ae.predict(X)   # In one step

    # Return encoded (transformed) features
    def transform(self, X):
        return self.encoder.predict(X)

    # Return decoded (inverse transformed) features
    def inverse_transform(self, X_encoded):
        return self.decoder.predict(X_encoded)

    def predict_classes(self, X):
        return self.ae.predict_classes(X)

    def save(self, filename, **kwargs):
        self.ae.save(filename, kwargs)


if __name__ == "__main__":
    assert False    # Not supposed to be run as a separate script
