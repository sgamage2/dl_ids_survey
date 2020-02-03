import utility
import models.ann
import models.ae
import models.ae_ann
import models.dbn
import models.lstm
import models.rf

import sys, os, time, logging, csv, glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')

# Common params
hdf_key = 'my_key'


def setup_logging(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info('Created directory: {}'.format(output_dir))

    # Setup logging
    log_filename = output_dir + '/' + 'run_log.log'

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename, 'w+'),
                  logging.StreamHandler()],
        level=logging.INFO
    )
    logging.info('Initialized logging. log_filename = {}'.format(log_filename))


def create_tensorboard_log_dir(results_dir):
    log_dir = results_dir + '/tf_logs_run_' + time.strftime("%Y_%m_%d-%H_%M_%S")
    os.makedirs(log_dir)
    logging.info('Created tensorboard log directory: {}'.format(log_dir))
    return log_dir

def print_help():
    logging.info('Usage:')
    logging.info('./run_experiments.py <experiments_filename>')


def get_filename_from_args():
    num_args = len(sys.argv)

    if num_args != 2:
        print_help()
        sys.exit()

    filename = sys.argv[1]
    isfile = os.path.isfile(filename)

    if not isfile:
        logging.info('File {} does not exist'.format(filename))
        sys.exit()

    return filename


def get_experiments(filename):
    experiments = []

    with open(filename, mode='r') as experiments_file:
        csv_dict_reader = csv.DictReader(filter(lambda row: row[0]!='#', experiments_file))

        for row in csv_dict_reader:
            experiments.append(row)

    logging.info('Read {} experiments from file: {}'.format(len(experiments), filename))

    return experiments


def convert_params_to_correct_types(params):
    converted_params = {}

    for key, val in params.items():
        new_val = utility.convert(val)

        if type(new_val) == str:    # If param is a comma separated string, convert it to a list of the elements
            elements = new_val.split(",")
            if len(elements) > 1:
                new_val = [utility.convert(x) for x in elements]

                if new_val[-1] == '':   # One-element string (nothing after the comma)
                    new_val = new_val[:-1]

        converted_params[key] = new_val

    return converted_params


def load_datasets(data_dir):
    # Lists of train, val, test files (X and y)
    X_train_files = glob.glob(data_dir + '/' + 'X_train*')
    y_train_files = glob.glob(data_dir + '/' + 'y_train*')
    X_val_files = glob.glob(data_dir + '/' + 'X_val*')
    y_val_files = glob.glob(data_dir + '/' + 'y_val*')
    X_test_files = glob.glob(data_dir + '/' + 'X_test*')
    y_test_files = glob.glob(data_dir + '/' + 'y_test*')

    X_train_files.sort()
    y_train_files.sort()
    X_val_files.sort()
    y_val_files.sort()
    X_test_files.sort()
    y_test_files.sort()

    assert len(X_train_files) > 0
    assert len(y_train_files) > 0

    logging.info('Reading X, y files')
    X_train_dfs = [utility.read_hdf(file, hdf_key) for file in X_train_files]
    X_val_dfs = [utility.read_hdf(file, hdf_key) for file in X_val_files]
    X_test_dfs = [utility.read_hdf(file, hdf_key) for file in X_test_files]
    y_train_dfs = [utility.read_hdf(file, hdf_key) for file in y_train_files]
    y_val_dfs = [utility.read_hdf(file, hdf_key) for file in y_val_files]
    y_test_dfs = [utility.read_hdf(file, hdf_key) for file in y_test_files]

    X_train = pd.concat(X_train_dfs)
    X_val = pd.concat(X_val_dfs)
    X_test = pd.concat(X_test_dfs)
    y_train = pd.concat(y_train_dfs)
    y_val = pd.concat(y_val_dfs)
    y_test = pd.concat(y_test_dfs)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_model(model, exp_params, datasets, is_first_time=True):
    if is_first_time:
        logging.info('Initializing model')
        model.initialize(exp_params)
    else:
        model.set_params(exp_params)

    logging.info('Training model')
    t0 = time.time()

    training_data_feed = exp_params['training_data_feed']
    if training_data_feed == 'preload':
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets
        history = model.fit(X_train, y_train, X_val, y_val, X_test, y_test) # X_test, y_test is only for StopperOnGoal callback in ANN

    time_to_train = time.time() - t0
    logging.info('Training complete. time_to_train = {:.2f} sec, {:.2f} min'.format(time_to_train, time_to_train / 60))

    # Save the best trained model (in case we need to continue training from this point on)
    # We assume that the model classes will restore the best weights (epoch with lowest validation error) at end of training
    model_filepath = exp_params['results_dir'] + '/best_model.pickle'
    utility.save_obj_to_disk(model, model_filepath)
    logging.info('Model saved to {}'.format(model_filepath))

    # Save training history graph
    if history is not None:
        utility.save_training_history(history, exp_params['results_dir'])
        utility.plot_training_history(history, exp_params['results_dir'])


def evaluate_classification_and_report(y_true, y_pred, excel_writer, dataset_name, normal_label):
    utility.print_evaluation_report(y_pred, y_true, dataset_name)

    labels, multi_class_conf_mat = utility.get_conf_mat(y_pred, y_true)
    multi_class_metrics, mc_avg_metrics = utility.evaluate_on_conf_mat(labels, multi_class_conf_mat)

    utility.print_avg_metrics(dataset_name, mc_avg_metrics)

    labels, binary_conf_mat = utility.convert_to_binary_conf_mat(labels, multi_class_conf_mat, normal_label, 'attack')
    binary_metrics, bin_avg_metrics = utility.evaluate_on_conf_mat(labels, binary_conf_mat)

    curr_row = utility.write_to_excel(excel_writer, dataset_name, multi_class_metrics, mc_avg_metrics)
    curr_row = utility.write_to_excel(excel_writer, dataset_name, binary_metrics, bin_avg_metrics, start_row=curr_row)


def evaluate_reconstruction_and_report(x_true, x_recon, excel_writer, dataset_name):
    reconstruction_mse = ((x_true - x_recon) ** 2).sum(axis=1) # Reconstruction error of each example
    dataset_mse = reconstruction_mse.mean()
    dataset_rmse = dataset_mse ** 0.5

    logging.info('{} dataset reconstruction MSE = {:.2f}, RMSE = {:.2f}'.format(dataset_name, dataset_mse, dataset_rmse))

    return reconstruction_mse


def compute_prediction_time(model, exp_params, datasets):
    reps = exp_params['repetitions']
    num_samples = exp_params['num_samples']

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets

    total_samples = 0
    for i in range(1, reps+1):
        logging.info('Making predictions: iteration {}'.format(i))

        if exp_params['model'] == 'lstm':
            X = X_train[np.random.randint(X_train.shape[0], size=num_samples), :, :]
        else:
            X = X_train.sample(num_samples)

        # if type(X_train) != pd.DataFrame:   # This is the case with lstm flows
        #     X_train = pd.DataFrame(X_train)

        t0 = time.time()
        y_pred = model.predict_classes(X)
        time_to_predict = time.time() - t0

        total_samples += y_pred.shape[0]

        logging.info('Making predictions complete. time_to_predict = {:.6f} sec'.format(time_to_predict))

    logging.info('Total no. of predictions = {}'.format(total_samples))


def test_classifier(label_encoder, model, exp_params, datasets):
    logging.info('Making predictions on training, validation, testing data')

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets

    # Make predictions
    t0 = time.time()
    y_train_pred = model.predict_classes(X_train)
    if X_val is not None:
        y_val_pred = model.predict_classes(X_val)
    y_test_pred = model.predict_classes(X_test)

    time_to_predict = time.time() - t0
    logging.info('Making predictions complete. time_to_predict = {:.2f} sec, {:.2f} min'.format(time_to_predict, time_to_predict / 60))

    # Convert integer predictions to original string labels
    y_train_pred_inv = label_encoder.inverse_transform(y_train_pred.flatten())
    if X_val is not None:
        y_val_pred_inv = label_encoder.inverse_transform(y_val_pred.flatten())
    y_test_pred_inv = label_encoder.inverse_transform(y_test_pred.flatten())

    logging.info('Evaluating predictions (results)')

    # Evaluate predictions (metrics), write results to Excel file

    exp_name = exp_params['results_dir'].split('/')[-1]
    results_file = exp_params['results_dir'] + '/' + exp_name + '_results.xlsx'
    excel_writer = pd.ExcelWriter(results_file)

    # Eval and report on Test, Val, Train sets
    evaluate_classification_and_report(y_test, y_test_pred_inv, excel_writer, "Testing", exp_params['normal_label'])
    if X_val is not None:
        evaluate_classification_and_report(y_val, y_val_pred_inv, excel_writer, "Validation", exp_params['normal_label'])
    evaluate_classification_and_report(y_train, y_train_pred_inv, excel_writer, "Training", exp_params['normal_label'])

    excel_writer.save()
    logging.info('Results saved to: {}'.format(results_file))


def create_model(model_name):
    if model_name == 'ann':
        model = models.ann.ANN()
    elif model_name == 'ae':
        model = models.ae.AE()
    elif model_name == 'ae_ann':
        model = models.ae_ann.AE_ANN()
    elif model_name == 'dbn':
        model = models.dbn.DBN()
    elif model_name == 'lstm':
        model = models.lstm.LSTMClassifer()
    elif model_name == 'rf':
        model = models.rf.RandomForest()
    else:
        assert False

    return model


# Some params are not required for all experiment specs
# But they are needed for the code to be written in a consistent and readable manner
# Therefore, sensible default values are set for those specs here
def set_exp_param_defaults(exp_params):
    model_origin = exp_params.get('model_origin', 'new')
    action = exp_params.get('action', 'train_test')
    training_data_feed = exp_params.get('training_data_feed', 'preload')
    training_set = exp_params.get('training_set', 'train_set_only')
    class_weights = exp_params.get('class_weights', 0)

    tensorboard_log_dir = create_tensorboard_log_dir(exp_params['results_dir'])

    exp_params['model_origin'] = model_origin
    exp_params['action'] = action
    exp_params['training_data_feed'] = training_data_feed
    exp_params['tensorboard_log_dir'] = tensorboard_log_dir
    exp_params['training_set'] = training_set
    exp_params['class_weights'] = class_weights


def test_model(model_type, label_encoder, model, exp_params, orig_datasets):
    if model_type == 'classifier':
        training_data_feed = exp_params['training_data_feed']
        if training_data_feed == 'preload':
            test_classifier(label_encoder, model, exp_params, orig_datasets)
        else:
            assert False
    else:
        assert False


def statistical_test_two_models(model1, model2, exp_params, datasets, label_encoder):
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, f1_score
    from scipy.stats import ttest_ind

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    kf = KFold(n_splits=10)

    logging.info('Running CV evaluations')

    model_1_metrics = []
    model_2_metrics = []

    fold = 1
    for train_index, test_index in kf.split(X_test):
        print('Evaluating CV fold {}'.format(fold))
        X1, X2 = X_test[train_index], X_test[test_index]
        y1, y2 = y_test[train_index], y_test[test_index]
        y_true = y2

        y_model1 = model1.predict_classes(X2)
        y_model1 = label_encoder.inverse_transform(y_model1.flatten())  # Convert integer labels to original strings

        y_model2 = model2.predict_classes(X2)
        y_model2 = label_encoder.inverse_transform(y_model2.flatten())  # Convert integer labels to original strings

        a1 = accuracy_score(y_true, y_model1)
        a2 = accuracy_score(y_true, y_model2)

        model_1_metrics.append(a1)
        model_2_metrics.append(a2)

        fold += 1

    print(model_1_metrics)
    print(model_2_metrics)

    logging.info('Running ttest_ind (independent samples')

    ttest, pval = ttest_ind(model_1_metrics, model_2_metrics)

    print('ttest_ind: t statistic = {:.4f}, p-value = {}'.format(ttest, pval))


def run_experiment_actions(model_origin, model_type, model_name, action, resources):
    exp_params, orig_datasets, label_enc_datasets, label_encoder = resources

    # ---------------------------
    # Set variables based on experiment params

    if model_origin == 'new':
        model = create_model(model_name)
        is_first_time = True
    elif model_origin == 'loaded':
        model_location = exp_params['model_location']
        logging.info('Loading model from file: {}'.format(model_location))
        model = utility.load_obj_from_disk(model_location)
        is_first_time = False

        if 'model2_location' in exp_params:
            model2_location = exp_params['model2_location']
            logging.info('Loading model 2 from file: {}'.format(model2_location))
            model2 = utility.load_obj_from_disk(model2_location)
    else:
        assert False

    # ---------------------------
    # Run the experiment actions

    if action == 'train_test':
        train_model(model, exp_params, label_enc_datasets, is_first_time)
        test_model(model_type, label_encoder, model, exp_params, orig_datasets)
        # test_model_func()
    elif action == 'test_only':
        test_model(model_type, label_encoder, model, exp_params, orig_datasets)
    elif action == 'compute_prediction_time':
        compute_prediction_time(model, exp_params, orig_datasets)
    elif action == 'stat_test_2_models':
        statistical_test_two_models(model, model2, exp_params, orig_datasets, label_encoder)
    else:
        assert False  # Unknown action


def prepare_tabular_data(dataset_dir, concat_train_valid):
    # Load data
    logging.info('Loading datsets from: {}'.format(dataset_dir))
    datasets_orig = load_datasets(dataset_dir)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets_orig

    if concat_train_valid:
        X_train = pd.concat([X_train, X_val])
        y_train = pd.concat([y_train, y_val])
        X_val = None
        y_val = None

    # One-hot encode class labels (needed as output layer has multiple nodes)
    label_encoder, unused = utility.encode_labels(pd.concat([y_train, y_val, y_test]), encoder=None)
    unused, y_train_enc = utility.encode_labels(y_train, encoder=label_encoder)
    y_val_enc = None
    if X_val is not None:
        unused, y_val_enc = utility.encode_labels(y_val, encoder=label_encoder)
    unused, y_test_enc = utility.encode_labels(y_test, encoder=label_encoder)

    datasets_orig = (X_train, y_train), (X_val, y_val), (X_test, y_test)
    datasets_enc = (X_train, y_train_enc), (X_val, y_val_enc), (X_test, y_test_enc)

    return datasets_orig, datasets_enc, label_encoder


def prepare_sequence_data(dataset_dir, time_steps, concat_train_valid):
    # Load data
    logging.info('Loading datsets from: {}'.format(dataset_dir))
    datasets_loaded = load_datasets(dataset_dir)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets_loaded

    if concat_train_valid:
        X_train = pd.concat([X_train, X_val])
        y_train = pd.concat([y_train, y_val])
        X_val = None
        y_val = None

    # datasets_orig = (X_train, y_train), (X_val, y_val), (X_test, y_test)

    # Prepare sequences of flows (for LSTM input)
    logging.info('Preparing flow sequences')
    t0 = time.time()

    X_train, y_train_seq = utility.extract_flow_sequences(X_train, y_train, time_steps, None)

    y_val_seq = None
    if X_val is not None:
        X_val, y_val_seq = utility.extract_flow_sequences(X_val, y_val, time_steps, None)
    X_test, y_test_seq = utility.extract_flow_sequences(X_test, y_test, time_steps, None)

    logging.info('Extracting flows complete. time_taken = {:.2f} sec'.format(time.time() - t0))

    # One-hot encode class labels (needed as output layer has multiple nodes)
    y_list = [y_train_seq.flatten()]
    if y_val_seq is not None:
        y_list.append(y_val_seq.flatten())
    y_list.append(y_test_seq.flatten())
    all_y = np.hstack(y_list)

    label_encoder, all_y_enc = utility.encode_labels(all_y, encoder=None)
    unused, y_train_enc = utility.encode_labels(y_train_seq.flatten(), encoder=label_encoder)
    y_val_enc = None
    if y_val_seq is not None:
        unused, y_val_enc = utility.encode_labels(y_val_seq.flatten(), encoder=label_encoder)
    unused, y_test_enc = utility.encode_labels(y_test_seq.flatten(), encoder=label_encoder)

    y_train_enc = y_train_enc.reshape(y_train_seq.shape[0], y_train_seq.shape[1], all_y_enc.shape[1])
    if y_val_seq is not None:
        y_val_enc = y_val_enc.reshape(y_val_seq.shape[0], y_val_seq.shape[1], all_y_enc.shape[1])
    y_test_enc = y_test_enc.reshape(y_test_seq.shape[0], y_test_seq.shape[1], all_y_enc.shape[1])

    # batch_size * time_steps in the prepared seq
    train_num_flows = X_train.shape[0] * X_train.shape[1]
    test_num_flows = X_test.shape[0] * X_test.shape[1]

    y_values = None
    if y_val is not None:
        val_num_flows = X_val.shape[0] * X_val.shape[1]
        y_values = y_val[0:val_num_flows]

    datasets_orig = (X_train, y_train[0:train_num_flows]), (X_val, y_values), (X_test, y_test[0:test_num_flows])

    datasets_enc = (X_train, y_train_enc), (X_val, y_val_enc), (X_test, y_test_enc)

    return datasets_orig, datasets_enc, label_encoder


def run_experiment(exp_params):
    setup_logging(exp_params['results_dir'])

    exp_num = exp_params['experiment_num']
    logging.info('================= Running experiment no. {}  ================= \n'.format(exp_num))

    logging.info('Experiment parameters given below')
    logging.info("\n{}".format(exp_params))

    set_exp_param_defaults(exp_params)

    model_origin = exp_params['model_origin']
    model_type = exp_params['model_type']
    model = exp_params['model']
    training_data_feed = exp_params['training_data_feed']
    action = exp_params['action']
    dataset_dir = exp_params['dataset_dir']
    training_set = exp_params['training_set']

    concat_train_valid = False
    if training_set == 'train_valid_concat':
        concat_train_valid = True

    # ---------------------------------------
    # Prepare data

    if training_data_feed == 'preload':
        if model == 'lstm':
            time_steps = exp_params['lstm_time_steps']
            datasets_orig, datasets_enc, label_encoder = prepare_sequence_data(dataset_dir, time_steps, concat_train_valid)
            (X_train, y_train_enc), (X_val, y_val_enc), (X_test, y_test_enc) = datasets_enc
            # Set number of input nodes (features) and output nodes (no. of classes)
            exp_params['input_nodes'] = X_train.shape[2]
            exp_params['output_nodes'] = y_train_enc.shape[2]
        else:
            datasets_orig, datasets_enc, label_encoder = prepare_tabular_data(dataset_dir, concat_train_valid)
            (X_train, y_train_enc), (X_val, y_val_enc), (X_test, y_test_enc) = datasets_enc
            # Set number of input nodes (features) and output nodes (no. of classes)
            exp_params['input_nodes'] = X_train.shape[1]
            exp_params['output_nodes'] = y_train_enc.shape[1]
    else:
        assert False

    # ---------------------------------------
    # Run the experiment model

    exp_resources = (exp_params, datasets_orig, datasets_enc, label_encoder)

    run_experiment_actions(model_origin, model_type, model, action, exp_resources)

    logging.info('================= Finished running experiment no. {} ================= \n'.format(exp_num))


def main():
    # Disable GPU as it appears to be slower than CPU (to enable GPU, comment out this line)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Quick logging setup. Proper logging (to file) is setup later for each experiment
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)

    filename = get_filename_from_args()

    experiments = get_experiments(filename)
    # logging.info(experiments)

    logging.info('================= Started running experiments ================= \n')

    for exp in experiments:
        exp_params = convert_params_to_correct_types(exp)
        run_experiment(exp_params)

    logging.info('================= Finished running {} experiments ================= \n'.format(len(experiments)))


if __name__ == "__main__":
    main()
