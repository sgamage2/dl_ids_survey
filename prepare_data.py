import os, logging
from pprint import pformat
import utility
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re, math


class Params:
    pass


# Script params
params = Params()

# Common params
params.hdf_key = 'my_key'
# ---- Small datasets
# params.output_dir = '../Datasets/small_datasets/kdd99'
# params.output_dir = '../Datasets/small_datasets/nsl_kdd_five_classes'
# params.output_dir = '../Datasets/small_datasets/nsl_kdd_five_classes_hard_test_set'
# params.output_dir = '../Datasets/small_datasets/ids2017'
# params.output_dir = '../Datasets/small_datasets/ids2018'
# ---- Full datasets
# params.output_dir = '../Datasets/full_datasets/kdd99_five_classes'
# params.output_dir = '../Datasets/full_datasets/ids2017'
# params.output_dir = '../Datasets/full_datasets/ids2018'
params.output_dir = '../Datasets/dummy'

# KDD99 params
params.kdd99_10_percent_dataset_file = '../Datasets/KDD_99/training_set/kddcup.data_10_percent_corrected'
params.kdd99_full_train_dataset_file = '../Datasets/KDD_99/training_set/kddcup.data.corrected'
params.kdd99_full_test_dataset_file = '../Datasets/KDD_99/test_set/corrected'
params.kdd99_map_to_five_classes = True

params.kdd_five_class_map = {
    'back': 'dos',
    'buffer_overflow': 'u2r',
    'ftp_write': 'r2l',
    'guess_passwd': 'r2l',
    'imap': 'r2l',
    'ipsweep': 'probe',
    'land': 'dos',
    'loadmodule': 'u2r',
    'multihop': 'r2l',
    'neptune': 'dos',
    'nmap': 'probe',
    'perl': 'u2r',
    'phf': 'r2l',
    'pod': 'dos',
    'portsweep': 'probe',
    'rootkit': 'u2r',
    'satan': 'probe',
    'smurf': 'dos',
    'spy': 'r2l',
    'teardrop': 'dos',
    'warezclient': 'r2l',
    'warezmaster': 'r2l',
    # Following attack types are only in the test set
    # Source: Feature Selection for Intrusion Detection System Using Ant Colony Optimization
    'mscan': 'probe',
    'apache2': 'dos',
    'processtable': 'dos',
    'snmpguess': 'u2r',
    'saint': 'probe',
    'mailbomb': 'dos',
    'snmpgetattack': 'r2l',
    'httptunnel': 'u2r',
    'named': 'r2l',
    'ps': 'u2r',
    'sendmail': 'r2l',
    'xterm': 'u2r',
    'xlock': 'r2l',
    'xsnoop': 'r2l',
    'udpstorm': 'dos',
    'sqlattack': 'u2r',
    'worm': 'u2r',
}

# NSL_KDD params
params.nsl_train_file = '../Datasets/NSL_KDD/KDDTrain+.txt'
# params.nsl_test_file = '../Datasets/NSL_KDD/KDDTest+.txt'
params.nsl_test_file = '../Datasets/NSL_KDD/KDDTest-21.txt'
params.nsl_map_to_five_classes = True

# IDS 2017 params
params.ids2017_small = False
params.ids2017_datasets_dir = '../Datasets/CIC_IDS_2017/MachineLearningCSV/MachineLearningCVE'
params.ids2017_files_list = [
                'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
                'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
                'Friday-WorkingHours-Morning.pcap_ISCX.csv',
                'Monday-WorkingHours.pcap_ISCX.csv',
                'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
                'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',   # Issue with flows file
                'Tuesday-WorkingHours.pcap_ISCX.csv',
                'Wednesday-workingHours.pcap_ISCX.csv'
                ]

params.ids2017_hist_num_bins = 10000

params.ids2017_flows_dir = '../Datasets/CIC_IDS_2017/GeneratedLabelledFlows/TrafficLabelling'
params.ids2017_flow_seqs_max_flow_seq_length = 100
params.ids2017_flow_seqs_max_flow_duration_secs = 3

# IDS 2018 params
params.ids2018_datasets_dir = '../Datasets/CIC_IDS_2018'
params.ids2018_files_list = [
                'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv',
                'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv',
                'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv',
                'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv_split_1.csv',
                'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv_split_2.csv',
                'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv_split_3.csv',
                'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv_split_4.csv',
                'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv',
                'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv',
                'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv',
                'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv',
                'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
                'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv'
                ]
params.ids2018_all_X_filename = 'ids2018_all_X.h5'
params.ids2018_all_y_filename = 'ids2018_all_y.h5'
params.ids2018_load_scaler_obj = True
params.ids2018_scaler_obj_path = params.output_dir + '/' + 'partial_scaler_obj.pickle'

params.ids2018_shrink_to_rate = 0.2


def initial_setup(output_dir, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info('Created directory: {}'.format(output_dir))

    # Setup logging
    log_filename = output_dir + '/' + 'run_log.log'

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename, 'w+'),
                  logging.StreamHandler()],
        level=logging.INFO
    )
    logging.info('Initialized logging. log_filename = {}'.format(log_filename))

    logging.info('Running script with following parameters\n{}'.format(pformat(params.__dict__)))


def prepare_kdd99_small_datasets(params):
    # Load dataset
    logging.info('Loading datasets')
    dataset_df = utility.load_datasets([params.kdd99_10_percent_dataset_file], header_row=None)
    logging.info('Follow is 5 rows of the loaded dataset')
    logging.info('\n{}'.format(dataset_df.head()))

    # Remove samples from classes with a very small no. of samples (cannot split with those classes)
    logging.info('Removing instances of rare classes')
    rare_classes = ['loadmodule.', 'ftp_write.', 'multihop.', 'phf.', 'perl.', 'spy.']
    dataset_df.drop(dataset_df[dataset_df[41].isin(rare_classes)].index, inplace=True)  # Inplace drop
    # dataset_df = dataset_df[~dataset_df[41].isin(rare_classes)]    # Also works

    X = dataset_df.iloc[:, :-1]  # All columns except the last
    y = dataset_df.iloc[:, -1]  # last column


    # Check class labels (counts, percentages of each class)
    label_counts, label_perc = utility.count_labels(y)

    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Label percentages below")
    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info("\n{}".format(label_perc))

    # One-hot encode the 3 non-numeric fields
    logging.info('One-hot-encoding columns 1, 2, 3')
    X = utility.one_hot_encode(X, columns=[1, 2, 3])

    # Split into 3 sets (train, validation, test)
    logging.info('Splitting datasets into 3 (train, validation, test)')
    splits = utility.split_dataset(X, y, [0.6, 0.2, 0.2])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits

    # Scaling
    logging.info('Scaling numeric features (columns 4 to 41)')
    # columns = list(range(4, 40 + 1))  # These are the numeric fields to be scaled: bug: will skip some important numeric features
    columns = list(range(0, X_train.shape[1]))  # These are the numeric fields to be scaled
    X_train_scaled, scaler = utility.scale_training_set(X_train, scale_type='standard', columns=columns)
    X_val_scaled = utility.scale_dataset(X_val, scaler=scaler, columns=columns)
    X_test_scaled = utility.scale_dataset(X_test, scaler=scaler, columns=columns)

    # Save data files in HDF format
    logging.info('Saving prepared datasets (train, val, test) to: {}'.format(params.output_dir))
    utility.write_to_hdf(X_train_scaled, params.output_dir + '/' + 'X_train.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_train, params.output_dir + '/' + 'y_train.h5', params.hdf_key, 5)

    utility.write_to_hdf(X_val_scaled, params.output_dir + '/' + 'X_val.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_val, params.output_dir + '/' + 'y_val.h5', params.hdf_key, 5)

    utility.write_to_hdf(X_test_scaled, params.output_dir + '/' + 'X_test.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_test, params.output_dir + '/' + 'y_test.h5', params.hdf_key, 5)

    logging.info('Saving complete')

    print_dataset_sizes(splits)


def prepare_kdd99_full_datasets(params):
    # Load dataset
    logging.info('Loading datasets')
    train_set_df = utility.load_datasets([params.kdd99_full_train_dataset_file], header_row=None)
    test_set_df = utility.load_datasets([params.kdd99_full_test_dataset_file], header_row=None)

    logging.info('Follow is 5 rows of the loaded training dataset')
    logging.info('\n{}'.format(train_set_df.head()))
    logging.info('Follow is 5 rows of the loaded test dataset')
    logging.info('\n{}'.format(test_set_df.head()))

    X_train = train_set_df.iloc[:, :-1]  # All columns except the last
    y_train = train_set_df.iloc[:, -1]  # last column

    X_test = test_set_df.iloc[:, :-1]  # All columns except the last
    y_test = test_set_df.iloc[:, -1]  # last column

    if params.kdd99_map_to_five_classes:
        num_classes = len(set(params.kdd_five_class_map.values()))
        assert num_classes == 4
    
        y_train.replace(params.kdd_five_class_map, inplace=True)
        y_test.replace(params.kdd_five_class_map, inplace=True)
    
        assert y_train.nunique() == num_classes + 1 # +1 for the normal class
        assert y_test.nunique() == num_classes + 1


    # Check class labels of train set (counts, percentages of each class)
    label_counts, label_perc = utility.count_labels(y_train)

    logging.info("Training set: Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Training set: Label percentages below")
    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info("\n{}".format(label_perc))

    # Check class labels of test set (counts, percentages of each class)
    label_counts, label_perc = utility.count_labels(y_test)

    logging.info("Test set: Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Test set: Label percentages below")
    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info("\n{}".format(label_perc))

    # One-hot encode the 3 non-numeric fields (do it on full dataset X_combined, so that all values are available to the encoder)
    logging.info('One-hot-encoding columns 1, 2, 3')
    X_train_len = X_train.shape[0]

    X_combined = pd.concat([X_train, X_test])
    X_combined_encoded = utility.one_hot_encode(X_combined, columns=[1, 2, 3])

    X_train = X_combined_encoded.iloc[:X_train_len, :]  # Separate again
    X_test = X_combined_encoded.iloc[X_train_len:, :]

    # Split train set into 2 sets (train, validation)
    logging.info('Splitting train set into 2 sets (train, validation)')
    splits = utility.split_dataset(X_train, y_train, [0.8, 0.2])
    (X_train, y_train), (X_val, y_val) = splits

    # Scaling
    logging.info('Scaling all features')
    # columns = list(range(4, 40 + 1))  # These are the numeric fields to be scaled: bug: will skip some important numeric features
    columns = list(range(0, X_train.shape[1]))  # These are the numeric fields to be scaled
    X_train_scaled, scaler = utility.scale_training_set(X_train, scale_type='standard', columns=columns)
    X_val_scaled = utility.scale_dataset(X_val, scaler=scaler, columns=columns)
    X_test_scaled = utility.scale_dataset(X_test, scaler=scaler, columns=columns)

    # Save data files in HDF format
    logging.info('Saving prepared datasets (train, val, test) to: {}'.format(params.output_dir))
    utility.write_to_hdf(X_train_scaled, params.output_dir + '/' + 'X_train.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_train, params.output_dir + '/' + 'y_train.h5', params.hdf_key, 5)

    utility.write_to_hdf(X_val_scaled, params.output_dir + '/' + 'X_val.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_val, params.output_dir + '/' + 'y_val.h5', params.hdf_key, 5)

    utility.write_to_hdf(X_test_scaled, params.output_dir + '/' + 'X_test.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_test, params.output_dir + '/' + 'y_test.h5', params.hdf_key, 5)

    logging.info('Saving complete')

    print_dataset_sizes(((X_train, y_train), (X_val, y_val), (X_test, y_test)))


def prepare_nsl_kdd_datasets(params):
    # Load dataset
    logging.info('Loading datasets')
    train_df = utility.load_datasets([params.nsl_train_file], header_row=None)
    test_df = utility.load_datasets([params.nsl_test_file], header_row=None)

    logging.info('Follow is 5 rows of the loaded train dataset')
    logging.info('\n{}'.format(train_df.head()))

    logging.info('Follow is 5 rows of the loaded test dataset')
    logging.info('\n{}'.format(test_df.head()))

    # Remove last column (difficulty level)
    logging.info('Removing last column (difficulty level)')
    train_df.drop(columns=train_df.columns[-1], inplace=True)
    test_df.drop(columns=test_df.columns[-1], inplace=True)

    X = train_df.iloc[:, :-1]  # All columns except the last
    y = train_df.iloc[:, -1]  # last column

    X_test = test_df.iloc[:, :-1]  # All columns except the last
    y_test = test_df.iloc[:, -1]  # last column

    # Check class labels in the train set (counts, percentages of each class)
    label_counts, label_perc = utility.count_labels(y_test)
    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    if params.nsl_map_to_five_classes:
        num_classes = len(set(params.kdd_five_class_map.values()))
        assert num_classes == 4
    
        y.replace(params.kdd_five_class_map, inplace=True)
        y_test.replace(params.kdd_five_class_map, inplace=True)
    
        assert y.nunique() == num_classes + 1 # +1 for the normal class
        assert y_test.nunique() == num_classes + 1

    # Check class labels (counts, percentages of each class)
    label_counts, label_perc = utility.count_labels(y)

    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Label percentages below")
    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info("\n{}".format(label_perc))

    # One-hot encode the 3 non-numeric fields (do it on full dataset X_combined, so that all values are available to the encoder)
    logging.info('One-hot-encoding columns 1, 2, 3')
    X_len = X.shape[0]

    X_combined = pd.concat([X, X_test])
    X_combined_encoded = utility.one_hot_encode(X_combined, columns=[1, 2, 3])

    X = X_combined_encoded.iloc[:X_len, :]  # Separate again
    X_test = X_combined_encoded.iloc[X_len:, :]

    # Split into 2 sets (train, validation)
    logging.info('Splitting training set into 2 (train, validation)')
    splits = utility.split_dataset(X, y, [0.8, 0.2])
    (X_train, y_train), (X_val, y_val) = splits

    logging.info('Removing unseen (new) classes in test set')
    rows_before = X_test.shape[0]
    X_test, y_test = utility.remove_new_classes(y_train, X_test, y_test)
    logging.info('Removed {} rows'.format(rows_before - X_test.shape[0]))

    # Scaling
    logging.info('Scaling all features')
    # columns = list(range(4, 40 + 1))  # These are the numeric fields to be scaled
    columns = list(range(0, X_train.shape[1]))  # These are the numeric fields to be scaled
    X_train_scaled, scaler = utility.scale_training_set(X_train, scale_type='standard', columns=columns)
    X_val_scaled = utility.scale_dataset(X_val, scaler=scaler, columns=columns)
    X_test_scaled = utility.scale_dataset(X_test, scaler=scaler, columns=columns)

    # Save data files in HDF format
    logging.info('Saving prepared datasets (train, val, test) to: {}'.format(params.output_dir))
    utility.write_to_hdf(X_train_scaled, params.output_dir + '/' + 'X_train.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_train, params.output_dir + '/' + 'y_train.h5', params.hdf_key, 5)

    utility.write_to_hdf(X_val_scaled, params.output_dir + '/' + 'X_val.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_val, params.output_dir + '/' + 'y_val.h5', params.hdf_key, 5)

    utility.write_to_hdf(X_test_scaled, params.output_dir + '/' + 'X_test.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_test, params.output_dir + '/' + 'y_test.h5', params.hdf_key, 5)

    logging.info('Saving complete')

    print_dataset_sizes(((X_train, y_train), (X_val, y_val), (X_test, y_test)))


def print_dataset_sizes(datasets):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets
    logging.info("No. of features = {}".format(X_train.shape[1]))
    logging.info("Training examples = {}".format(X_train.shape[0]))
    logging.info("Validation examples = {}".format(X_val.shape[0]))
    logging.info("Test examples = {}".format(X_test.shape[0]))


def prepare_ids2017_datasets(params):
    # Load data
    logging.info('Loading datasets')
    data_files_list = [params.ids2017_datasets_dir + '/' + filename for filename in params.ids2017_files_list]
    all_data = utility.load_datasets(data_files_list, header_row=0, strip_col_name_spaces=True)
    # utility.print_info(all_data)

    # Remove unicode values in class labels
    logging.info('Converting unicode labels to ascii')
    all_data['Label'] = all_data['Label'].apply(lambda x: x.encode('ascii', 'ignore').decode("utf-8"))
    all_data['Label'] = all_data['Label'].apply(lambda x: re.sub(' +', ' ', x)) # Remove double spaces

    # Following type conversion and casting (both) are necessary to convert the values in cols 14, 15 detected as objects
    # Otherwise, the training algorithm does not work as expected
    logging.info('Converting object type in columns 14, 15 to float64')
    all_data['Flow Bytes/s'] = all_data['Flow Bytes/s'].apply(lambda x: np.float64(x))
    all_data['Flow Packets/s'] = all_data['Flow Packets/s'].apply(lambda x: np.float64(x))
    all_data['Flow Bytes/s'] = all_data['Flow Bytes/s'].astype(np.float64)
    all_data['Flow Packets/s'] = all_data['Flow Packets/s'].astype(np.float64)

    # Remove some invalid values/ rows in the dataset
    # nan_counts = all_data.isna().sum()
    # logging.info(nan_counts)
    logging.info('Removing invalid values (inf, nan)')
    prev_rows = all_data.shape[0]
    all_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_data.dropna(inplace=True)  # Some rows (1358) have NaN values in the Flow Bytes/s column. Get rid of them
    logging.info('Removed no. of rows = {}'.format(prev_rows - all_data.shape[0]))

    # Remove samples from classes with a very small no. of samples (cannot split with those classes)
    logging.info('Removing instances of rare classes')
    rare_classes = ['Infiltration', 'Web Attack Sql Injection', 'Heartbleed']
    all_data.drop(all_data[all_data['Label'].isin(rare_classes)].index, inplace=True)  # Inplace drop

    # Check class labels
    label_counts, label_perc = utility.count_labels(all_data['Label'])
    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Label percentages below")
    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info("\n{}".format(label_perc))

    X = all_data.loc[:, all_data.columns != 'Label']  # All columns except the last
    y = all_data['Label']

    # Take only 8% as the small subset
    if params.ids2017_small:
        logging.info('Splitting datset into 2 (small subset, discarded)')
        splits = utility.split_dataset(X, y, [0.08, 0.92])
        (X, y), (discarded, discarded) = splits
        logging.info('Small subset no. of examples = {}'.format(X.shape[0]))

    # Split into 3 sets (train, validation, test)
    logging.info('Splitting training set into 3 (train, validation, test)')
    splits = utility.split_dataset(X, y, [0.6, 0.2, 0.2])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits

    # Scaling
    logging.info('Scaling features (assuming all are numeric)')
    columns = list(range(0, X_train.shape[1]))  # These are the numeric fields to be scaled
    X_train_scaled, scaler = utility.scale_training_set(X_train, scale_type='standard', columns=columns)
    X_val_scaled = utility.scale_dataset(X_val, scaler=scaler, columns=columns)
    X_test_scaled = utility.scale_dataset(X_test, scaler=scaler, columns=columns)

    # Save data files in HDF format
    logging.info('Saving prepared datasets (train, val, test) to: {}'.format(params.output_dir))

    utility.write_to_hdf(X_train_scaled, params.output_dir + '/' + 'X_train.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_train, params.output_dir + '/' + 'y_train.h5', params.hdf_key, 5)

    utility.write_to_hdf(X_val_scaled, params.output_dir + '/' + 'X_val.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_val, params.output_dir + '/' + 'y_val.h5', params.hdf_key, 5)

    utility.write_to_hdf(X_test_scaled, params.output_dir + '/' + 'X_test.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_test, params.output_dir + '/' + 'y_test.h5', params.hdf_key, 5)

    logging.info('Saving complete')

    print_dataset_sizes(splits)


def add_additional_items_to_dict(dict, extra_char):
    new_dict = {}
    for key, val in dict.items():
        new_key = key + extra_char
        new_dict[new_key] = val

    dict.update(new_dict)


def prepare_ids2018_datasets_stage_1(params):
    all_ys = {}

    for idx, filename in enumerate(params.ids2018_files_list):
        logging.info('Processing file # {}, filename = {}'.format(idx + 1, filename))

        filepath = params.ids2018_datasets_dir + '/' + filename
        data_df = utility.load_datasets([filepath], header_row=0)
        # data_df = utility.load_datasets([filename], header_row=0, columns_to_read=['Dst Port', 'Flow Duration'])

        logging.info('Sorting by Timestamp')
        data_df.sort_values(by=['Timestamp'], inplace=True)

        cols_to_remove = ['Protocol', 'Timestamp', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP']
        available_cols_to_remove = [col for col in cols_to_remove if col in data_df.columns]
        logging.info('Removing unnecessary columns: {}'.format(available_cols_to_remove))
        data_df.drop(available_cols_to_remove, axis=1, inplace=True)

        logging.info('Removing non-float rows')
        prev_rows = data_df.shape[0]
        utility.remove_non_float_rows(data_df, cols=['Dst Port'])
        logging.info('Removed no. of rows = {}'.format(prev_rows - data_df.shape[0]))

        skip_cols = ['Label']
        logging.info('Converting columns of type object to float')
        utility.convert_obj_cols_to_float(data_df, skip_cols)

        # Remove some invalid values/ rows in the dataset
        # nan_counts = data_df.isna().sum()
        # logging.info(nan_counts)
        logging.info('Removing invalid values (inf, nan)')
        prev_rows = data_df.shape[0]
        data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        data_df.dropna(inplace=True)  # Some rows (1358) have NaN values in the Flow Bytes/s column. Get rid of them
        logging.info('Removed no. of rows = {}'.format(prev_rows - data_df.shape[0]))

        X = data_df.loc[:, data_df.columns != 'Label']  # All columns except the last
        y = data_df['Label']

        all_ys[filename] = y

        X_filename = params.ids2018_datasets_dir + '/' + params.ids2018_all_X_filename
        y_filename = params.ids2018_datasets_dir + '/' + params.ids2018_all_y_filename

        if idx == 0:
            mode = 'w'
        else:
            mode = 'a'  # append

        utility.write_to_hdf(X, X_filename, key=filename, compression_level=5, mode=mode)
        utility.write_to_hdf(y, y_filename, key=filename, compression_level=5, mode=mode)

        logging.info('\n--------------- Processing file complete ---------------\n')

    all_ys_df = pd.concat(list(all_ys.values()))

    # Check class labels
    label_counts, label_perc = utility.count_labels(all_ys_df)
    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Label percentages below")
    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info("\n{}".format(label_perc))


def extract_scale_and_write(X_df, y_df, indexes_to_extract, scaler_ob, suffix_str, suffic_index):
    X_extracted = X_df.loc[indexes_to_extract, X_df.columns != 'Label']
    y_extracted = y_df.loc[indexes_to_extract]

    columns = list(range(0, X_extracted.shape[1]))
    X_scaled = utility.scale_dataset(X_extracted, scaler=scaler_ob, columns=columns)

    X_filename = params.output_dir + '/' + 'X_' + suffix_str + '_' + str(suffic_index) + '.h5'
    y_filename = params.output_dir + '/' + 'y_' + suffix_str + '_' + str(suffic_index) + '.h5'

    utility.write_to_hdf(X_scaled, X_filename, params.hdf_key, 5, format='table')
    utility.write_to_hdf(y_extracted, y_filename, params.hdf_key, 5, format='table')


def prepare_ids2018_datasets_stage_2(params):
    X_filename = params.ids2018_datasets_dir + '/' + params.ids2018_all_X_filename
    y_filename = params.ids2018_datasets_dir + '/' + params.ids2018_all_y_filename

    X_store = pd.HDFStore(X_filename, 'r')
    y_store = pd.HDFStore(y_filename, 'r')

    # print(y_store.keys())

    # Load all y dfs in the y_store, and create corresponding X_info dfs

    logging.info("Loading y dfs in the y_store, and creating corresponding X_info dfs")

    y_all_list = []
    X_info_all_list = []

    for key in y_store.keys():
        print('key = {}'.format(key))
        y_file = y_store[key]
        y_all_list.append(y_file)

        X_info = pd.DataFrame({'file_key': [key] * y_file.shape[0], 'index_in_file': y_file.index})
        X_info_all_list.append(X_info)

    y_all = pd.concat(y_all_list)
    X_info_all = pd.concat(X_info_all_list)

    # Check class labels
    label_counts, label_perc = utility.count_labels(y_all)
    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Label percentages below")
    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info("\n{}".format(label_perc))


    # Split into 3 sets (train, validation, test)
    logging.info('Splitting dataset set into 3 (train, validation, test)')
    splits = utility.split_dataset(X_info_all, y_all, [0.6, 0.2, 0.2])
    (X_info_train, y_train), (X_info_val, y_val), (X_info_test, y_test) = splits

    if params.ids2018_load_scaler_obj:
        logging.info('Loading partial scaler. path = {}'.format(params.ids2018_scaler_obj_path))
        scaler_obj = utility.load_obj_from_disk(params.ids2018_scaler_obj_path)
    else:
        # Extract training examples at each key and build partial scaler
        logging.info('Extracting training examples at each key and building partial scaler')

        scaler_obj = None
        for i, key in enumerate(X_store.keys()):
            print('key = {}'.format(key))
            X_in_file = X_store[key]

            train_indexes_in_file = X_info_train.loc[X_info_train['file_key'] == key, 'index_in_file']
            X_extracted = X_in_file.loc[train_indexes_in_file, X_in_file.columns != 'Label']

            columns = list(range(0, X_extracted.shape[1]))
            scaler_obj = utility.partial_scaler(X_extracted, scale_type='standard', columns=columns, scaler_obj=scaler_obj)

        utility.save_obj_to_disk(scaler_obj, params.ids2018_scaler_obj_path)
        logging.info("Partial scaler parameters below. \n{}".format(scaler_obj.get_params()))

    for i, key in enumerate(X_store.keys()):
        print('key = {}'.format(key))
        X_in_file = X_store[key]
        y_in_file = y_store[key]

        train_indexes_in_file = X_info_train.loc[X_info_train['file_key'] == key, 'index_in_file']
        val_indexes_in_file = X_info_val.loc[X_info_val['file_key'] == key, 'index_in_file']
        test_indexes_in_file = X_info_test.loc[X_info_test['file_key'] == key, 'index_in_file']

        extract_scale_and_write(X_in_file, y_in_file, train_indexes_in_file, scaler_obj, 'train', i + 1)
        extract_scale_and_write(X_in_file, y_in_file, val_indexes_in_file, scaler_obj, 'val', i + 1)
        extract_scale_and_write(X_in_file, y_in_file, test_indexes_in_file, scaler_obj, 'test', i + 1)


def prepare_ids2018_shrink_dataset(params):
    X_filename = params.ids2018_datasets_dir + '/' + params.ids2018_all_X_filename
    y_filename = params.ids2018_datasets_dir + '/' + params.ids2018_all_y_filename

    X_store = pd.HDFStore(X_filename, 'r')
    y_store = pd.HDFStore(y_filename, 'r')

    # Load all y dfs in the y_store, and create corresponding X_info dfs

    logging.info("Loading y dfs in the y_store, and creating corresponding X_info dfs")

    y_all_list = []
    X_info_all_list = []

    for key in y_store.keys():
        print('key = {}'.format(key))
        y_file = y_store[key]
        y_all_list.append(y_file)

        X_info = pd.DataFrame({'file_key': [key] * y_file.shape[0], 'index_in_file': y_file.index})
        X_info_all_list.append(X_info)

    y_all = pd.concat(y_all_list)
    X_info_all = pd.concat(X_info_all_list)

    # ------------

    logging.info('Shrinking dataset')

    # X_info_shrunk, y_shrunk = shrink_dataset(X_info_all, y_all, shrink_to_rate)

    shrink_to_rate = params.ids2018_shrink_to_rate
    jump = math.ceil(1 / shrink_to_rate)

    assert X_info_all.shape[0] == y_all.shape[0]

    X_info_shrunk = X_info_all.iloc[::jump, :]
    y_shrunk = y_all.iloc[::jump]

    logging.info('No. of rows after shrinking dataset: {}'.format(X_info_shrunk.shape[0]))

    # Extract the final records into two dfs X, y

    X_shrunk_dfs = []
    y_shrunk_dfs = []

    for key in X_store.keys():
        print('key = {}'.format(key))
        X_in_file = X_store[key]
        y_in_file = y_store[key]

        indexes_in_file = X_info_shrunk.loc[X_info_shrunk['file_key'] == key, 'index_in_file']

        X_extracted = X_in_file.loc[indexes_in_file, X_in_file.columns != 'Label']
        y_extracted = y_in_file.loc[indexes_in_file]

        X_shrunk_dfs.append(X_extracted)
        y_shrunk_dfs.append(y_extracted)

    X_shrunk_df = pd.concat(X_shrunk_dfs)
    y_shrunk_df = pd.concat(y_shrunk_dfs)

    # Check class labels
    label_counts, label_perc = utility.count_labels(y_shrunk_df)
    logging.info("Label counts below")
    logging.info("\n{}".format(label_counts))

    logging.info("Label percentages below")
    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info("\n{}".format(label_perc))

    # ------------

    # Split into 3 sets (train, validation, test)
    logging.info('Splitting dataset set into 3 (train, validation, test)')
    splits = utility.split_dataset(X_shrunk_df, y_shrunk_df, [0.6, 0.2, 0.2])
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits

    # Scaling
    logging.info('Scaling features (assuming all are numeric)')
    columns = list(range(0, X_train.shape[1]))  # These are the numeric fields to be scaled
    X_train_scaled, scaler = utility.scale_training_set(X_train, scale_type='standard', columns=columns)
    X_val_scaled = utility.scale_dataset(X_val, scaler=scaler, columns=columns)
    X_test_scaled = utility.scale_dataset(X_test, scaler=scaler, columns=columns)

    # Save data files in HDF format
    logging.info('Saving prepared datasets (train, val, test) to: {}'.format(params.output_dir))

    utility.write_to_hdf(X_train_scaled, params.output_dir + '/' + 'X_train.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_train, params.output_dir + '/' + 'y_train.h5', params.hdf_key, 5)

    utility.write_to_hdf(X_val_scaled, params.output_dir + '/' + 'X_val.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_val, params.output_dir + '/' + 'y_val.h5', params.hdf_key, 5)

    utility.write_to_hdf(X_test_scaled, params.output_dir + '/' + 'X_test.h5', params.hdf_key, 5)
    utility.write_to_hdf(y_test, params.output_dir + '/' + 'y_test.h5', params.hdf_key, 5)

    logging.info('Saving complete')

    print_dataset_sizes(splits)


def shrink_dataset(X_info_df, y_df, shrink_to_rate):
    jump = math.ceil(1/shrink_to_rate)

    assert X_info_df.shape[0] == y_df.shape[0]

    X_info_shrunk = X_info_df.iloc[::jump, :]
    y_shrunk = y_df.iloc[::jump, :]

    logging.info('No. of rows after shrinking dataset: {}'.format(X_info_shrunk.shape[0]))


def main():
    initial_setup(params.output_dir, params)
    add_additional_items_to_dict(params.kdd_five_class_map, '.')

    # --------------------------------------

    # prepare_kdd99_small_datasets(params)

    # prepare_kdd99_full_datasets(params)

    prepare_nsl_kdd_datasets(params)

    # prepare_ids2017_datasets(params)  # Small subset vs. full is controlled by config flag


    # Following 3 are for preparing the IDS 2018 dataset (20% subset)
    # prepare_ids2018_datasets_stage_1(params)
    # prepare_ids2018_datasets_stage_2(params)
    # prepare_ids2018_shrink_dataset(params)

    logging.info('Data preparation complete')


if __name__ == "__main__":
    main()

