import time, logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit # Does not seem to work with multiple classes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import re, pickle, math


def read_csv(filename, header_row=0, dtypes=None, columns_to_read=None):
    t0 = time.time()
    logging.info('Reading CSV dataset {}'.format(filename))

    if columns_to_read is not None:
        dataset_df = pd.read_csv(filename, header=header_row, dtype=dtypes, usecols=columns_to_read)
    else:
        dataset_df = pd.read_csv(filename, header=header_row, dtype=dtypes)

    logging.info('Reading complete. time_to_read={:.2f} sec'.format(time.time() - t0))

    return dataset_df   # This is a Pandas DataFrame


def read_hdf(filename, key):
    t0 = time.time()
    logging.info('Reading HDF dataset {}'.format(filename))

    dataset_df = pd.read_hdf(filename, key=key)

    logging.info('Reading complete. time_to_read={:.2f} seconds'.format(time.time() - t0))

    return dataset_df  # This is a Pandas DataFrame


def write_to_hdf(df, filename, key, compression_level, mode='a', format='fixed'):
    logging.info('Writing dataset to HDF5 format. filename={}'.format(filename))
    t0 = time.time()

    df.to_hdf(filename, key=key, mode=mode, complevel=compression_level, complib='zlib', format=format)

    logging.info('Writing complete. time_to_write={}'.format(time.time() - t0))


def write_to_csv(df, filename, write_index=False):
    logging.info('Writing dataset to CSV file. filename={}'.format(filename))
    t0 = time.time()

    df.to_csv(filename, index=write_index)

    logging.info('Writing complete. time_to_write={}'.format(time.time() - t0))


def load_datasets(files_list, header_row=0, strip_col_name_spaces=False, dtypes=None, columns_to_read=None):
    def strip_whitespaces(str):
        return str.strip()

    logging.info('Loading datasets in files')

    dfs = []
    for filename in files_list:
        df = read_csv(filename, header_row, dtypes=dtypes, columns_to_read=columns_to_read)
        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)

    if strip_col_name_spaces:
        all_data.rename(columns=strip_whitespaces, inplace=True)

    logging.info('Loading datasets complete')
    return all_data


def print_info(df):
    logging.info("Dataset shape = {}".format(df.shape))
    df.info(memory_usage='deep')    # Some info about the dataset (memory usage etc.)

    pd.set_option("display.precision", 4)  # Show more decimals
    logging.info(df.head())


def count_labels(label_col):
    label_counts = label_col.value_counts(dropna=False)
    total_count = sum(label_counts)

    logging.info("Total count = {}".format(total_count))

    label_percentages = label_counts / total_count

    return label_counts, label_percentages


def print_dataset_label_info(label_col, dataset_name):
    logging.info("\n------------ Label percentages of {} ------------\n".format(dataset_name))
    count_labels(label_col)
    logging.info("\n-------------------------------------------------\n")


# Return 3 (X, y) pairs for training, validation, test
# StratifiedShuffleSplit does not seem to work with multiple classes
def split_dataset_deprecated(dataset):
    logging.info('Preparing dataset - split into training, validation, test - 60%, 20%, 20%')

    X = dataset.loc[:, dataset.columns != 'Label']    # All columns except the last
    y = dataset['Label']    # Last column

    # First split (60% training, 40% for the rest (validation + test))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.4)
    count = 0
    for train_index, rest_index in sss2.split(X, y):
        count += 1
        X_train, X_rest = X.loc[train_index, :], X.loc[rest_index, :]
        y_train, y_rest = y[train_index], y[rest_index]

    assert count == 1

    # Second split (split the rest of the data to validation and test)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
    count = 0
    for val_index, test_index in sss2.split(X_rest, y_rest):
        count += 1
        X_val, X_test = X.loc[val_index, :], X.loc[test_index, :]
        y_val, y_test = y[val_index], y[test_index]

    assert count == 1

    logging.info('Dataset sizes: original_size={}, train_size={}, val_size={}, test_size={}'
          .format(len(X), len(X_train), len(X_val), len(X_test)))

    logging.info('Prepared dataset')

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def split_dataset(X, y, split_rates, random_seed=None):
    assert sum(split_rates) == 1

    X_2 = X
    y_2 = y
    result_splits = []

    for i, split in enumerate(split_rates[:-1]):  # Must not split at the last element
        remain_rate_sum = sum(split_rates[i:])
        remain_rate = 1 - (split / remain_rate_sum)
        # print("i={}, split={}, remain_rate={}, remain_rate_sum={}".format(i, split, remain_rate, remain_rate_sum))

        # Split into 2 parts, X_1 and X_2
        X_1, X_2, y_1, y_2 = train_test_split(X_2, y_2, stratify=y_2, test_size=remain_rate, random_state=random_seed)
        result_splits.append((X_1, y_1))

    result_splits.append((X_2, y_2))    # Final remaining part

    return result_splits


def convert_labels_to_numbers(labels_list, label_col, start_with_one=True):
    offset = 0
    if start_with_one:
        offset = 1

    label_to_num_map = dict([(y, x + offset) for x, y in enumerate(sorted(set(labels_list)))])

    label_col.replace(label_to_num_map, inplace=True)

    return label_to_num_map


def one_hot_encode(X, columns):
    encoded_df = pd.get_dummies(X, columns=columns) # This deletes the original cols
    # encoded_df.to_csv('encoded_df.csv')   # Debugging
    # encoded_df.drop(columns=columns, inplace=True)
    # out = pd.concat([X, encoded_df], axis=1)
    return encoded_df


def encode_labels(y, encoder=None):
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(y)

    encoded_Y = encoder.transform(y)

    n_classes = encoder.classes_.shape[0]

    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y, num_classes=n_classes)

    return (encoder, dummy_y)


def scale_training_set(X_train, scale_type, columns):
    # X = X_train[columns]
    X = X_train.iloc[:, columns]
    if scale_type == 'standard':
        scaler_obj = StandardScaler()
    elif scale_type == 'min_max':
        scaler_obj = MinMaxScaler(feature_range=(0, 1))

    logging.info('Created new scaler: {}'.format(scaler_obj))

    scaler_obj.fit(X)
    scaled_X = scaler_obj.transform(X)
    scaled_X_df = pd.DataFrame(scaled_X, columns=X.columns)

    # X_train_remain = X_train.drop(columns=orig_columns) # Drop original columns

    # X_train_remain = X_train.drop(columns, axis='index') # Drop original columns

    # orig_columns = X_train.columns[columns].tolist()
    orig_columns = X_train.columns[columns]
    X_train_remain = X_train.drop(orig_columns, axis='columns') # Drop original columns

    X_train_remain.reset_index(drop=True, inplace=True)
    scaled_X_df.reset_index(drop=True, inplace=True)

    X_train_scaled = pd.concat([X_train_remain, scaled_X_df], axis=1)

    return X_train_scaled, scaler_obj


def partial_scaler(X, scale_type, columns, scaler_obj=None):
    if scaler_obj is None:
        if scale_type == 'standard':
            scaler_obj = StandardScaler()
        elif scale_type == 'min_max':
            scaler_obj = MinMaxScaler(feature_range=(0, 1))

    X = X.iloc[:, columns]

    scaler_obj.partial_fit(X)

    return scaler_obj


def scale_dataset(X_orig, scaler, columns):
    # X = X_orig[columns]
    X = X_orig.iloc[:, columns]

    scaled_X = scaler.transform(X)
    scaled_X_df = pd.DataFrame(scaled_X, columns=X.columns)

    # X_remain = X_orig.drop(columns=columns)  # Drop original columns

    orig_columns = X_orig.columns[columns]
    X_remain = X_orig.drop(orig_columns, axis='columns')  # Drop original columns

    X_remain.reset_index(drop=True, inplace=True)
    scaled_X_df.reset_index(drop=True, inplace=True)

    X_scaled = pd.concat([X_remain, scaled_X_df], axis=1)

    return X_scaled


def convert(val):
    constructors = [int, float, str]
    for c in constructors:
        try:
            return c(val)
        except ValueError:
            pass


def save_training_history(history, save_dir):
    filename = save_dir + '/' + 'training_error_history' + '.csv'

    train_loss = np.array(history.history['loss'])
    n = train_loss.shape[0]
    epoch = np.arange(1, n+1)
    val_loss = np.zeros(n)
    val_f1 = np.zeros(n)

    if 'val_loss' in history.history:
        val_loss = history.history['val_loss']

    if 'val_f1' in history.history:
        val_f1 = history.history['val_f1']

    history_df = pd.DataFrame({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_f1': val_f1})

    history_df.to_csv(filename, index=None)
    logging.info('Training history saved to: {}'.format(filename))


def plot_training_history(history, save_dir=None):
    plt.figure()
    plt.title("Training history")

    plt.plot(history.history['loss'], label='training_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='validation_loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')

    if save_dir:
        filename = save_dir + '/' + 'training_error_history' + '.png'
        plt.savefig(filename, bbox_inches='tight')
        logging.info('Plot saved to: {}'.format(filename))

    if 'val_f1' in history.history:
        plt.figure()
        plt.plot(history.history['val_f1'], label='validation_f1_score')

        plt.xlabel("Epoch")
        plt.ylabel("F1-score")
        plt.legend(loc='upper right')

        if save_dir:
            filename = save_dir + '/' + 'training_f1_history' + '.png'
            plt.savefig(filename, bbox_inches='tight')
            logging.info('Plot saved to: {}'.format(filename))


def evaluate_on_conf_mat(labels, conf_mat):
    FP = conf_mat.sum(axis=0) - np.diag(conf_mat)
    FN = conf_mat.sum(axis=1) - np.diag(conf_mat)
    TP = np.diag(conf_mat)
    TN = conf_mat.sum() - (FP + FN + TP)

    # Per class metrics
    TPR = TP / (TP + FN)    # Sensitivity, hit rate, recall, or true positive rate
    TNR = TN / (TN + FP)    # Specificity or true negative rate
    PPV = TP / (TP + FP)    # Precision or positive predictive value
    NPV = TN / (TN + FN)    # Negative predictive value
    FPR = FP / (FP + TN)    # Fall out or false positive rate
    FNR = FN / (TP + FN)    # False negative rate or miss rate
    FDR = FP / (TP + FP)    # False discovery rate
    ACC = (TP + TN) / (TP + FP + FN + TN)   # Accuracy
    F1 = (2 * PPV * TPR) / (PPV + TPR)  # F1 score
    support = TP + FN

    # Micro averages
    micro_avgs = {}
    micro_avgs['TPR'] = np.sum(TP) / np.sum(TP + FN)
    micro_avgs['TNR'] = np.sum(TN) / np.sum(TN + FP)
    micro_avgs['PPV'] = np.sum(TP) / np.sum(TP + FP)
    micro_avgs['NPV'] = np.sum(TN) / np.sum(TN + FN)
    micro_avgs['FPR'] = np.sum(FP) / np.sum(FP + TN)
    micro_avgs['FNR'] = np.sum(FN) / np.sum(TP + FN)
    micro_avgs['FDR'] = np.sum(FP) / np.sum(TP + FP)

    # micro_avgs['ACC'] = np.sum(TP + TN) / np.sum(TP + FP + FN + TN)
    # This is the correct way to compute micro-accuracy going by definition of micro-average
    # However, micro-accuracy is not practically useful, and therefore, overall accuracy is computed here
    micro_avgs['ACC'] = np.diag(conf_mat).sum() / conf_mat.sum()

    micro_PPV = micro_avgs['PPV']
    micro_TPR = micro_avgs['TPR']
    micro_avgs['F1'] = (2 * micro_PPV * micro_TPR) / (micro_PPV + micro_TPR)

    metrics = {'TPR': TPR, 'TNR': TNR, 'PPV': PPV, 'NPV': NPV,
               'FPR': FPR, 'FNR': FNR, 'FDR': FDR, 'ACC': ACC,
               'F1': F1}

    avg_metrics = {'avg_labels': ['Micro avg', 'Macro avg', 'Weighted avg']}

    for key, val in metrics.items():
        val[np.isnan(val)] = 0
        micro_avg = micro_avgs[key]
        macro_avg = np.mean(val)
        weighted_avg = np.sum(val * support) / np.sum(support)
        avg_metrics[key] = np.array([micro_avg, macro_avg, weighted_avg])

    metrics['support'] = support
    metrics['labels'] = labels
    metrics['conf_mat'] = conf_mat

    return metrics, avg_metrics


def get_conf_mat(y_pred, y_true):
    conf_mat = confusion_matrix(y_true, y_pred)

    report = classification_report(y_true, y_pred, output_dict=True)
    labels = [key for (key, val) in report.items()]
    labels = labels[0: len(conf_mat)]  # This list contains 3 additional items at the end; remove them

    return labels, conf_mat


def convert_to_binary_conf_mat(labels, conf_mat, normal_label, attack_label):
    regex = '^.*' + normal_label + '.*$'
    normal_idx = [i for i, label in enumerate(labels) if re.search(regex, label, re.IGNORECASE)]

    # Verify that at least one of the "normal" class exists
    assert len(normal_idx) == 1   # If this assertion fails, check that your normal class label is correct
    normal_idx = normal_idx[0]

    norm_norm = conf_mat[normal_idx, normal_idx]
    norm_attack = np.sum(conf_mat[normal_idx, :]) - norm_norm
    attack_norm = np.sum(conf_mat[:, normal_idx]) - norm_norm
    attack_attack = np.sum(conf_mat) - norm_norm - norm_attack - attack_norm

    binary_labels = [attack_label, normal_label]
    binary_conf_mat = np.array([[attack_attack, attack_norm], [norm_attack, norm_norm]])

    return binary_labels, binary_conf_mat


def print_evaluation_report(y_pred, y_true, dataset_name):
    report_str = classification_report(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info('Dataset: {}. Classification report below'.format(dataset_name))

    logging.info('\n{}'.format(report_str))
    logging.info('Overall accuracy (micro avg): {}'.format(accuracy))


def print_avg_metrics(dataset_name, avg_metrics):
    df_avg_results = pd.DataFrame({'average type': avg_metrics['avg_labels'], 'avg accuracy': avg_metrics['ACC'],
                                   'avg precision': avg_metrics['PPV'],
                                   'avg detection rate (recall)': avg_metrics['TPR'],
                                   'avg false alarm rate': avg_metrics['FPR'],
                                   'avg false negative rate': avg_metrics['FNR'],
                                   'avg f1': avg_metrics['F1']})

    logging.info('Average metrics for {} dataset below\n{}'.format(dataset_name, df_avg_results.to_string()))


def write_to_excel(excel_writer, sheet, metrics, avg_metrics, start_row=0):
    labels = metrics['labels']
    accuracy = metrics['ACC']
    precision = metrics['PPV']
    detection_rate = metrics['TPR']
    false_alarm_rate = metrics['FPR']
    false_negative_rate = metrics['FNR']
    f1 = metrics['F1']
    conf_mat = metrics['conf_mat']

    curr_row = start_row


    # Write per-class metrics

    df_class_results = pd.DataFrame({'class': labels, 'accuracy': accuracy, 'precision': precision,
                                     'detection rate (recall)': detection_rate, 'false alarm rate': false_alarm_rate,
                                     'false negative rate': false_negative_rate, 'f1': f1})
    df_class_results.to_excel(excel_writer, sheet_name=sheet, startrow=curr_row, index=False)
    curr_row = curr_row + len(df_class_results) + 2


    # Write average metrics (overall)

    df_avg_results = pd.DataFrame({'average type': avg_metrics['avg_labels'], 'avg accuracy': avg_metrics['ACC'],
                                   'avg precision': avg_metrics['PPV'], 'avg detection rate (recall)': avg_metrics['TPR'],
                                   'avg false alarm rate': avg_metrics['FPR'], 'avg false negative rate': avg_metrics['FNR'],
                                   'avg f1': avg_metrics['F1']})
    df_avg_results.to_excel(excel_writer, sheet_name=sheet, startrow=curr_row, index=False)
    curr_row = curr_row + len(df_avg_results) + 2


    # Write confusion matrix

    multi_index_row = [['True Class' for i in range(0, len(labels))], labels]
    multi_index_col = [['Predicted Class' for i in range(0, len(labels))], labels]

    df_conf_mat = pd.DataFrame(conf_mat, index=multi_index_row, columns=multi_index_col)
    df_conf_mat.to_excel(excel_writer, sheet_name=sheet, startrow=curr_row)
    curr_row = curr_row + len(df_conf_mat) + 6

    return curr_row


def convert_to_binary_classes(y, normal_str):
    y = pd.DataFrame(y)
    regex = "^(.(?<!" + normal_str + "))*?$"
    y_ret = y.replace(to_replace=regex, value='attack.', regex=True)
    return y_ret


def remove_new_classes(y1, X2, y2):
    y1.reset_index(drop=True, inplace=True)
    X2.reset_index(drop=True, inplace=True)
    y2.reset_index(drop=True, inplace=True)

    unique_y1_labels = y1.unique()
    correct_label_locs = y2.isin(unique_y1_labels)

    new_y2 = y2[correct_label_locs]
    new_X2 = X2[correct_label_locs]

    return new_X2, new_y2


def save_obj_to_disk(obj, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)


def load_obj_from_disk(filepath):
    with open(filepath, 'rb') as file:
        obj = pickle.load(file)

    return obj


def plot_histogram(elements, num_bins, color, xlabel, title):
    plt.figure()
    n, bins, patches = plt.hist(elements, num_bins, facecolor=color, alpha=0.5)
    plt.xlabel(xlabel)
    plt.title(title)


def extract_flow_sequences(X, y, max_seq_length, max_seq_duration_secs):
    end = 0
    num_seqs = math.floor(X.shape[0] / max_seq_length)

    # shape = (batch_size, time_steps, features)
    all_seqs_X = np.zeros((num_seqs, max_seq_length, X.shape[1]))
    all_seqs_y = np.empty(shape=(num_seqs, max_seq_length, 1), dtype='object')

    # all_seqs_X = []
    # all_seqs_y = []

    for iter in range(num_seqs):
        start = end
        end = start + max_seq_length

        seq_X = X.iloc[start:end, :]
        seq_y = y.iloc[start:end]


        all_seqs_X[iter] = seq_X.values
        seq_y = np.array(seq_y.values).reshape(seq_y.shape[0], 1)
        all_seqs_y[iter] = seq_y

        # all_seqs_X.append(seq_X)
        # all_seqs_y.append(seq_y)

        # completion = iter * 100 / num_seqs
        # if num_seqs / iter % 10 == 0:

    return (all_seqs_X, all_seqs_y)


def remove_non_float_rows(df, cols):
    for col_name in df.columns:
        if col_name in cols and df[col_name].dtype == np.object:
            num_rows = df.shape[0]
            # print('num_rows = {}'.format(num_rows))
            batch_size = 1000
            num_batches = int(np.ceil(num_rows / batch_size))
            i = 0
            dummy_list = [] # To prevent the apply() call from getting optimized out
            # dummy_floats = []
            error_indexes = []  # List of indexes with non-float values
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, num_rows)
                try:
                    curr_batch = df.loc[df.index[start:end], col_name].apply(lambda x: np.float64(x))
                    dummy_list.append(curr_batch.shape)
                    # print("Batch done: i = {}".format(i))
                    i = i + 1
                except ValueError as error:
                    print('Conversion error in column: {}, in rows {}:{}'.format(col_name, start, end))
                    error_batch = df.loc[df.index[start:end], col_name]
                    for index, element in error_batch.iteritems():
                        try:
                            f = np.float64(element)
                        except ValueError as error:
                            error_indexes.append(index)

            print('column = {}, error indexes = {}'.format(col_name, error_indexes))
            # print('dummy_floats = {}'. format(dummy_floats))
            df.drop(error_indexes, inplace=True)  # Remove the non-float rows


def convert_obj_cols_to_float(df, skip_cols):
    for col_name in df.columns:
        if col_name in skip_cols:
            continue
        if df[col_name].dtype == np.object:
            try:
                df.loc[:, col_name] = df[col_name].apply(lambda x: np.float64(x))   # .loc[] avoids making copies of df cols
            except ValueError as error:
                print('Conversion error in column: {}'.format(col_name))
                assert False


def split_csv_file(filename, num_splits, output_dir):
    data_df = load_datasets([filename], header_row=0)
    num_rows = data_df.shape[0]
    split_size = int(np.ceil(num_rows / num_splits))

    for idx in range(num_splits):
        start = idx * split_size
        end = min(start + split_size, num_rows)

        split = data_df.iloc[start:end]

        split_filename = output_dir + '/' + filename + '_split_' + str(idx + 1) + '.csv'
        write_to_csv(split, split_filename)

