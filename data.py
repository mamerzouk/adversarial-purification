import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_unsw(train_path="./UNSW-NB15/UNSW_NB15_testing-set.csv", test_path="./UNSW-NB15/UNSW_NB15_training-set.csv"):
    # import the dataset into pandas DataFrames
    df_training = pd.read_csv(train_path)
    df_testing = pd.read_csv(test_path)

    # stack the training and testing sets
    df_data = pd.concat([df_training, df_testing], axis=0)

    # remove the columns 'id' and 'attack_cat'
    df_data.drop('id', inplace=True, axis=1)
    df_data.drop('attack_cat', inplace=True, axis=1)

    # 'is_ftp_login' should be a binary feature, we remove the instances that hold the values 2 and 4
    df_data = df_data[df_data['is_ftp_login'] != 2]
    df_data = df_data[df_data['is_ftp_login'] != 4]

    categorical_features = ['state', 'service', 'proto']
    df_data = pd.get_dummies(df_data, columns=categorical_features, prefix=categorical_features, prefix_sep=":")
    # move the labels back to the last column in lowercase
    df_data['labels'] = df_data.pop('label')

    # Min-Max normalization on the non-binary features
    # the min and max values are computed on the training set
    continuous_features = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst']
    min = df_data[:df_training.shape[0]][continuous_features].min()
    max = df_data[:df_training.shape[0]][continuous_features].max()
    df_data[continuous_features] = (df_data[continuous_features] - min) / (max - min)

    # Normalize numerical columns:
    scaler = MinMaxScaler()

    means = df_data[:df_training.shape[0]][continuous_features].mean() # Store means and std before normalization
    stds = df_data[:df_training.shape[0]][continuous_features].std()
    scaler.fit(df_data[:df_training.shape[0]][continuous_features]) # Fit only on the training data to avoid bias
    scaled_array = scaler.transform(df_data[continuous_features])
    df_data[continuous_features] = scaled_array

    # split training and testing sets
    formated_train = df_data[:df_training.shape[0]]
    formated_test = df_data[df_training.shape[0]:]

    x_train = formated_train.values[:,:-1]
    y_train = formated_train.values[:,-1]
    x_test = formated_test.values[:,:-1]
    y_test = formated_test.values[:,-1]

    return x_train.astype(float), y_train.astype(float), x_test.astype(float), y_test.astype(float)


def preprocess_kdd(train_path='./NSL-KDD/KDDTrain+.txt', test_path='./NSL-KDD/KDDTest+.txt'):    
    # Import the training and testing datasets from .CSV to Pandas DataFrames
    features =  ["duration","protocol_type","service","flag","src_bytes",
            "dst_bytes","land_f","wrong_fragment","urgent","hot","num_failed_logins",
            "logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
            "is_host_login","is_guest_login","count","srv_count","serror_rate",
            "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels","dificulty"]
    df_training = pd.read_csv(train_path, names=features)
    df_testing = pd.read_csv(test_path, names=features)
    # Stack the training and test sets
    df_data = pd.concat([df_training, df_testing], axis=0)

    # Drop the last column (difficulty)
    df_data.drop('dificulty', inplace=True, axis=1)
    # Drop the 19th column wich is full of 0, so has std=0. which causes issues for the normalization
    df_data.drop('num_outbound_cmds', inplace=True, axis=1)

    # Transform the nominal attribute "Attack type" into binary (0 : normal / 1 : attack)
    df_data['labels'] = (df_data['labels'] != 'normal').astype('int64')


    # 0-1 values for the 'su_attempted' column
    df_data['su_attempted'] = df_data['su_attempted'].replace(2.0, 0.0)

    # Separate categorical and numerical data
    categorical = df_data[['protocol_type', 'service', 'flag', 'labels', 'su_attempted', 'is_guest_login', 'is_host_login']]
    df_data = df_data.drop(['protocol_type', 'service', 'flag', 'labels', 'su_attempted', 'is_guest_login', 'is_host_login'], axis=1)
    col_num = df_data.columns
    idx_num = df_data.index

    scaler = MinMaxScaler()

    scaler.fit(df_data[0:df_training.shape[0]]) # Fit only on the training data to avoid bias
    means = df_data[0:df_training.shape[0]].mean() # Store means and std before normalization
    stds = df_data[0:df_training.shape[0]].std()
    scaled_array = scaler.transform(df_data)
    

    df_data = pd.DataFrame(scaled_array, columns = col_num, index = idx_num)

    # Add one-hot encoding for the categorical data
    df_data = pd.concat([df_data, categorical[['su_attempted', 'is_guest_login', 'is_host_login']]], axis=1)
    df_data = pd.concat([df_data, pd.get_dummies(categorical['protocol_type'])], axis=1)
    df_data = pd.concat([df_data, pd.get_dummies(categorical['service'])], axis=1)
    df_data = pd.concat([df_data, pd.get_dummies(categorical['flag'])], axis=1)

    # Add the label column, unmodified
    df_data = pd.concat([df_data, categorical['labels']], axis=1)

    # Separate training and testing sets
    formated_train = df_data.iloc[:df_training.shape[0]]
    formated_test = df_data.iloc[df_training.shape[0]:]

    x_train = formated_train.values[:,:-1]
    y_train = formated_train.values[:,-1]
    x_test = formated_test.values[:,:-1]
    y_test = formated_test.values[:,-1]

    return x_train.astype(float), y_train.astype(float), x_test.astype(float), y_test.astype(float)