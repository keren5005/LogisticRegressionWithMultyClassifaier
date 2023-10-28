import random
import pandas as pd
from OrdinalEncoder import  OrdinalEncoder
from MinMaxScalerClass import MinMaxScaler

#This function takes a confusion matrix m as input and prints it in a formatted way
def print_confusion_matrix(m):
    tp = m[0][0]
    fn = m[0][1]
    fp = m[1][0]
    tn = m[1][1]
    print("--------------------")
    print(f"{tp} | {fp}")
    print(f"{fn} | {tn}")
    print("--------------------")

#This function loads the hsbdemo dataset from a CSV file
# and performs scaling and ordinal encoding on selected columns
def load_and_scale(fname):
    raw = pd.read_csv('hsbdemo.csv')
    columns = ['female', 'ses', 'schtyp', 'prog',
               'read', 'write', 'math', 'science',
               'socst', 'honors', 'awards'
               ]
    ordinals = ['female', 'ses', 'schtyp', 'prog', 'honors', 'awards']

    # columns = ['ses', 'prog', 'write']
    # ordinals = ['ses', 'prog']

    ordinal_encoders = {}
    scalers = {}

    df = pd.DataFrame()
    for col in columns:
        if col in ordinals:
            encoder = OrdinalEncoder()
            df[col] = encoder.fit_transform(raw[col].values)
            ordinal_encoders[col] = encoder
        else:
            encoder = MinMaxScaler()
            df[col] = encoder.fit_transform(encoder.fit_transform(raw[col].values))
            scalers[col] = encoder
    return df, ordinal_encoders, scalers

#This function performs a train-test split on the input data
def train_test_split(test_size: float, X, y, random_state=None):
    n = len(y)
    indices = [i for i in range(n)]
    if random_state is not None:
        random.seed(random_state)
    random.shuffle(indices)
    test_idx = int(n * test_size)
    return X[indices[0:test_idx]], X[indices[test_idx:-1]], y[indices[0:test_idx]], y[indices[test_idx:-1]]
