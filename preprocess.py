import pandas as pd
import numpy as np
import pickle as pkl
import csv
import re
import argparse
from collections import Counter
from sklearn.preprocessing import StandardScaler

# cmd line argument parsing
parser = argparse.ArgumentParser(description='GEO matrix series preprocessing')
parser.add_argument(
    '--txt-path',
    action='store',
    type=str,
    required=True,
    help='Path to the series matrix file.')
parser.add_argument(
    '--csv-path',
    action='store',
    type=str,
    required=False,
    help='Path to the .csv file to create.')
parser.add_argument(
    '--pkl-path',
    action='store',
    type=str,
    required=True,
    help="Path to the .pkl file to create (a serialized dictionary with " +
      "the following keys: 'X', 'y', 'features_ids', 'samples_ids').")
parser.add_argument(
    '--target-header',
    action='store',
    type=str,
    required=True,
    help='Header that identifies targets; WARNING: if there are multiple' +
    'lines with the same header the first line will be used.')
parser.add_argument(
    '--target-regexs',
    nargs='+',
    type=str,
    required=True,
    help='Regular expression to select samples by their target.')
parser.add_argument(
    '--new-targets',
    nargs='+',
    type=str,
    required=True,
    help='Targets to use instead.')
parser.add_argument(
    '--noise',
    action='store',
    type=int,
    required=False,
    help='Percentage of gaussian noise to add (default: 0).')
parser.add_argument(
    '--log-transform',
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to apply a logarithimc transformation."
)
parser.add_argument(
    '--std-samples',
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to standardize samples."
)

args = parser.parse_args()
txt_path = args.txt_path
csv_path = args.csv_path
pkl_path = args.pkl_path
target_header = args.target_header
target_regexs = args.target_regexs
new_targets_names = args.new_targets
noise = args.noise
log_transform = args.log_transform
std_samples = args.std_samples

MATRIX_BEGIN_HEADER = '!series_matrix_table_begin'
MATRIX_END_HEADER = '!series_matrix_table_end'
FEATURE_HEADER = 'ID_REF'
DELIMITER = '\t'
COMMENT = '!'

# read all targets
targets = []
with open(txt_path, newline='') as csvfile:
    file = csv.reader(csvfile, delimiter=DELIMITER)
    for row in file:
        if row and row[0] == target_header:
            targets = row[1:]
            break

# read all data
df = pd.read_csv(txt_path, sep=DELIMITER, comment=COMMENT).dropna() # drop empty values
n_features = df.shape[0]
n_samples = df.shape[1] - 1
if n_samples != len(targets):
    raise ValueError("The number of samples (%d) is not equal to the number "+
                     "of targets (%d)."%(n_samples, len(targets)))

# take requested samples
columns_to_drop = []
new_targets = []
for i, target in enumerate(targets):
    found = False
    for regex, new_t in zip(target_regexs, new_targets_names):
        if re.match(regex, target):
            new_targets.append(new_t)
            found = True
            break
    if not found:
        columns_to_drop.append(i+1)
df.drop(df.columns[columns_to_drop], axis=1, inplace=True)
n_samples = len(df.columns)-1

# convert data to numpy array and transpose it
# (samples on the rows, features on the columns)
X = np.transpose(df.drop(FEATURE_HEADER, axis=1).to_numpy())

# add noise
if noise:
    # compute mean for each feature
    mean = abs(X.mean(axis=0))
    # compute std as percentage of the mean of each feature
    std = mean*(noise/100)
    # for each feature sample n_samples points from gaussian centered in 0 with respective std
    noise = np.transpose(np.array(
    [np.random.normal(0, s, X.shape[0]) for s in std]
    ))
    # add noise to data
    X = X + noise

# log2 transform
if log_transform:
    min = np.min(X)
    if min < 0:
        X += abs(min)
        min = 0
    if min >= 0 and min < 1:
        X += 1
    X = np.log2(X)

# standardize samples
if std_samples:
    scaler = StandardScaler()
    X = np.transpose(scaler.fit_transform(np.transpose(X)))

dataset = {}
dataset['X'] = X
dataset['y'] = np.array(new_targets)
dataset['features_ids'] = list(df[FEATURE_HEADER])
dataset['samples_ids'] = df.columns.values[1:]

# display dataset info
print("# Features = %d"%n_features)
print("# Samples = %d"%n_samples)
print("# Samples per class = %s"%dict(Counter(new_targets)))

# write parsed data to csv
if csv_path:
    df = pd.DataFrame(data=X, columns=dataset['features_ids'])
    df.insert(0, 'SAMPLE_ID', dataset['samples_ids'])
    df.insert(0, 'TARGET', dataset['y'])
    df.to_csv(csv_path)

# serialize dict with parsed data
with open(pkl_path, 'wb') as fp :
    pkl.dump(dataset, fp)