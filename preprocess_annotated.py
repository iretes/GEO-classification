import pandas as pd
import numpy as np
import argparse
from collections import Counter
from sklearn.preprocessing import StandardScaler
import pickle as pkl

parser = argparse.ArgumentParser(description='Preprocess annotated dataset')
parser.add_argument(
    '--dataset',
    action='store',
    type=str,
    required=True,
    help='Path to the csv file.')
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
parser.add_argument(
    '--pkl-path',
    action='store',
    type=str,
    required=True,
    help="Path to the .pkl file to create (a serialized dictionary with " +
      "the following keys: 'X', 'y', 'features_ids', 'samples_ids').")
parser.add_argument(
    '--csv-path',
    action='store',
    type=str,
    required=False,
    help='Path to the .csv file to create.')

args = parser.parse_args()
dataset = args.dataset
target_regexs = args.target_regexs
new_targets_names = args.new_targets
log_transform = args.log_transform
std_samples = args.std_samples
pkl_path = args.pkl_path
csv_path = args.csv_path

data = pd.read_csv(dataset).dropna(axis=0)
targets = list(data.loc['TARGET'])
new_targets = []
targets_to_drop = []
for i, target in enumerate(targets):
    found = False
    for regex, new_t in zip(target_regexs, new_targets_names):
        if re.match(regex, target):
            new_targets.append(new_t)
            found = True
            break
    if not found:
        targets_to_drop.append(i)
data.drop(data.columns[targets_to_drop], axis=1, inplace=True)

data = data.drop("TARGET")
data = data.apply(pd.to_numeric)
data = data.groupby(by=data.index).mean() # average features with same id
X = np.transpose(data.to_numpy())

if log_transform:
    min = np.min(X)
    if min < 0:
        X += abs(min)
        min = 0
    if min >= 0 and min < 1:
        X += 1
    X = np.log2(X)

if std_samples:
    scaler = StandardScaler()
    X = np.transpose(scaler.fit_transform(np.transpose(X)))

y = np.array(new_targets)
features_ids = list(data.index)
samples_ids = list(data.columns)

if len(features_ids) != X.shape[1]:
	raise ValueError("# feature wrong")
if len(samples_ids) != X.shape[0]:
	raise ValueError("# samples wrong")

print("# Features = %d"%len(features_ids))
print("# Samples = %d"%len(samples_ids))
print("# Samples per class = %s"%dict(Counter(new_targets)))

dataset = {}
dataset['X'] = X
dataset['y'] = y
dataset['features_ids'] = features_ids
dataset['samples_ids'] = samples_ids

if csv_path:
    df = pd.DataFrame(data=X, columns=dataset['features_ids'])
    df.insert(0, 'SAMPLE_ID', dataset['samples_ids'])
    df.insert(0, 'TARGET', dataset['y'])
    df.to_csv(csv_path)

with open(pkl_path, 'wb') as fp :
    pkl.dump(dataset, fp)