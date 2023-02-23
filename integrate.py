import numpy as np
import pickle as pkl
import argparse
import pandas as pd

# Cmd line argument parsing
parser = argparse.ArgumentParser(description='Integrate datasets')
parser.add_argument(
    '--pkl-in',
    nargs='+',
    type=str,
    required=True,
    help="List of .pkl files to integrate."
)
parser.add_argument(
    '--pkl-out',
    nargs='+',
    type=str,
    required=True,
    help='Path to the .pkl file/s to create. If a single path is passed, ' +
    'datasets will be merged, otherwise new datasets will be created with ' +
    'the only features they all share.'
)
parser.add_argument(
    '--csv-out',
    nargs='+',
    type=str,
    default=[],
    help='Path to the .csv file/s to create. If a single path is passed, ' + 
    'datasets will be merged, otherwise new datasets will be created with ' + 
    'the only features they all share.'
)

args = parser.parse_args()
in_paths = args.pkl_in
out_pkls = args.pkl_out
out_csvs = args.csv_out

if len(out_pkls) > 1 and len(out_pkls) != len(in_paths):
    raise ValueError("Wrong number of arguments.")
if out_csvs and len(out_csvs) != len(out_pkls):
    raise ValueError("Wrong number of arguments.")

# unpickle first dataset
with open(in_paths[0], 'rb') as fp :
    dataset = pkl.load(fp)
X = dataset['X']
y = dataset['y']
ids = dataset['samples_ids']
features = dataset['features_ids']

# concatenate datasets
row_idx = [] # row indeces where each dataset starts
for file in in_paths[1:]:
    row_idx.append(X.shape[0])
    with open(file, 'rb') as fp :
        dataset = pkl.load(fp)
    X2 = dataset['X']
    if dataset['features_ids'] != features:
        column_idx1 = [] # column indeces of the shared features in the merged dataset
        column_idx2 = [] # column indeces of the shared features in the dataset to append
        for i, f in enumerate(features):
            try:
                j = dataset['features_ids'].index(f)
                column_idx1.append(i)
                column_idx2.append(j)
            except ValueError:
                continue
        X = X[:][:, column_idx1]
        X2 = dataset['X'][:][:, column_idx2]
        features = list(features[i] for i in column_idx1)
    X = np.concatenate((X, X2), axis=0)
    y = np.concatenate((y, dataset['y']), axis=0)
    ids = np.concatenate((ids, dataset['samples_ids']), axis=0)
row_idx.append(X.shape[0])

if len(out_pkls) > 1:
    # split merged dataset and save them
    start_idx = 0
    for i, (end_idx, path) in enumerate(zip(row_idx, out_pkls)):
        new_dataset = {}
        new_dataset['X'] = X[start_idx:end_idx,:]
        new_dataset['y'] = y[start_idx:end_idx]
        new_dataset['features_ids'] = features
        new_dataset['samples_ids'] = ids[start_idx:end_idx]
        start_idx = end_idx
        print("# Features = %d"%new_dataset['X'].shape[1])
        print("# Samples = %d"%new_dataset['X'].shape[0])
        with open(path, 'wb') as fp :
            pkl.dump(new_dataset, fp)
        if out_csvs:
            df = pd.DataFrame(data=new_dataset['X'], columns=new_dataset['features_ids'])
            df.insert(0, 'SAMPLE_ID', new_dataset['samples_ids'])
            df.insert(0, 'TARGET', new_dataset['y'])
            df.to_csv(out_csvs[i])
else:
    # save merged dataset
    new_dataset = {}
    new_dataset['X'] = X
    new_dataset['y'] = y
    new_dataset['features_ids'] = features
    new_dataset['samples_ids'] = ids
    print("# Features = %d"%X.shape[1])
    print("# Samples = %d"%X.shape[0])
    with open(out_pkls[0], 'wb') as fp :
        pkl.dump(new_dataset, fp)
    if out_csvs:
        df = pd.DataFrame(data=X, columns=new_dataset['features_ids'])
        df.insert(0, 'SAMPLE_ID', new_dataset['samples_ids'])
        df.insert(0, 'TARGET', new_dataset['y'])
        df.to_csv(out_csvs[0])