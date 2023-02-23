import pickle as pkl
import argparse

# How to access predictions

parser = argparse.ArgumentParser(description='Read predictions')
parser.add_argument(
    '--pkl',
    action='store',
    type=str,
    required=True,
    help="Path to the .pkl file that contains predictions."
)

args = parser.parse_args()
pklpath = args.pkl

with open(pklpath, 'rb') as fp :
    dict = pkl.load(fp)

"""
Example of dict fields

dict = {
    "KNN": {
        "preds" : ["Cancer", "Normal", ...],
        "probs" : [0.6, 0.9, ...],
        "true" : ["Normal", "Normal", ...]
    },
    ...
}
"""

clfs = ['RAC', 'NC', 'KNN', 'SVM', 'RF']
used_clf = None
for clf in clfs:
    if clf in dict:
        used_clf = clf
        print("%s prediction (and proability) for the first sample"%clf)
        print("%s (%s)"%(str(dict[clf]['preds'][0]), str(dict[clf]['probs'][0])))
print("True target of the first sample")
print(dict[used_clf]['true'][0])