#!/bin/bash

# ----------------------------------- Step 1 -----------------------------------
python ./preprocess.py \
--txt-path './datasets/GSE13355_series_matrix.txt' \
--csv-path './datasets/GSE13355.csv' \
--pkl-path './datasets/GSE13355.pkl' \
--target-header '!Sample_characteristics_ch1' \
--target-regexs '^involved.*' '^.*controls.*$' \
--new-targets 'Psoriasis' 'Normal'

python ./preprocess.py \
--txt-path './datasets/GSE14905_series_matrix.txt' \
--csv-path './datasets/GSE14905.csv' \
--pkl-path './datasets/GSE14905.pkl' \
--target-header '!Sample_characteristics_ch1' \
--target-regexs '^.*psoriasis.*$' '^.*normal.*$' \
--new-targets 'Psoriasis' 'Normal'

# --------------------------------- Step 1.2 -----------------------------------
python ./integrate.py \
--pkl-in './datasets/GSE13355.pkl' './datasets/GSE14905.pkl' \
--pkl-out './datasets/GSE13355-GSE14905.pkl'

python ./integrate.py \
--pkl-in './datasets/GSE13355.pkl' './datasets/GSE14905.pkl' \
--pkl-out './datasets/GSE13355_sub.pkl' './datasets/GSE14905_sub.pkl'

# ----------------------------------- Step 2 -----------------------------------

# Create folders to store results
mkdir './results/'
mkdir './results/GSE13355/'
mkdir './results/GSE14905/'
mkdir './results/GSE13355-GSE14905/'
mkdir './results/GSE13355(TR)-GSE14905(TS)/'

# Compare classifiers on GSE13355
python ./compare.py \
--nested-cv \
--dev-dataset './datasets/GSE13355.pkl' \
--output-path './results/GSE13355/' \
--best-score 'concat f1 weighted' \
--n-splits 5 \
--ext-n-splits 5 \
--verbose

# Compare classifiers on GSE14905
python ./compare.py \
--nested-cv \
--dev-dataset './datasets/GSE14905.pkl' \
--output-path './results/GSE14905/' \
--best-score 'concat f1 weighted' \
--n-splits 5 \
--ext-n-splits 5 \
--verbose

# Compare classifiers on the dataset obtained merging GSE13355 and GSE14905
python ./compare.py \
--nested-cv \
--dev-dataset './datasets/GSE13355-GSE14905.pkl' \
--output-path './results/GSE13355-GSE14905/' \
--best-score 'concat f1 weighted' \
--n-splits 5 \
--ext-n-splits 5 \
--verbose

# Compare classifiers training them on GSE13355 and testing them on GSE14905
python ./compare.py \
--dev-dataset './datasets/GSE13355_sub.pkl' \
--test-dataset './datasets/GSE14905_sub.pkl' \
--output-path './results/GSE13355(TR)-GSE14905(TS)/' \
--best-score 'concat f1 weighted' \
--n-splits 5 \
--ext-n-splits 5 \
--verbose