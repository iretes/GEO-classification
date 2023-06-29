# GEO Classification 🧬🤖
Implementation of a pipeline to compare the performance of Machine Learning classifiers on Gene Expression Omnibus (GEO) data.

## Installation
Install Python:

`sudo apt install python3`

Install pip:

`sudo apt install --upgrade python3-pip`

Install requirements:

`python -m pip install --requirement requirements.txt`

## Usage

### Step 1
Run the script `preprocess.py` to read, pre-process and save data into `.pkl` and/or `.csv` files from a GEO matrix series file.
```
python preprocess.py [-h] --txt-path TXT-PATH [--csv-path CSV-PATH] 
                        --pkl-path PKL-PATH --target-header TARGET-HEADER
                        --target-regexs TARGET-REGEXS [TARGET-REGEXS ...]
                        --new-targets NEW-TARGETS [NEW-TARGETS ...]
                        [--noise NOISE] [--log-transform | --no-log-transform]
                        [--std-samples | --no-std-samples]
```
| Argument | Optional | Description    |
| -------- | -------- | -------------- |
| -h, --help | ✔️ | Show the help message and exit. |
| --txt-path TXT-PATH | | Path to the series matrix file. |
| --csv-path CSV-PATH | ✔️ | Path to the `.csv` file to create. |
| --pkl-path PKL-PATH | | Path to the `.pkl` file to create (a serialized dictionary with the following keys: 'X', 'y', 'features_ids', 'samples_ids'). |
| --target-header TARGET-HEADER | | Header that identifies targets; WARNING: if there are multiple lines with the same header the first line will be used. |
| --target-regexs TARGET-REGEXS [TARGET-REGEXS ...] | | Regular expression to select samples by their target.
| --new-targets NEW-TARGETS [NEW-TARGETS ...] | | Targets to use instead. |
| --noise NOISE | ✔️ | Percentage of gaussian noise to add (default: 0). |
| --log-transform, --no-log-transform | ✔️ | Whether to apply a logarithimc transformation (default: False). |
| --std-samples, --no-std-samples | ✔️ | Whether to standardize samples (default: False). |
### Step 1.2 (optional)
Running the script `integrate.py` you can either merge pre-processed datasets obtained from experiments conducted on the same platform into a single one, or delete from them features that are not shared by all.
```
python integrate.py [-h] --pkl-in PKL-IN [PKL-IN ...] --pkl-out PKL-OUT [PKL-OUT ...]
                    [--csv-out CSV-OUT [CSV-OUT ...]]
```
| Argument | Optional | Description    |
| -------- | -------- | -------------- |
| -h, --help | ✔️ | Show the help message and exit. |
| --pkl-in PKL-IN [PKL-IN ...] | | List of `.pkl` files to integrate. |
| --pkl-out PKL-OUT [PKL-OUT ...] | | Path to the `.pkl` file/s to create. If a single path is passed, datasets will be merged, otherwise new datasets will be created with the only features they all share. |
| --csv-out CSV-OUT [CSV-OUT ...] | ✔️ | Path to the `.csv` file/s to create. If a single path is passed, datasets will be merged, otherwise new datasets will be created with the only features they all share. |

### Step 2
Run the script `compare.py` to compare the performances of ML classifiers on pre-processed datasets.
You can choose to compare the following classifiers: [Nearest Centroid](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html) (NC), [K-Nearest Neighbors](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) (KNN),
[Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) (SVM), [Gaussian Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) (GNB), [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) (RF), [Extreme Gradient Boosting](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwirouKg7tP-AhUSRfEDHaUdDnkQFnoECAwQAQ&url=https%3A%2F%2Fxgboost.readthedocs.io%2F&usg=AOvVaw1Rb2paRgUY_gHcA0BusqY4) (XGB), Rank Aggregation Classifier (RAC).

The table below shows the hyperparameters that will be explored. The unreported hyperparameters have been set to their default values.
| Classifier | Hyperparameter    | Values                                                 |
| ---------- | ----------------- | ------------------------------------------------------ |
| 🟥 NC      | `metric`            | `'euclidean', 'manhattan'`                           |
| 🟧 KNN     | `weights`           | `'uniform', 'distance'`                              |
| 🟧 KNN     | `n_neighbors`       | `range(3, max_neighbors+1, 2)`<sup>1</sup>           |
| 🟨 SVM     | `C`                 | `0.001, 0.1, 1, 100`                                 |
| 🟨 SVM     | `kernel`            | `'linear', 'poly', 'rbf', 'sigmoid'`                 |
| 🟨 SVM     | `degree`            | `2, 3, 4`                                            |
| 🟨 SVM     | `gamma`             | `'scale', 'auto', 0.001, 0.1, 1, 100`                |
| 🟩 GNB     | `var_smoothing`     | `1e-9, 1e-3, 1e-1, 0.5`                              |
| 🟦 RF      | `n_estimators`      | `50, 100, 500`                                       |
| 🟦 RF      | `min_samples_split` | `0.1, 0.25, 0.33, 2`                                 |
| 🟪 RAC     | `r_method`          | `'average', 'min', 'max'`                            |
| 🟪 RAC     | `ra_method`         | `'borda', 'borda_median', 'borda_gmean', 'borda_l2'` |
| 🟪 RAC     | `metric`            | `'spearman', 'kendall'`                              |
| 🟪 RAC     | `weighted`          | `True, False, (1/3*n_feature, 1/3*n_feature)`        |
| 🟪 RAC     | `p`                 | `1, 2, 3/4`                                          |
| ⬛ XGB     | `n_estimators`      | `50, 100, 200`                                       |
| ⬛ XGB     | `eta`               | `0.01, 0.1, 0.2, 0.3`                                |

<sup>1</sup> To avoid ties the number of neighbors is odd and max_neighbors is equal to the dimension of the training fold divided by the number of classes if such quantity is less than 20, otherwise it is equal to 20.

The performances of each classifier will be evaluated by computing the following scores: [accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score), [f1 micro](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score), [f1 macro](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score), [f1 weighted](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score), [AUROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score), [AUROC weighted](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score).
Scores will be computed either averaging their values over the test folds, or on the concatenation of the predictions made for each test fold.
Results will be saved in `.csv` files.
The predictions made by the classifiers will be serialized in the file `predictions.pkl`, to access them
see the script `read_predictions.py`.
```
python compare.py [-h] [--nested-cv | --no-nested-cv] --dev-dataset DEV-DATASET
                    [--test-dataset TEST-DATASET] [--pos-lbl POS-LBL]
                    --output-path OUTPUT-PATH
                    [--classifiers CLASSIFIERS [CLASSIFIERS ...]]
                    [--best-score BEST-SCORE] [--n-splits N-SPLITS] 
                    [--ext-n-splits EXT-N-SPLITS] [--loo | --no-loo]
                    [--ext-loo | --no-ext-loo]
                    [--random-state RANDOM-STATE] [--standardize | --no-standardize]
                    [--verbose | --no-verbose]
```
| Argument | Optional | Description    |
| -------- | -------- | -------------- |
| -h, --help | ✔️ | Show the help message and exit. |
|--nested-cv, --no-nested_cv | ✔️ | Whether to perform nested cross validation (default: False). |
|--dev-dataset DEV-DATASET | | Path to the development dataset. |
| --test-dataset TEST-DATASET | ✔️ | Path to the test dataset. |
| --pos-lbl POS-LBL | ✔️ | Label of the 'positive' class in binary classification. |
| --output-path OUTPUT-PATH | | Path where to save results. |
| --classifiers CLASSIFIERS [CLASSIFIERS ...] | ✔️ | Classifiers to compare, must be in ['RAC', 'NC', 'KNN', 'SVM', 'GNB', 'RF', 'XGB']. If not specified, all the classifiers will be compared. |
| --best-score BEST-SCORE | ✔️ | Scorer function to use to determine the best parameters, must be in ['accuracy', 'mean accuracy', 'concat accuracy', 'f1 micro', 'mean f1 micro', 'concat f1 micro', 'f1 macro', 'mean f1 macro', 'concat f1 macro', 'f1 weighted', 'mean f1 weighted', 'concat f1 weighted', 'aggregated rank']. If not specified, all the scores will be evaluated.
| --n-splits N-SPLITS | ✔️ | Number of folds to use; with --nested_cv refers to the number of internal folds; ignored with --loo (default: 5). |
| --ext-n-splits EXT-N-SPLITS | ✔️ | Number of folds to use in the external cross validation; ignored with --ext-loo (default: 5). |
| --loo, --no-loo | ✔️ | Whether to perform leave one out cross validation; with --nested_cv leave one out cross validation will be done in the internal cross validation (default: False). |
| --ext-loo, --no-ext-loo | ✔️ | Whether to perform leave one out cross validation in the external cross validation (default: False).
| --random-state RANDOM-STATE | ✔️ | Seed to get reproducible results (default: None). |
| --standardize, --no-standardize | ✔️ | Whether to standardize features (default: False). |
| --verbose, --no-verbose | ✔️ | Whether to print verbose output (default: False). |

**NOTE**: The steps above refer to the integration of datasets coming from the same platform, if you want to integrate dataset coming from different platfomrs you need to: 1) annotate the features using the R script `annotate.R`, passing as arguments the GEO accesion code (in this case the dataset will be automatically downloaded), the target header and the path to the file to create; 2) run the script `preprocess_annotated.py`, whose arguments are described in the following; 3) run the script `integrate.py`.
```
python preprocess_annotated.py [-h] --dataset dataset [--csv-path CSV-PATH] 
                        --pkl-path PKL-PATH
                        --target-regexs TARGET-REGEXS [TARGET-REGEXS ...]
                        --new-targets NEW-TARGETS [NEW-TARGETS ...]
                        [--log-transform | --no-log-transform]
                        [--std-samples | --no-std-samples]
```
| Argument | Optional | Description    |
| -------- | -------- | -------------- |
| -h, --help | ✔️ | Show the help message and exit. |
| --dataset DATASET | | Path to the `.csv` file with the annotated dataset. |
| --csv-path CSV-PATH | ✔️ | Path to the `.csv` file to create. |
| --pkl-path PKL-PATH | | Path to the `.pkl` file to create (a serialized dictionary with the following keys: 'X', 'y', 'features_ids', 'samples_ids'). |
| --target-regexs TARGET-REGEXS [TARGET-REGEXS ...] | | Regular expression to select samples by their target.
| --new-targets NEW-TARGETS [NEW-TARGETS ...] | | Targets to use instead. |
| --log-transform, --no-log-transform | ✔️ | Whether to apply a logarithimc transformation (default: False). |
| --std-samples, --no-std-samples | ✔️ | Whether to standardize samples (default: False). |

### Recursive Feature Elimination

Run the script `RFE.py` to apply [Recursive Feature Elimination](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html) and compare the performances of the classifiers over the dataset with the selected features. The performances and the selected feature will be saved in `.csv` files. The predictions made by the classifiers will be serialized in the file `predictions.pkl`, to access them see the script `read_predictions.py`. The ids of the selected features will be saved in `.csv` files (one for each classifier).
```
python RFE.py [-h] --dataset DATASET --rac-params RAC_PARAMS --svm-params SVM_PARAMS
                    --rf-params RF_PARAMS --xgb-params XGB_PARAMS 
                    [--pos-lbl POS_LBL] --output-path OUTPUT_PATH
                    [--n-splits N_SPLITS] [--random-state RANDOM_STATE]
                    [--standardize | --no-standardize] 
                    [--n-features-to-select N_FEATURES_TO_SELECT]
                    [--step STEP] [--best-score BEST_SCORE] [--verbose | --no-verbose]
```
| Argument | Optional | Description    |
| -------- | -------- | -------------- |
| -h, --help | ✔️ | Show the help message and exit. |
| --dataset DATASET | | Path to the dataset. |
| --rac-params RAC-PARAMS | | Path to the `.json` file with RAC parameters. |
| --svm-params SVM-PARAMS | | Path to the `.json` file with SVM parameters (kernel must be 'linear'). |
| --rf-params RF-PARAMS | | Path to the `.json` file with RF parameters. | 
| --xgb-params XGB-PARAMS | | Path to the `.json` file with XGB parameters. |
| --pos-lbl POS-LBL | ✔️ | Label of the 'positive' class in binary classification. |
| --output-path OUTPUT-PATH | | Path where to save results. |
| --n-splits N-SPLITS | ✔️ | Number of folds to use in the cross validation (default: 5). |
| --random-state RANDOM-STATE | ✔️ | Seed to get reproducible results (default: None). |
| --standardize, --no-standardize | ✔️ | Whether to standardize features (default: False). |
| --n-features-to-select | ✔️ | RFE parameter (default: 20).|
| --step | ✔️ | RFE parameter (default: 0.5). |
| --verbose, --no-verbose | ✔️ | Whether to print verbose output (default: False). |

### Feature importances

Run the script `feature_importances.py` to save the features ranked by importance in `.csv` files (one for each classifier).
```
python feature_importances.py [-h] --dataset DATASET 
                    --rac-params RAC_PARAMS --svm-params SVM_PARAMS
                    --rf-params RF_PARAMS --xgb-params XGB_PARAMS 
                    --output-path OUTPUT_PATH
```
| Argument | Optional | Description    |
| -------- | -------- | -------------- |
| -h, --help | ✔️ | Show the help message and exit. |
| --dataset DATASET | | Path to the dataset. |
| --rac-params RAC-PARAMS | | Path to the `.json` file with RAC parameters. |
| --svm-params SVM-PARAMS | | Path to the `.json` file with SVM parameters (kernel must be 'linear'). |
| --rf-params RF-PARAMS | | Path to the `.json` file with RF parameters. | 
| --xgb-params XGB-PARAMS | | Path to the `.json` file with XGB parameters. |
| --output-path OUTPUT-PATH | | Path where to save results. |

### Usage example
See file `run_pipeline.sh`.

## Results
The pipeline was applied to the datasets outlined in **Table 1**. The experimental results are reported in **Tables 2-5**.

**Table 1**: Datasets details
<p align="left">
    <img align="center" src='tables/datasets.png' width="550px">
</p>
**Table 2**: F1 and AUROC scores from nested cross validation on the concatenated datasets (’*’ for z-score normalized samples)
<p align="left">
    <img align="center" src='tables/nested.png' width="550px">
</p>

**Table 3:** F1 and AUROC scores on test set, when training and testing on different datasets (’*’ for z-score normalized samples)
<p align="left">
    <img align="center" src='tables/hold_out.png' width="550px">
</p>

**Table 4:** F1 score from a simple cross validation run (’*’ for z-score normalized samples)
<p align="left">
    <img align="center" src='tables/f1_cv.png' width="550px">
</p>

**Table 5:** Known/Unknown genes in the litarature related to Psoriasis among the genes with the highest 5 importance scores
<p align="left">
    <img align="center" src='tables/known_genes.png' width="550px">
</p>

The list of genes with the highest 5 importance values on the Psoriasis datasets, together with an evaluation of their relevance for the disease is reported in the file `psoriasis_genes.pdf` inside the folder `tables`.
