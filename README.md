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
The pipeline was applied to the datasets outlined in **Table 1**. The experimental results are reported in **Tables 2-8**.

### Table 1: Datasets details
| **Name (accession)** | **Platform (accession)** | **Classifier (\#samples)**  | **\#Features** | **Description** |
|----------------------|--------------------------|-----------------------------|----------------|-----------------|
| LC1 (GSE19804) | Affymetrix Human Genome U133 Plus 2.0 Array (GPL570) | Lung Cancer (60), Control(60) | 54675 | Genome-wide screening of transcriptional modulation in non-smoking female lung cancer in Taiwan |
| LC2 (GSE43346) | Affymetrix Human Genome U133 Plus 2.0 Array (GPL570) | Small Cell Lung Cancer (23), Control (42) | 54675 | Gene repression with H3K27me3 modification in human small cell lung cancer |
| PSO1 (GSE14905) | Affymetrix Human Genome U133 Plus 2.0 Array (GPL570) | Psoriasis (61), Control (21) | 54675 | Type I Interferon: Potential Therapeutic Target for Psoriasis? |
| PSO2 (GSE13355) | Affymetrix Human Genome U133 Plus 2.0 Array (GPL570) | Psoriasis (58), Control (64) | 54675 | Gene expression data of skin from psoriatic patients and normal controls |
| SK1 (GSE15605) | Affymetrix Human Genome U133 Plus 2.0 Array (GPL570) | Primary Melanoma (46), Metastatic Melanoma (12), Control (16) | 44137 | Transcriptome profiling identifies HMGA2 as a novel gene in melanoma progression |
| SK2 (GSE46517) | Affymetrix Human Genome U133A Array (GPL96) | Primary Melanoma (31), Metastatic Melanoma (73), Control (7) | 22215 | Human melanoma samples comparing nevi and primary and metastatic melanoma |
| LK1 (GSE51082) | Affymetrix Human Genome U133A Array (GPL96) | Acute Myeloid Leukemia (37), Chronic Lymphocytic Leukemia (41), Chronic Myeloid Leukemia (22), Myelodysplastic Syndrome (10), Precursor B-cell Acute Lymphoblastic Leukemia (17), T-cell Acute Lymphoblastic Leukemia (12) | 22283 | Expression data from mononuclear cells from bone marrow and peripheral blood of leukemia patient samples |
| LK2 (GSE51082) | Affymetrix Human Genome U133B Array (GPL97) | Acute Myeloid Leukemia (37), Chronic Lymphocytic Leukemia (41), Chronic Myeloid Leukemia (22), Myelodysplastic Syndrome (10), Precursor B-cell Acute Lymphoblastic Leukemia (17), T-cell Acute Lymphoblastic Leukemia (13) | 22645 | Expression data from mononuclear cells from bone marrow and peripheral blood of leukemia patient samples |
| AD1 (GSE63060) | Illumina HumanHT-12 V3.0 expression beadchip (GPL6947) | Alzheimer's disease (145),  Control (104) | 38323 | Alzheimer, MCI and control samples from AddneuroMed Cohort (batch 1) |
| AD2 (GSE63061) | Illumina HumanHT-12 V4.0 expression beadchip (GPL10558) | Alzheimer's disease (140), Control (135) | 32049 | Alzheimer, MCI and control samples from AddneuroMed Cohort (batch 2) |
| AD3 (GSE33000)| Rosetta/Merck Human 44k 1.1 microarray (GPL4372) | Alzheimer's disease (310), Control (157) | 38734 | Gene expression profiles of human prefrontal cortex brain tissues |
| AD4 (GSE44770) | Rosetta/Merck Human 44k 1.1 microarray (GPL4372) | Alzheimer's disease (129), Control (101) | 39005 | Multi-tissue gene expression profiles of human brain (PFC) |
| PD1 (GSE62283) | Invitrogen ProtoArray v5.0 (GPL13669) | Parkinson's disease (132), Control (156) | 9480 | Diagnosis of Early-Stage Parkinson's Disease Using Autoantibodies as Blood-based Biomarkers |
| PD2 (GSE29654) | Invitrogen ProtoArray v5.0 (GPL13669) | Parkinson's disease (174), Control (80) | 9480 | Diagnosis of Parkinson's Disease Based on Disease-Specific Autoantibody Profiles |

### Table 2: F1 scores from nested cross validation on the merged datasets (’*’ for z-score normalized samples)

| **Dataset**                        | **RAC**        | **NC**         | **KNN**        | **SVM**        | **GNB**        | **RF**         | **XGB**        |
|------------------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| LC1+LC2                            | 0.930          | 0.795          | 0.929          | **0.968**      | *0.604*        | 0.962          | 0.930          |
| LC1+LC2*                           | 0.930          | 0.930          | *0.913*        | **0.962**      | 0.919          | 0.946          | 0.941          |
| PSO1+PSO2                          | 0.894          | *0.612*        | 0.961          | 0.975          | *0.612*        | **0.980**      | 0.936          |
| PSO1+PSO2*                         | *0.894*        | *0.894*        | 0.966          | **0.985**      | *0.894*        | 0.971          | 0.946          |
| SK1+SK2                            | 0.677          | *0.664*        | 0.898          | **0.935**      | 0.683          | 0.892          | 0.885          |
| SK1+SK2*                           | 0.677          | *0.664*        | 0.914          | **0.935**      | 0.677          | 0.919          | 0.913          |
| LK1+LK2                            | 0.829          | *0.801*        | 0.932          | **0.996**      | 0.939          | 0.967          | 0.936          |
| LK1+LK2*                           | 0.829          | *0.421*        | 0.920          | **0.996**      | 0.701          | 0.960          | 0.939          |
| AD1+AD2                            | 0.596          | *0.535*        | 0.695          | **0.744**      | *0.535*        | 0.692          | 0.692          |
| AD1+AD2*                           | *0.596*        | 0.645          | 0.702          | **0.748**      | 0.624          | 0.694          | 0.719          |
| AD3+AD4                            | 0.867          | *0.847*        | 0.944          | **0.973**      | 0.854          | 0.957          | 0.967          |
| AD3+AD4*                           | *0.867*        | 0.881          | 0.947          | **0.971**      | 0.883          | 0.955          | 0.966          |
| AD1+AD2+AD3+AD4                    | 0.695          | *0.557*        | 0.796          | 0.792          | *0.557*        | 0.790          | **0.808**      |
| AD1+AD2+AD3+AD4*                   | 0.695          | 0.651          | 0.799          | 0.794          | *0.557*        | 0.821          | **0.831**      |
| PD1+PD2                            | 0.886          | *0.759*        | 0.927          | 0.948          | 0.803          | 0.937          | **0.959**      |
| PD1+PD2*                           | 0.886          | 0.873          | 0.916          | 0.943          | *0.843*        | **0.946**      | 0.939          |
| **Mean**                           | 0.797          | *0.696*        | 0.885          | **0.916**      | 0.698          | 0.897          | 0.889          |
| **Rank**                           | 5              | 7              | 4              | 1              | 6              | 2              | 3              |
| **Mean\***                         | 0.797          | *0.745*        | 0.885          | **0.917**      | 0.762          | 0.902          | 0.899          |
| **Rank\***                         | 5              | 7              | 4              | 1              | 6              | 2              | 3              |

### Table 3: AUROC scores from nested cross validation on the merged datasets (’*’ for z-score normalized samples)

| **Dataset**                        | **RAC**        | **NC**         | **KNN**        | **SVM**        | **GNB**        | **RF**         | **XGB**        |
|------------------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| LC1+LC2                            | 0.950          | 0.919          | 0.949          | 0.980          | *0.755*        | **0.984**      | 0.979          |
| LC1+LC2*                           | 0.950          | 0.955          | *0.922*        | **0.982**      | 0.969          | 0.980          | 0.962          |
| PSO1+PSO2                          | 0.863          | 0.693          | 0.977          | **0.994**      | *0.633*        | 0.990          | 0.977          |
| PSO1+PSO2*                         | *0.863*        | 0.934          | 0.977          | 0.985          | 0.876          | **0.989**      | 0.968          |
| SK1+SK2                            | 0.823          | 0.857          | 0.947          | **0.975**      | *0.734*        | 0.971          | 0.943          |
| SK1+SK2*                           | 0.823          | 0.875          | 0.952          | 0.974          | *0.740*        | **0.975**      | 0.963          |
| LK1+LK2                            | 0.946          | *0.928*        | 0.971          | **1.000**      | 0.983          | 0.996          | 0.996          |
| LK1+LK2*                           | 0.946          | *0.724*        | 0.958          | **1.000**      | 0.934          | 0.997          | 0.994          |
| AD1+AD2                            | 0.634          | 0.550          | 0.755          | **0.830**      | *0.549*        | 0.772          | 0.787          |
| AD1+AD2*                           | *0.634*        | 0.684          | 0.761          | **0.825**      | 0.674          | 0.786          | 0.793          |
| AD3+AD4                            | 0.922          | 0.931          | 0.977          | **0.997**      | *0.860*        | 0.988          | 0.991          |
| AD3+AD4*                           | 0.922          | 0.935          | 0.975          | **0.997**      | *0.895*        | 0.987          | 0.992          |
| AD1+AD2+AD3+AD4                    | 0.808          | 0.552          | 0.847          | 0.784          | *0.539*        | 0.895          | **0.906**      |
| AD1+AD2+AD3+AD4*                   | 0.808          | 0.658          | 0.888          | 0.880          | *0.543*        | **0.917**      | 0.916          |
| PD1+PD2                            | 0.967          | *0.811*        | 0.965          | **0.983**      | 0.876          | **0.983**      | 0.978          |
| PD1+PD2*                           | 0.967          | 0.937          | 0.956          | **0.987**      | *0.890*        | 0.986          | 0.983          |
| **Mean**                           | 0.864          | 0.780          | 0.924          | 0.943          | *0.741*        | 0.936          | **0.945**      |
| **Rank**                           | 5              | 6              | 4              | 2              | 7              | 3              | 1              |
| **Mean\***                         | 0.864          | 0.838          | 0.924          | **0.954**      | *0.815*        | 0.952          | 0.946          |
| **Rank\***                         | 5              | 6              | 4              | 1              | 7              | 2              | 3              |

### Table 4: F1 scores on test set, when training and testing on different datasets (’*’ for z-score normalized samples)

| **Training**                       | **Test**       | **RAC**        | **NC**         | **KNN**        | **SVM**        | **GNB**        | **RF**         | **XGB**        |
|------------------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| LC1                                | LC2            | 0.708          | 0.602          | **0.865**      | 0.501          | *0.185*        | 0.217          | 0.490          |
| LC1*                               | LC2*           | 0.708          | 0.773          | **0.865**      | 0.410          | *0.185*        | 0.360          | 0.677          |
| LC2                                | LC1            | *0.333*        | *0.333*        | *0.333*        | *0.333*        | *0.333*        | **0.659**      | *0.333*        |
| LC2*                               | LC1*           | *0.333*        | *0.333*        | **0.606**      | 0.369          | 0.369          | 0.352          | 0.369          |
| PSO1                               | PSO2           | 0.984          | **0.992**      | *0.306*        | 0.361          | *0.306*        | 0.361          | 0.361          |
| PSO1*                              | PSO2*          | 0.984          | **0.992**      | 0.611          | 0.876          | *0.306*        | 0.361          | 0.569          |
| PSO2                               | PSO1           | *0.104*        | 0.512          | 0.666          | 0.653          | **0.669**      | 0.635          | 0.635          |
| PSO2*                              | PSO1*          | *0.104*        | 0.449          | **0.713**      | 0.627          | 0.601          | 0.635          | 0.635          |
| SK1                                | SK2            | **0.836**      | 0.454          | 0.204          | *0.007*        | 0.122          | 0.154          | 0.564          |
| SK1*                               | SK2*           | 0.836          | **0.827**      | *0.313*        | 0.823          | 0.499          | 0.702          | 0.553          |
| SK2                                | SK1            | 0.535          | 0.532          | 0.629          | 0.535          | *0.397*        | 0.627          | **0.680**      |
| SK2*                               | SK1*           | 0.535          | 0.481          | **0.728**      | 0.634          | *0.045*        | 0.666          | 0.526          |
| LK1                                | LK2            | 0.809          | 0.595          | **0.986**      | 0.879          | 0.335          | 0.440          | *0.331*        |
| LK1*                               | LK2*           | **0.809**      | *0.133*        | *0.133*        | 0.324          | *0.133*        | 0.198          | 0.411          |
| LK2                                | LK1            | 0.828          | 0.649          | 0.700          | **0.894**      | 0.753          | 0.642          | *0.560*        |
| LK2*                               | LK1*           | **0.828**      | 0.126          | *0.051*        | 0.144          | 0.134          | 0.164          | 0.062          |
| AD1                                | AD2            | 0.593          | 0.640          | **0.655**      | 0.343          | 0.343          | *0.323*        | 0.343          |
| AD1*                               | AD2*           | 0.593          | **0.621**      | 0.608          | 0.343          | 0.579          | *0.323*        | 0.486          |
| AD2                                | AD1            | 0.665          | **0.693**      | 0.680          | 0.429          | 0.625          | *0.246*        | *0.246*        |
| AD2*                               | AD1*           | 0.665          | 0.695          | 0.651          | *0.246*        | **0.701**      | 0.582          | 0.387          |
| AD3                                | AD4            | 0.857          | *0.808*        | 0.887          | **1.000**      | 0.830          | **1.000**      | **1.000**      |
| AD3*                               | AD4*           | *0.857*        | 0.870          | 0.931          | 0.996          | 0.883          | 0.987          | **1.000**      |
| AD4                                | AD3            | *0.875*        | 0.882          | 0.898          | **0.961**      | 0.877          | 0.937          | 0.950          |
| AD4*                               | AD3*           | *0.875*        | 0.876          | 0.941          | **0.963**      | 0.884          | 0.932          | 0.950          |
| PD1                                | PD2            | *0.288*        | 0.431          | 0.973          | **1.000**      | 0.802          | **1.000**      | **1.000**      |
| PD1*                               | PD2*           | 0.288          | *0.199*        | 0.953          | **1.000**      | 0.795          | **1.000**      | **1.000**      |
| PD2                                | PD1            | **0.726**      | 0.706          | 0.669          | 0.675          | 0.587          | 0.572          | *0.536*        |
| PD2*                               | PD1*           | **0.726**      | 0.608          | 0.614          | 0.625          | 0.657          | 0.572          | *0.545*        |
| **Mean**                           |                | 0.653          | 0.631          | **0.675**      | 0.612          | *0.512*        | 0.558          | 0.574          |
| **Rank**                           |                | 2              | 3              | 1              | 4              | 7              | 6              | 5              |
| **Mean\***                         |                | **0.653**      | 0.570          | 0.623          | 0.599          | *0.484*        | 0.560          | 0.584          |
| **Rank\***                         |                | 1              | 5              | 2              | 3              | 7              | 6              | 4              |

### Table 5: AUROC scores on test set, when training and testing on different datasets (’*’ for z-score normalized samples)

| **Training**                       | **Test**       | **RAC**        | **NC**         | **KNN**        | **SVM**        | **GNB**        | **RF**         | **XGB**        |
|------------------------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| LC1                                | LC2            | 0.941          | **0.970**      | 0.957          | 0.947          | *0.500*        | 0.797          | 0.865          |
| LC1*                               | LC2*           | 0.941          | 0.950          | 0.957          | **0.978**      | *0.500*        | 0.767          | 0.870          |
| LC2                                | LC1            | 0.797          | 0.848          | 0.592          | **0.893**      | *0.500*        | 0.819          | *0.500*        |
| LC2*                               | LC1*           | 0.797          | 0.843          | 0.650          | **0.883**      | *0.517*        | 0.818          | *0.517*        |
| PSO1                               | PSO2           | **1.000**      | **1.000**      | *0.500*        | *0.500*        | *0.500*        | *0.500*        | *0.500*        |
| PSO1*                              | PSO2*          | **1.000**      | **1.000**      | 0.663          | 0.961          | *0.500*        | 0.988          | 0.569          |
| PSO2                               | PSO1           | 0.759          | **0.781**      | 0.715          | 0.770          | 0.693          | *0.500*        | *0.500*        |
| PSO2*                              | PSO1*          | *0.759*        | 0.782          | 0.772          | 0.764          | 0.769          | **0.821**      | 0.500          |
| SK1                                | SK2            | 0.853          | **0.892**      | 0.819          | 0.836          | *0.500*        | 0.882          | 0.707          |
| SK1*                               | SK2*           | 0.853          | 0.896          | 0.858          | **0.918**      | 0.825          | 0.916          | *0.556*        |
| SK2                                | SK1            | 0.878          | **0.897**      | 0.836          | 0.787          | 0.764          | 0.784          | *0.743*        |
| SK2*                               | SK1*           | 0.878          | **0.895**      | 0.802          | 0.832          | *0.500*        | 0.839          | 0.755          |
| LK1                                | LK2            | 0.943          | 0.853          | **0.990**      | 0.980          | 0.848          | 0.893          | *0.839*        |
| LK1*                               | LK2*           | **0.943**      | 0.673          | *0.514*        | 0.892          | 0.521          | 0.776          | 0.766          |
| LK2                                | LK1            | 0.943          | 0.896          | 0.963          | **0.999**      | 0.959          | 0.943          | *0.893*        |
| LK2*                               | LK1*           | **0.943**      | 0.891          | *0.507*        | 0.817          | 0.624          | 0.780          | 0.625          |
| AD1                                | AD2            | 0.667          | 0.687          | 0.725          | **0.751**      | 0.500          | *0.369*        | 0.511          |
| AD1*                               | AD2*           | 0.667          | 0.691          | 0.666          | **0.751**      | 0.614          | *0.452*        | 0.625          |
| AD2                                | AD1            | 0.736          | 0.744          | 0.738          | 0.500          | **0.750**      | *0.419*        | 0.500          |
| AD2*                               | AD1*           | 0.736          | 0.746          | 0.757          | 0.817          | 0.737          | 0.691          | *0.677*        |
| AD3                                | AD4            | 0.942          | 0.938          | 0.974          | **1.000**      | *0.844*        | **1.000**      | **1.000**      |
| AD3*                               | AD4*           | 0.942          | 0.944          | 0.989          | **1.000**      | *0.903*        | **1.000**      | **1.000**      |
| AD4                                | AD3            | 0.929          | 0.929          | 0.943          | **0.993**      | *0.859*        | 0.982          | 0.986          |
| AD4*                               | AD3*           | 0.929          | 0.934          | 0.922          | **0.994**      | *0.863*        | 0.976          | 0.986          |
| PD1                                | PD2            | 0.944          | *0.808*        | 0.999          | **1.000**      | 0.938          | **1.000**      | **1.000**      |
| PD1*                               | PD2*           | 0.944          | *0.730*        | 0.991          | **1.000**      | 0.980          | **1.000**      | **1.000**      |
| PD2                                | PD1            | **0.816**      | 0.783          | 0.678          | 0.810          | 0.587          | 0.721          | *0.565*        |
| PD2*                               | PD1*           | **0.816**      | 0.728          | 0.669          | 0.766          | 0.770          | 0.785          | *0.631*        |
| **Mean**                           |                | **0.868**      | 0.859          | 0.816          | 0.840          | *0.695*        | 0.758          | 0.722          |
| **Rank**                           |                | 1              | 2              | 4              | 3              | 7              | 5              | 6              |
| **Mean\***                         |                | 0.868          | 0.836          | 0.766          | **0.884**      | *0.688*        | 0.829          | 0.720          |
| **Rank\***                         |                | 2              | 3              | 5              | 1              | 7              | 4              | 6              |

### Table 6: F1 score from a simple cross validation run (’*’ for z-score normalized samples)

| **Dataset**      | **RAC**        | **NC**         | **KNN**        | **SVM**        | **GNB**        | **RF**         | **XGB**        |
|------------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| LC1              | 0.925          | 0.925          | *0.908*        | 0.950          | 0.933          | 0.958          | **0.975**      |
| LC1*             | 0.925          | 0.925          | *0.908*        | 0.958          | 0.933          | 0.958          | **0.975**      |
| LC2              | 0.985          | 0.985          | **1.000**      | **1.000**      | 0.985          | **1.000**      | *0.861*        |
| LC2*             | 0.985          | 0.985          | **1.000**      | **1.000**      | 0.985          | **1.000**      | *0.923*        |
| PSO1             | 0.916          | *0.884*        | 0.950          | **0.975**      | 0.922          | **0.975**      | 0.962          |
| PSO1*            | 0.916          | *0.884*        | 0.950          | **0.975**      | 0.963          | **0.975**      | 0.950          |
| PSO2             | **1.000**      | **1.000**      | **1.000**      | **1.000**      | **1.000**      | **1.000**      | **1.000**      |
| PSO2*            | **1.000**      | **1.000**      | **1.000**      | **1.000**      | **1.000**      | **1.000**      | **1.000**      |
| SK1              | 0.801          | *0.776*        | 0.869          | **0.874**      | 0.795          | 0.788          | 0.781          |
| SK1*             | 0.801          | *0.776*        | 0.850          | **0.890**      | 0.781          | 0.811          | 0.794          |
| SK2              | 0.938          | 0.929          | 0.938          | **0.955**      | *0.904*        | 0.936          | 0.927          |
| SK2*             | 0.938          | 0.929          | 0.930          | **0.964**      | *0.911*        | 0.936          | 0.918          |
| LK1              | 0.861          | *0.819*        | 0.956          | **0.993**      | 0.940          | 0.971          | 0.928          |
| LK1*             | 0.861          | *0.785*        | 0.941          | **0.993**      | 0.921          | 0.964          | 0.971          |
| LK2              | 0.857          | *0.787*        | 0.914          | **1.000**      | 0.900          | 0.964          | 0.909          |
| LK2*             | 0.857          | *0.762*        | 0.942          | **1.000**      | 0.942          | 0.957          | 0.935          |
| AD1              | 0.706          | *0.684*        | 0.750          | **0.830**      | 0.685          | 0.741          | 0.816          |
| AD1*             | 0.706          | *0.684*        | 0.750          | **0.834**      | 0.685          | 0.730          | 0.816          |
| AD2              | 0.625          | *0.621*        | 0.658          | **0.712**      | 0.633          | 0.694          | 0.711          |
| AD2*             | 0.625          | *0.617*        | 0.662          | **0.713**      | 0.633          | 0.691          | 0.696          |
| AD3              | 0.865          | *0.823*        | 0.889          | **0.957**      | 0.844          | 0.904          | 0.914          |
| AD3*             | *0.865*        | 0.874          | 0.901          | **0.946**      | 0.882          | 0.913          | 0.925          |
| AD4              | 0.879          | 0.874          | 0.874          | **0.913**      | *0.866*        | 0.909          | 0.896          |
| AD4*             | *0.879*        | *0.879*        | 0.892          | **0.913**      | 0.883          | **0.913**      | 0.904          |
| PD1              | 0.782          | 0.750          | 0.840          | 0.903          | *0.635*        | 0.885          | **0.924**      |
| PD1*             | 0.782          | *0.764*        | 0.827          | **0.920**      | 0.792          | 0.879          | 0.906          |
| PD2              | **1.000**      | *0.934*        | **1.000**      | **1.000**      | **1.000**      | **1.000**      | **1.000**      |
| PD2*             | **1.000**      | *0.988*        | **1.000**      | **1.000**      | **1.000**      | **1.000**      | **1.000**      |
| **Mean**         | 0.867          | *0.842*        | 0.896          | **0.933**      | 0.860          | 0.909          | 0.900          |
| **Rank**         | 5              | 7              | 4              | 1              | 6              | 2              | 3              |
| **Mean\***       | 0.867          | *0.847*        | 0.897          | **0.936**      | 0.879          | 0.909          | 0.908          |
| **Rank\***       | 6              | 7              | 4              | 1              | 5              | 2              | 3              |

### Table 7: Number of genes in the literature related to Psoriasis among the 5 genes selected by RFE

| **Dataset**      | **RAC**                    | **SVM**                    | **RF**                    | **XGB**                    |
|------------------|----------------------------|----------------------------|---------------------------|----------------------------|
| PSO1             | 5                          | 5                          | 2                         | 2                          |
| PSO2             | 5                          | 5                          | 4                         | 1                          |
| PSO1+PSO2        | 5                          | 5                          | 4                         | 3                          |

### Table 8: Genes selected by RFE which have also been documented in the literature

| **Gene symbol** | **Description** | **Dataset (Classifier)** | **Reference** |
|-----------------|-----------------|--------------------------|---------------|
| DEFB4A/B        | Defensin beta 4A/B | PSO1+PSO2 (RAC, SVM, XGB), PSO1 (SVM), PSO2 (RAC, SVM) | 1, 2 |
| PI3             | Peptidase inhibitor 3  | PSO1+PSO2 (SVM, RF), PSO2 (RAC, SVM) | 1, 2 |
| SERPINB4        | Serpin family B member 4  | PSO1+PSO2 (SVM, XGB), SPO2 (SVM)  | 1, 2 |
| JUN             | Jun proto-oncogene, AP-1 transcription factor subunit  | PSO1+PSO2 (XGB), PSO1 (XGB) | 3 |
| S100A7A         | S100 calcium binding protein A7A | PSO1+PSO2 (SVM), PSO2 (SVM) | 1 |
| IL36G           | Interleukin 36 gamma | PSO1+PSO2 (RAC) | 1 |
| LCE3D           | Late cornified envelope 3D | PSO1+PSO2 (RAC) | 1 |
| GPR15LG         | G protein-coupled receptor 15 ligand | PSO1+PSO2 (RAC) | 4 |
| KRT16           | Keratin 16 | PSO1+PSO2 (RF) | 1, 2 |
| FABP5           | Fatty acid binding protein 5 | PSO1+PSO2 (RF) | 1 |
| LTF             | Lactotransferrin | 5 |
| STC1            | Stanniocalcin 1 | PSO1 (RAC) | 6 |
| S100A12         | S100 calcium binding protein A12 | PSO1 (RAC) | 1, 2 |
| IGFL1           | IGF like family member 1 | PSO1 (RAC) | 7 |
| OASL            | 2’-5’-oligoadenylate synthetase like | PSO1 (RAC) | 8 |
| IGHV4-31        | Immunoglobulin heavy variable 4-31 | 9 |
| S100A7          | S100 calcium binding protein A7 | PSO1 (SVM) | 1, 2 |
| RGS1            | Regulator of G protein signaling 1 | PSO1 (SVM) | 2 |
| HLA-DQA1        | Major histocompatibility complex, class II, DQ alpha 1 | PSO1 (SVM) | 10 |
| FERMT1          | FERM domain containing kindlin 1 | PSO1 (RF) | 11 |
| CD59            | CD59 molecule (CD59 blood group) | PSO1 (RF) | 12 |
| ZCCHC10         | Zinc finger CCHC-type containing 10 | PSO1 (XGB) | 13 |
| KRT77           | Keratin 77 | PSO2 (RAC) | 1 |
| IL37            | Interleukin 37 | PSO2 (RAC) | 14 |
| CKMT1A/B        | Creatine kinase, mitochondrial 1A/B | PSO2 (RF) | 15 |
| S100A9          | S100 calcium binding protein A9 | PSO2 (RF) | 1, 2 |
| MPZL2           | Myelin protein zero like | PSO2 (RF) | 2 |
| HMOX2           | Heme oxygenase 2 | PSO2 (RF) | 16 |
| PDZK1IP1        | PDZK1 interacting protein 1 | PSO2 (XGB) | 17 |

**References**

1. Rioux Genevieve, Ridha Zainab, Simard Melissa, Turgeon Florence, Guerin Sylvain L., Pouliot Roxane, **Transcriptome profiling analyses in psoriasis: A dynamic contribution of keratinocytes to the pathogenesis**
2. Sevimoglu Tuba, Arga Kazim Yalcin, **Computational systems biology of psoriasis: Are we ready for the age of omics and systems biomarkers?**
3. Zenz Rainer, Wagner Erwin F., **Jun signalling in the epidermis: From developmental defects to psoriasis and skin tumors**
4. Chen Caifeng, Wu Na, Duan Qiqi, Yang Huizi, Wang Xin, Yang Peiwen, Zhang Mengdi, Liu Jiankang, Liu Zhi, Shao Yongping et al., **C10orf99 contributes to the development of psoriasis by promoting the proliferation of keratinocytes**
5. Xie Shaoqiong, Chen Zhongjian, Wang Qingliang, Song Xun, Zhang Linglin, **Comparisons of gene expression in normal, lesional, and non-lesional psoriatic skin using DNA microarray techniques**
6. Ezure Tomonobu, Amano Satoshi, **Stanniocalcin-1 mediates negative regulatory action of epidermal layer on expression of matrix-related genes in dermal fibroblasts**
7. Lobito Adrian A., Ramani Sree R., Tom Irene, Bazan J. Fernando, Luis Elizabeth, Fairbrother Wayne J., Ouyang Wenjun, Gonzalez Lino C., **Murine insulin growth factor-like (IGFL) and human IGFL1 proteins are induced in inflammatory skin conditions and bind to a novel tumor necrosis factor receptor family member, IGFLR1**
8. Huang Yan-Zhou, Zheng Yu-Xin, Zhou Yuan, Xu Fan, Cui Ying-Zhe, Chen Xue-Yan, Wang Zhao-Yuan, Yan Bing-Xi, Zheng Min, Man Xiao-Yong, **OAS1, OAS2, and OAS3 Contribute to Epidermal Keratinocyte Proliferation by Regulating Cell Cycle and Augmenting IFN-1--Induced Jak1--Signal Transducer and Activator of Transcription 1 Phosphorylation in Psoriasis**
9. Chang Hsin-Wen, Yan Di, Singh Rasnik, Bui Audrey, Lee Kristina, Truong Alexa, Milush Jeffrey M., Somsouk Ma, Liao Wilson, **Multiomic Analysis of the Gut Microbiome in Psoriasis Reveals Distinct Host--Microbe Associations**
10. Zhou Xingchen, He Yijing, Kuang Yehong, Chen Wangqing, Zhu Wu, **HLA-DQA1 and DQB1 Alleles are Associated with Acitretin Response in Patients with Psoriasis**
11. Zhou Xue, Chen Youdong, Cui Lian, Shi Yuling, Guo Chunyuan, **Advances in the pathogenesis of psoriasis: From keratinocyte perspective**
12. Venneker G. T., Asghar S. S., **CD59: a molecule involved in antigen presentation as well as downregulation of membrane attack complex**
13. Lowes Michelle A., Suarez-Farinas Mayte, Krueger James G., **Immunology of psoriasis**
14. Teng Xiu, Hu Zhonglan, Wei Xiaoqiong, Wang Zhen, Guan Ting, Liu Ning, Liu Xiao, Ye Ning, Deng Guohua, Luo Can et al., **IL-37 ameliorates the inflammatory process in psoriasis by suppressing proinflammatory cytokine production**
15. Sobolev V.V.,  Mezentsev A.V., Ziganshin R.H., Soboleva A.G., Denieva M., Korsunskaya I.M., Svitich O.A., **LC-MS/MS analysis of lesional and normally looking psoriatic skin reveals significant changes in protein metabolism and RNA processing**
16. Hanselmann Christine, Mauch Cornelia, Werner Sabine, **Haem oxygenase-1: a novel player in cutaneous wound repair and psoriasis?**
17. Garcia-Heredia, Jose M., Carnero Amancio, **The cargo protein MAP17 (PDZK1IP1) regulates the immune microenvironment**