<div align="center">

# COVID-19 Outcome Classification

### Comparing classic machine-learning models on a historical patient dataset

<p>
  <a href="ML_Project.ipynb">Explore the Jupyter notebook</a> ·
  <a href="data.csv">View the dataset</a> ·
  <a href="project_375_381.pdf">Read the project brief</a>
</p>

</div>

---

## Overview

This project explores whether a recorded COVID-19 case was classified as **recovered or deceased** using demographic, location, exposure, symptom, and timing features. It walks through data preparation, model selection, and evaluation, then compares five classic binary classifiers.

The notebook’s best reported test-set result is an **SVM F1 score of 0.85** for the positive (death) class. This is a small educational experiment on a historical dataset—not a clinical tool, a current epidemiological analysis, or evidence that these features can predict an individual’s medical outcome.

## What’s inside

| File | Description |
| --- | --- |
| [`ML_Project.ipynb`](ML_Project.ipynb) | End-to-end analysis, preprocessing, model fitting, evaluation, and plots |
| [`data.csv`](data.csv) | Tabular dataset loaded by the notebook |
| [`project_375_381.pdf`](project_375_381.pdf) | Project brief |

## Workflow

1. **Load the data** with pandas. The notebook assigns 14 columns: 13 input features and the `result` target.
2. **Create train, validation, and test partitions** using a randomized 70% / 15% / 15% split with `random_state=44`.
3. **Scale features** with `StandardScaler` for models that benefit from standardized inputs. The scaler is fit on the training partition and applied to validation and test data.
4. **Tune and compare classifiers** using validation F1 or stratified 10-fold cross-validation, as appropriate.
5. **Evaluate on the held-out test partition** with F1, classification reports, and ROC plots.

### Models compared

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

## Results

The notebook reports the following **positive-class test F1 scores** from its saved run:

| Model | Test F1 |
| --- | ---: |
| KNN | 0.70 |
| Logistic Regression | 0.82 |
| Gaussian Naive Bayes | 0.71 |
| Decision Tree | 0.80 |
| **SVM** | **0.85** |

For the SVM, the saved classification report shows **0.73 precision and 1.00 recall** for class `1`, with **0.97 accuracy** overall. The test set contains 130 records, including only 11 positive cases, so the results—especially for the minority class—should be interpreted cautiously. These are outputs from the notebook’s recorded run; results may vary with dependency versions or changes to the data and split.

## Run the notebook

### Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- pandas
- NumPy
- Matplotlib
- scikit-learn

Install the packages with:

```bash
python -m pip install jupyter pandas numpy matplotlib scikit-learn
```

### Steps

```bash
git clone https://github.com/ElecSpartan/MachineLearningProject.git
cd MachineLearningProject
jupyter notebook ML_Project.ipynb
```

Run the notebook cells from top to bottom. Keep `data.csv` in the repository root, because the notebook reads it using the relative path `data.csv`.

## Dataset and interpretation notes

The notebook describes the data as historical COVID-19 case information beginning in January 2020. Its target is encoded as `result` (`0` = recovered, `1` = death). Inputs include location and country codes, gender, age, Wuhan-related exposure indicators, six symptom fields, and the difference in time before hospitalization (`diff_sym_hos`). Consult the dataset and project brief for the available data context; the notebook does not provide a complete data dictionary or provenance details for every field.

The classes are imbalanced, and the held-out test set has very few positive examples. Accuracy alone can therefore hide important errors. The notebook uses F1 and class-level precision/recall to make the comparison more informative, but its single split is not enough to establish robust generalization. The dataset is historical and should not be used to guide diagnosis, treatment, triage, or other real-world health decisions.

## Reproducibility

The initial train/validation/test split uses seed `44`; the stratified cross-validation runs use seed `42`. The notebook includes saved outputs, which are useful as a reference, but execute the cells to reproduce the analysis in your environment.

---

<div align="center">

**A practical, notebook-based comparison of classical classification methods.**

</div>
