# Readmission Prediction
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![size](https://img.shields.io/github/repo-size/freesinger/Readmission_Prediction.svg?style=plastic)
![stars](https://img.shields.io/github/stars/freesinger/Readmission_Prediction.svg?style=social)
## 1. Environments

`python3.6`

## 2. Libraries
```
- numpy
- pandas
- scipy
- imbalanced-learn
- XGBoost
- scikit-learn
- matplotlib
```

## 3. Files

`preprocess.py`: used for preprocessing data, generate the processed data file `preprocessed_data.csv` which saved in folder `data`.

`train.py`: used for training and output, test models are `XGBoost` and `Random Forest`. Accuracy, confusion matrix and overall report of models will shown after running.

## 4. Run

```
>_ python3.6 preprocess.py
>_ python3.6 train.py
```
