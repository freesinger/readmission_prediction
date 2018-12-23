# 程序运行说明

## 环境

python3.6

库：

- numpy
- pandas
- scipy
- imbalanced-learn
- XGBoost
- scikit-learn
- matplotlib

## 文件说明

`preprocess.py` : 用于数据预处理，运行后在 `data` 目录下生成预处理后的数据 `preprocessed_data.csv`

`train.py`: 用于训练与输出结果，采用了 `xgboost` 和 `Random Forest` 两种进行测试。运行后会在屏幕中输出相应的准确率、混淆矩阵等。

## 运行说明

在data目录下已经存放以前提前预处理好的数据 `preprocessed_data.csv`。可以直接运行`python train.py` 命令即可。

如果需要也执行预处理操作，运行 `python preprocess.py` 即可，运行后在 `data` 目录下生成预处理后的数据 `preprocessed_data.csv`。（需要将原来的数据文件`diabetic_data.csv` 存放到 `data` 目录下。）
