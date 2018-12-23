## 模型训练

其实这个实验的主要任务都是在之前数据清洗和预处理阶段，模型训练阶段只需要根据预处理后的数据选用相应的模型进行训练即可。

在之前预处理后得到的数据上，我们在训练过程中还直接将`encounter_id`、`patient_nbr`、`diag_2`、`diag_3`这四个特征列也删去了。

预处理后的数据中总共有 86556 个，其中每类样本数如下：

```
>30 readmmission 30708
<30 readmmission 9399
Never readmmission 46449
```

可以看到这三类样本分布并不是很均匀，所以在划分训练/测试数据之前，我们还采用了`smote`算法进行了采样（over-sampling），从而得到最终的训练/测试数据，最终采样后得到数据数为 39347 个，我们按照 $4 : 1$ 的比例进行训练/测试集的划分，最终训练数据集样本数有 11477 个，测试集样本数 27870 个。

我们最终采用了两种方法处理这个三分类问题，分别是**xgboost**和**Random Forest**算法，并进行了10-fold 交叉验证。实验结果如下：

### xgboost

```shell
Cross Validation score:  0.6083765322410007
Accuracy:  0.6102260495156082
Confusion matrix:
 [[7538  136 1584]
 [1348 6495 1494]
 [3913 2388 2974]]
Overall report：
               precision    recall  f1-score   support

           0       0.59      0.81      0.68      9258
           1       0.72      0.70      0.71      9337
           2       0.49      0.32      0.39      9275

   macro avg       0.60      0.61      0.59     27870
```

在测试集上的准确率是  61%，MACRO-F1的值是 0.59。

### Random Forest

```shell
Cross Validation Score:  0.6301298885238472
Accuracy:  0.6315034086831719
Confusion matrix:
 [[6858  160 2240]
 [1143 7057 1137]
 [3594 1996 3685]]
Overall report：
               precision    recall  f1-score   support

           0       0.59      0.74      0.66      9258
           1       0.77      0.76      0.76      9337
           2       0.52      0.40      0.45      9275

   macro avg       0.63      0.63      0.62     27870
```

在测试集上的准确率是  63%，MACRO-F1的值是 0.62。

同时，我们借助 scikit-learn 的`feature_importances_`对用于训练数据中的特征值重要性进行了排序，对于Random Forest的算法的结果如下：

![random-forest](./random-forest.jpg)

## 总结

在借助机器学习算法进行数据分析的过程中，往往数据清洗和预处理的工作更加关键与繁琐。

### 
