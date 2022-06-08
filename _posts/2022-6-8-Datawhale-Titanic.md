---
title: Datawhale
category: 数据分析
---

---------------------------
## 目录

- [1、数据集](#1数据集)
- [2、数据分析及预处理](#2数据分析及预处理)
    - [2.1 加载数据](#21-加载数据)
    - [2.2 数据探索](#22-数据探索)
    - [2.3 数据清洗](#23-数据清洗)
        - [2.3.1 缺失值处理](#231-缺失值处理)
        - [2.3.2 特征分布变化](#232-特征分布变换)
        - [2.3.3 特征重构](#233-特征重构)
- [3、建立模型](#3建立模型)
- [4、总结](#4总结)
---------------------------


## ***1、数据集*** 
&emsp;&emsp;数据选自Kaggle的泰坦尼克号数据集。该数据集记录了每个乘客的票价、船舱等级、年龄和性别等信息。目的是通过探索并建立机器学习模型预测每一位乘客的是否存活。 数据集从 https://www.kaggle.com/c/titanic/overview 下载。下载解压后得到891条训练集和418条测试集。我们的目的是通过各种数据分析的探索数据本身存在的规律，并利用训练集训练一个机器学习模型来预测乘客的是否存活。

## ***2、数据分析及预处理*** 
### ***2.1 加载数据*** 
&emsp;&emsp;泰坦尼克号数据以csv文件的形式给出。利用pandas库，可以很方便加载并处理csv文件。
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
rc = {'font.sans-serif': 'SimHei'}
sns.set(rc=rc)

df = pd.read_csv('titanic\\train.csv')

# 该数据集相对比较小，因此可以直接将其全部加载到内存中。有时候我们遇到的数据集相对比较大，无法将其完全加载至内存中时，可以通过read_csv函数的chunksize参数来指定每次加载的样本数

# df = pd.read_csv('titanic\\train.csv', chunksize=1000)
```
得到的数据后我们需要查看每个特征的数据类型、是否有缺失值、取值范围等基本信息。利用DataFrame类提供的方法可以轻松做到这一点。

```
# 查看数据大小
df.shape

# 每一特征的取值类型
df.info()

# 每一特征取值范围
df.describe()

# 是否有缺失值
df.isnull().sum()

# 是否有重复数据
df.duplicated()
```

&emsp;&emsp;通过上述代码我们可以知道其中“Age”、“Cabin”和“Embarked”这三个特征数据有缺失。因此，在建立机器学习模型之前，我们需要对缺失值进行特殊处理。有时候由于各种各样的原因，我们的数据集中总是会存在一些重复的数据，这些重复数据的存在不会对我们模型性能的提高有太大的帮助，相反当数据集较小时，这些重复的数据会使得我们的模型过多的关注这些重复数据，因此在建立模型时需要删除这些重复数据。

### ***2.2 数据探索*** 
&emsp;&emsp;数据探索的目的不外乎观察各个变量之间的关系，为后续的模型挑选合适的建模变量。在这里，个人将数据探索划分为两种类型：1）单变量的数据分布；2）变量之间的相关关系。

```
# 查看各变量之间的相互关系
# 修改列索引
new_columes = {
'PassengerId': '乘客ID',
'Survived': '是否幸存',
'Pclass': '乘客等级(1/2/3等舱位)',
'Name': '乘客姓名',
'Sex': '性别',
'Age': '年龄',
'SibSp': '堂兄弟/妹个数',
'Parch': '父母与小孩个数',
'Ticket': '船票信息',
'Fare': '票价',
'Cabin': '客舱',
'Embarked': '登船港口',
}

rename_func = lambda x: new_columes[x] if x in new_columes.keys() else x

df.columns = df.columns.map(rename_func)

corrmat = df.corr()
fig, ax = plt.subplots(1)
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()
```

<center>
<img src=https://cdn.jsdelivr.net/gh/wtyhainan/blog-img@main/titanic/corr.png>
</center>
<center>图1 特征关系图</center></br>


从特征之间的相关关系中可以看到，除了票价与客舱等级之间存在较强的线性关系之外，其余特征之间的相关性并不是特别强。在建模之前观察各变量之间的关系有助于我们剔除含有较强相关性的特征，保留一些相关性不大的特征，避免模型遭遇维度灾难。当然，更实际的做法是通过机器学习模型来完成特征筛选。

除了观察特征变量之间的相互关系，我们还可以观察特征变量与目标变量之间的关系。下图展式了性别与存活率之间的关系。

```
sex_survived = df.groupby(['Sex'])['Survived'].value_counts() / df.groupby(['Sex'])['Sex'].count()

sex_survived.unstack().plot(kind='bar')
```
<center>
<img src=https://cdn.jsdelivr.net/gh/wtyhainan/blog-img@main/titanic/sex-survived.png>
</center>

<center>图2 性别与存活率的关系</center></br>

从上图中可知，女性存活率达到了74%，远高于男性的19%。
根据特征关系图可知，存活率与客舱等级存在较强的相关关系，而客舱等级又与票价存在较强的相关性。下图客舱等级、票价及存活率之间的关系。
```
# 客舱等级与存活率之间的关系
pclass_survived = df.groupby(['Pclass', 'Survived'])['Survived'].count() / df.groupby(['Pclass'])['Survived'].count()
pclass_survived.unstack().plot(kind='bar', stacked='True')

# 票价与存活率之间的关系
fare_survived = (df.groupby(['Fare'])['Survived'].sum() / df.groupby(['Fare'])['Survived'].count()).sort_values(ascending=False)
survived_rate = fare_survived.to_numpy()
fare = fare_survived.index.to_numpy()
x = np.arange(0, len(survived_rate), 1)
fig, ax = plt.subplots(1)
ax.plot(x, survived_rate, 'ro')
for i in range(len(survived_rate)):
    xy = (x[i], survived_rate[i])
    s = str((fare[i]))
    ax.annotate(s, xy)
ax.set_xticks([])
ax.set_ylabel('存活率')

# 票价与客舱等级的关系
pclass_fare = df.groupby(['Pclass'])['Fare'].agg(['mean', 'std', 'median', 'max', 'min'])

#             mean        std   median       max  min
# Pclass                                              
# 1       84.154687  78.380373  60.2875  512.3292  0.0
# 2       20.662183  13.417399  14.2500   73.5000  0.0
# 3       13.675550  11.778142   8.0500   69.5500  0.0
```
<center>
<img src=https://cdn.jsdelivr.net/gh/wtyhainan/blog-img@main/titanic/pclass-fare-surviced.jpg>
</center>
<center>图3 pclass，fare与存活率之间的关系</center></br>

上面的数据告诉我们，不同客舱等级的存活率不同。但由于女性与男性之间的存活率存在较大的差异，在这里我们并不知道不同客舱存活率的差异是否由客舱男女比比例不同引起的，在这里我们需要进一步挖掘不同客舱的男女比例。
```
pclass_sex = df.groupby(['Pclass', 'Sex'])['Sex'].count()
pclass_sex.unstack().plot(kind='bar', stacked='True')
```
<center>
<img src=https://cdn.jsdelivr.net/gh/wtyhainan/blog-img@main/titanic/pclass-sex.png>
</center>
<center>图4 不同客舱的性别比例</center></br>

结合不同客舱性别比例及存活率，可知不同客舱的存活率的差异性并不完全由性别造成。

&emsp;&emsp;根据上面的分析我们得出如下结论：

1）女性存活率显著高于男性1等客舱存活率最高，2等级次之，3等级最低；

2）1等客舱存活率最高，2等级次之，3等级最低；

3）票价与存活率之间并无明显的关系，可能是不同客舱的票价的差异性较大引起；

4）不同客舱存活率的差异并不完全是由性别比例不同造成；

### ***2.3 数据清洗*** 
&emsp;&emsp;在实际应用中，由于这样那样的原因，我们需要建模的数据会存在缺失值、异常值等。在建模之前我们需要对这些异常数据做诸如填充、剔除等操作。目的是减少这些异常数据对模型性能的影响。

#### ***2.3.1 缺失值处理*** 
&emsp;&emsp;数据缺失是我们建模过程中不可避免的问题。对于数据缺失，我们应该知道“数据为什么缺失”，可能的原因有很多中，总的来说有以下三大类：1）无意的数据缺失。在做信息记录的过程中，由记录人员的工作疏忽造成信息缺失；2）有意的数据缺失。有些数据在特征描述中会规定将缺失值也作为一种特征，这时候缺失值就可以看作一种特殊的特征值；3）不存在。有些属性根本就不存在，比如一个未婚者的配偶名字就无法记录。

&emsp;&emsp;在对缺失值进行处理前，了解数据缺失的机制和形式是十分必要的。将数据集中不含缺失值的变量称为完全变量，数据集中含有缺失值的变量称为不完全变量。从缺失的分布来划分缺失类型，可分为三种：1）完全随机变量。指的是数据的缺失是完全随机的，不依赖于任何不完全变量和完全变量，不影响样本的无偏性；2）随机缺失。指数据的缺失不是完全随机的，即该类数据的缺失依赖于其他完全变量；3）非随机缺失。指数据的缺失与不完全变量自身的取值有关。

&emsp;&emsp;针对不同数据缺失的机制，采取不同的缺失值处理办法。随机缺失数据，我们可以利用其他变量对缺失值进行估计。完全随机缺失数据，由于该类缺失值不影响样本的无偏性，因此我们可以直接将该缺失值删除。对于非随机缺失数据，暂时没有很好的解决办法。

&emsp;&emsp;***删除***。直接去除含有缺失值的记录，这种处理方式是简单粗暴的，适用于数据量较大且缺失值比较小的情形。去掉后对总体影响不大。一般不建议这样做，因为很可能会造成数据丢失，数据偏移。

&emsp;&emsp;***填充***。填充是最常用的一种缺失值处理办法。通常会采用常量填充、统计值填充和模型填充。常量填充就是使用特定值填充缺失字段。统计值填充是使用特征分布的均值、中位数和众数等统计特征替代缺失值。模型填充就是利用完全数据训练一个机器学习模型，比如knn或者随机森林，然后使用该模型来预测缺失部分的取值。

#### ***2.3.2 特征分布变换***
&emsp;&emsp;许多机器学习模型要求变量分布要满足高斯分布（***其实不是很理解***），因此在建模之前需要将长尾数据分布通过对数变换、BOX-COX变换和平方变换等将其转换到近似正太分布。

#### ***2.3.3 特征重构***
&emsp;&emspl数据分箱就是按照某种规则将数据进行分类。一般在建立分类模型时，需要对连续变量离散化，特征离散化后，模型会更稳定，降低过拟合的风险。比如在建立申请信用卡模型时用logistic作为基模型就需要对连续变量进行离散化，离散化常用的方法就是分箱。分箱有以下重要意义及其优势：

&emsp;&emsp;1）离散特征的增加和减少都很容易，易于模型的快速迭代；

&emsp;&emsp;2）稀疏变量内积惩罚运算速度块，计算结果便于存储，容易扩展；

&emsp;&emsp;3）离散化后的特征对异常数据有很强的鲁棒性，比如一个特征是年龄>30是1，否则是0.如果特征没有离散化，一个异常数据“年龄=300”会给模型造成很大的干扰。

&emsp;&emsp;4）逻辑回归属于广义线性模型，表达能力有限；单变量离散化为N个后，每个变量有单独的权重，相当于为模型引入了非线性，能够提升模型表大能力，加大拟合；

&emsp;&emsp;5）特征离散化后，模型更稳定。比如如果对用户年龄离散化，20~30为一个区间，不会因为一个用户年龄增长一岁就变成一个完全不同的人。当然处于区间相邻的样本会刚好相反，所以怎么划分区间显得异常重要；

&emsp;&emsp;6）特征离散化后，起到了简化逻辑回归模型的作用，降低了模型过拟合的风险。可以将缺失值作为独立的一类特征带入模型。将所有变量变换到相似尺度上。

&emsp;&emsp;常用的分箱方法有：***1）卡方分箱***。自底向上的（即基于合并的）数据离散化放啊发。它依赖于卡方检验：具最小卡方值的相邻区间合并在一起，直到满足确定的停止准则。基本思想：对于精确的离散化，相对类频率在一个区间内应当完全一致。因此，如果两个相邻区间具有非常类似的类分布，则这两个区间可以合并；否则，它们应当保持分开。而第卡方值表明它们具有相似的类分布。***2）最小熵法分箱***。需要使总熵值达到最小，也就是使分箱能够最大限度地区分因变量的各类别。数据的熵值越低，说明数据之间的差异越小，最小熵划分就是为了使每箱中的数据具有最好的相似性。***3）无监督分箱，包括等距分箱和等频分箱***。

***注：以上内容摘自：https://www.jianshu.com/p/fbdce03302f6***

&emsp;&emsp;one-hot编码是将类别向量转换为机器学习算法易于利用的一种形式的过程。这样做的好处主要有：1）解决了分类器不好处理属性数据的问题；2）在一定程度上也起到了扩充数据的作用。通常使用one-hot编码来处理离散型数据。在回归、分类和聚类等机器学习算法中，特征之间距离的计算和相似度的计算是非常重要的，常用的距离或相似度的计算都是在欧式空间内完成，。使用one-hot编码可以将离散特征的取值扩展到欧式空间，离散特征的某个取值对应了欧式空间的某个点。

*woe编码属于有监督编码。对该编码理解程度不深，在此记下，后面有时间再研究。*

## ***3、建立模型***
&emsp;&emsp;经过上面的数据探索及处理阶段，最后我们需要对处理后的数据建立相应的模型用于预测人员存活率。机器学习算法根据是否需要标签数据可划分为监督学习和无监督学习，根据预测结果是离散还是连续的情况可划分为分类和回归。由此可知，泰坦尼克号数据需要选择一种监督学习模型来预测人员是否存活，这是一个分类问题。开始我们一般都会选择一个简单的机器学习模型来拟合数据，然后以此模型为基准改善模型或者数据来得到更好的结果。因此，对于小规模的数据集我们通常会选择KNN模型或者logistic回归当作基准模型。

```
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# train_clear_df_X和test_clear_df_X为干净数据（缺省数据被填充及无关特征如Name，PassengerId被删除）。
# 对数据做one-hot编码处理。
dataset = pd.concat([train_clear_df_X, test_clear_df_X], axis=0)
# 对Age做分箱？在这个小数据集上knn模型性能会略微下降，但是对svm没有影响。这是为什么呢？
dataset['Age'] = pd.cut(dataset['Age'], [-2, 0, 18, 28, 38, 48, 200], labels=['A', 'B', 'C', 'D', 'E', 'F'], )
dataset_onehot = pd.concat(
        [dataset[['SibSp', 'Fare']], pd.get_dummies(dataset[['Pclass', 'Sex', 'Parch', 'Cabin', 'Embarked', 'Age']])],
        axis=1)

trainX = dataset_onehot[0:len(train_clear_df_X)]
testX = dataset_onehot[len(train_clear_df_X):]

model = Pipeline([
        ('scale', RobustScaler()),  # 特征归一化
        # ('svm', svm.SVC())        # SVM模型
        ('knn', KNeighborsClassifier(n_neighbors=10))   # KNN模型
    ])
score = cross_val_score(model, trainX, trainy, cv=10)    # 5轮交叉验证
print('交叉验证得分：', score, np.mean(score))
model.fit(trainX, y=trainy)     # 训练
pred_y = model.predict(testX)   # 预测

print('验证集上的得分：\n', confusion_matrix(testy, y_pred=pred_y))
```


下图为经过预处理后的数据
<center>
<img src=https://cdn.jsdelivr.net/gh/wtyhainan/blog-img@main/titanic/clear_data.jpg>
</center>
<center>图5 预处理后的数据</center></br>

下图展示了knn及svm模型的预测结果
<center>
<img src=https://cdn.jsdelivr.net/gh/wtyhainan/blog-img@main/titanic/knn-result.jpg>
</center>
<center>图6 KNN结果</center></br>

<center><img src=https://cdn.jsdelivr.net/gh/wtyhainan/blog-img@main/titanic/svm-result.jpg></center>
<center>图7 SVM结果</center></br>

在训练模型之前，我们使用Scaler对特征数据做了归一化处理，这对于KNN及SVM这种需要计算距离的模型来说至关重要，因为原始特征的数据有时并不在同一个量级上，如果不做数据归一化处理，这会导致计算出来的距离仅决定于量级较大的特征，这实际上隐含了我们认为数据量级大的比数据量级小的特征更重要。然而，数据当中并未体现出某个特征相对于另一个特征对于模型有更显著的重要性。

## ***4、总结***
本次项目使用的是Kaggle上的泰坦尼克号存活率数据，我们的目的建立机器学习模型预测人员存活率。原始数据集中的多个特征存在缺省现象，因此在建立模型之前我们需要对缺失值进行处理。在实验中，测试了常用的若干种缺失值填充办法，由于数据集相对较小无法比较各个缺失值填充办法的有效性。在建立模型之前我们还探索了数据特征之间的相关性，并从中得出了几条结论（详见2.2小节）。最后我们使用KNN模型和SVM模型来预测人员存活率。从实验结果中得知KNN模型和SVM模型在训练集上的性能相当，在测试集上SVM模型比KNN模型要好。
