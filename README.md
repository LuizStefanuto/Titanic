# Intro

- Titanic data set available at https://www.kaggle.com/c/titanic/data 

- Goal is to predict if a passenger survived the sinking of the Titanic

- The "accuracy" is the percentage of passengers correctly predicted



# Getting the data

- from Anaconda Prompt, used the kaggle API with the command `kaggle competitions download -c titanic`
- created two dataframes with the pandas package

```python

import pandas as pd

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

```

# Preparing the data

- check datatypes and missing values

```python
train_df.info()
```
![](IMG/Screenshot-2018-6-7%20Titanic.png)

- drop "Cabin" feature since it is mostly incomplete
- "Name", "Ticket" and "PassangerId" features can be droped since they may not contribute to the analysis

```python

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

```
- extract titles from "Name"
```python


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

```

- Complete "Age" feature
- Complete "Embarked" feature
- create "Family" feature with "SibSp" and "Parch", to get a total count of family members
- turn "Age" into age bands
- create a "Fare" range

# Analysing the data

- The assumption that women, children and upper-class passengers were more likely to have survived can be added to the analysis
- Pivoting features
```python
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

```
- Analysing visualy
- a histogram chart is useful to analyse continuous numerical data

```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
```
