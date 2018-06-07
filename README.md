# Intro

- Titanic data set available at https://www.kaggle.com/c/titanic/data 

- Goal is to predict if a passenger survived the sinking of the Titanic

- The "accuracy" is the percentage of passengers correctly predicted



# Getting the data

- from Anaconda Prompt, used the kaggle API with the command `kaggle competitions download -c titanic`

```python
# data analysis and wrangling

import pandas as pd

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

```

# Preparing the data

```python
train_df.info()
```
![](IMG/Screenshot-2018-6-7%20Titanic.png)


