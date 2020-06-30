```python
import os
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import resample
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
```

# Data Reading

## Read Data


```python
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'status', 'occupation', 'relationship',
       'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income-class']

df = pd.read_csv('adult.data', names=cols)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income-class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>



## Overall Inspection


```python
# check data types of each column
df.dtypes
```




    age                int64
    workclass         object
    fnlwgt             int64
    education         object
    education-num      int64
    status            object
    occupation        object
    relationship      object
    race              object
    sex               object
    capital-gain       int64
    capital-loss       int64
    hours-per-week     int64
    native-country    object
    income-class      object
    dtype: object




```python
# take a look at a obejct-type value in the table.
# it's obvious that there are leading whitespaces in it, which can be annoying later on.
df.loc[0, 'income-class']
```




    ' <=50K'




```python
# remove leading and trailing whitespaces in values that have strings
string_df = df.select_dtypes(include='object')
df[string_df.columns] = string_df.apply(lambda x: x.str.strip())
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>income-class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check if there are missing values
df.isnull().sum()
```




    age               0
    workclass         0
    fnlwgt            0
    education         0
    education-num     0
    status            0
    occupation        0
    relationship      0
    race              0
    sex               0
    capital-gain      0
    capital-loss      0
    hours-per-week    0
    native-country    0
    income-class      0
    dtype: int64



# Descriptive Analysis (Univariate)

## Numeric Variables


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>32561.000000</td>
      <td>3.256100e+04</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
      <td>32561.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.581647</td>
      <td>1.897784e+05</td>
      <td>10.080679</td>
      <td>1077.648844</td>
      <td>87.303830</td>
      <td>40.437456</td>
    </tr>
    <tr>
      <th>std</th>
      <td>13.640433</td>
      <td>1.055500e+05</td>
      <td>2.572720</td>
      <td>7385.292085</td>
      <td>402.960219</td>
      <td>12.347429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>1.228500e+04</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>28.000000</td>
      <td>1.178270e+05</td>
      <td>9.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.000000</td>
      <td>1.783560e+05</td>
      <td>10.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>2.370510e+05</td>
      <td>12.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>90.000000</td>
      <td>1.484705e+06</td>
      <td>16.000000</td>
      <td>99999.000000</td>
      <td>4356.000000</td>
      <td>99.000000</td>
    </tr>
  </tbody>
</table>
</div>



From the summary above, we have several findings:
1. The majority of data in capital-gain and capital-loss is 0. 
2. 99999 in capital-gain looks suspicious, it might be a default value when there's no entry of data.
3. Minimum hours-per-week is 1 hour, and maximum hours-per-week is 99 hours. Both can be outliers.

In order to identify outliers in these numeric columns, we need to draw histograms and boxplots to check their distributions. To save space on this page, the histograms are not printed.


```python
# histogram and boxplot for numeric columns
# num_df = df.select_dtypes(include=np.number)
# num_cols = num_df.shape[1]
# plot_height = 3
# plot_width = 8
# fig, axes = plt.subplots(nrows=num_cols, sharey=True, figsize=(plot_width, num_cols*plot_height))

# for idx, col in enumerate(num_df.columns):
#     ax = axes[idx]
#     sns.distplot(num_df[col], kde=False , ax=ax)
#     #ax.hist(num_df[col], bins=20)
#     ax.set_title('Histogram of %s' % col, fontweight='bold')
    
# plt.tight_layout()
```


```python
fig, axes = plt.subplots(nrows=num_cols, sharey=True, figsize=(plot_width, num_cols*plot_height))

for idx, col in enumerate(num_df.columns):
    ax = axes[idx]
    sns.boxplot(num_df[col], ax=ax)
    ax.set_title('Boxplot of %s' % col, fontweight='bold')

plt.tight_layout()
```


![png](output_15_0.png)


The plots above suggest that:
1. The distribution of all numeric columns are very skewed.
2. All numeric columns have data points that fall out of normal ranges.

Since we don't have more detailed information about the dataset, we want to be conservative at this point and don't take any actions on the outliers suggested by the boxplots.  
However, we can process the two special columns: capital-gain and capital-loss where most values are zero. To not lose information that is potentially useful, we create new columns that indicate whether the values are zero or not instead of removing abnormal values or imputing them with zero, and remove column capital-gain and column capital-loss.


```python
# create two indicator columns
df['capital-gain-zero'] = 1
df.loc[df['capital-gain'] > 0, 'capital-gain-zero'] = 0
df['capital-loss-zero'] = 1
df.loc[df['capital-loss'] > 0, 'capital-loss-zero'] = 0
```


```python
df['capital-gain-zero'].value_counts()
```




    1    29849
    0     2712
    Name: capital-gain-zero, dtype: int64




```python
df['capital-loss-zero'].value_counts()
```




    1    31042
    0     1519
    Name: capital-loss-zero, dtype: int64



## Categorical Variables


```python
# categorical columns
cat_df = df.select_dtypes(include='object')
cat_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>native-country</th>
      <th>income-class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>State-gov</td>
      <td>Bachelors</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Self-emp-not-inc</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Private</td>
      <td>11th</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Private</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
num_cols = cat_df.shape[1]
plot_height = 6
plot_width = 10
#nrows = math.ceil(num_cols/2)
fig, axes = plt.subplots(nrows=num_cols, ncols=1, sharey=True, figsize=(plot_width, num_cols*plot_height))

for idx, col in enumerate(cat_df.columns):
#     row_idx = idx // 2
#     col_idx = idx % 2
    vc = cat_df[col].value_counts()/len(df) * 100
    # ax = axes[row_idx][col_idx]
    ax = axes[idx]
    sns.barplot(x=vc.index, y=vc.values, ax=ax)
    ax.tick_params(labelrotation=45)
    ax.set_title('Value counts of %s' % col, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('percentage')
plt.tight_layout()
```


![png](output_22_0.png)



```python
# the bar plot of native country looks messy so we print out value counts here
cat_df['native-country'].value_counts()
```




     United-States                 29170
     Mexico                          643
     ?                               583
     Philippines                     198
     Germany                         137
     Canada                          121
     Puerto-Rico                     114
     El-Salvador                     106
     India                           100
     Cuba                             95
     England                          90
     Jamaica                          81
     South                            80
     China                            75
     Italy                            73
     Dominican-Republic               70
     Vietnam                          67
     Guatemala                        64
     Japan                            62
     Poland                           60
     Columbia                         59
     Taiwan                           51
     Haiti                            44
     Iran                             43
     Portugal                         37
     Nicaragua                        34
     Peru                             31
     France                           29
     Greece                           29
     Ecuador                          28
     Ireland                          24
     Hong                             20
     Cambodia                         19
     Trinadad&Tobago                  19
     Laos                             18
     Thailand                         18
     Yugoslavia                       16
     Outlying-US(Guam-USVI-etc)       14
     Honduras                         13
     Hungary                          13
     Scotland                         12
     Holand-Netherlands                1
    Name: native-country, dtype: int64



The bar plots above suggest:
1. Heavily imbalanced distributions exist in:
    - work class where "private" is the dominant value
    - race where "white" is the dominant value
    - sex where "male" is the dominant value
    - native country where "United States" is the dominant value
    - income-class where "<=50K" is the dominant value, meaning the target variable is imbalanced

2. Column education and column ducation-num contain the same information
3. Column native-country has a high cardinality
4. "Never-worked" and "Without-pay" in column workclass could be outliers

Therefore, we want to map column education to column education-num and keep only one of them, in this case, education-num as it's already in a numeric format.  
In addition, we will encode "United States" in column native-country as 1 and all the other countries as 0 to handle the high cardinality, and remove column native-country.  
We also need to investigate what income-class that "Never-worked" and "Without-pay" map to.


```python
# Maps education to education-num
education = df[['education', 'education-num']]
education_dict = {}
for val in df['education'].unique():
    num = education[education['education']==val]['education-num'].unique()
    education_dict[num[0]] = val

education_dict
```




    {13: 'Bachelors',
     9: 'HS-grad',
     7: '11th',
     14: 'Masters',
     5: '9th',
     10: 'Some-college',
     12: 'Assoc-acdm',
     11: 'Assoc-voc',
     4: '7th-8th',
     16: 'Doctorate',
     15: 'Prof-school',
     3: '5th-6th',
     6: '10th',
     2: '1st-4th',
     1: 'Preschool',
     8: '12th'}




```python
df['is_USA'] = 1
df.loc[df['native-country'] != 'United-States', 'is_USA'] = 0
df = df.drop(['capital-gain', 'capital-loss', 'native-country', 'education'], axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>hours-per-week</th>
      <th>income-class</th>
      <th>capital-gain-zero</th>
      <th>capital-loss-zero</th>
      <th>is_USA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>40</td>
      <td>&lt;=50K</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>13</td>
      <td>&lt;=50K</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>40</td>
      <td>&lt;=50K</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>40</td>
      <td>&lt;=50K</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>40</td>
      <td>&lt;=50K</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# How much do "Never-worked" and "Without-pay" earn?
no_pay_df = df.loc[df['workclass'].isin(['Never-worked', 'Without-pay']), ['workclass', 'income-class']]
```


```python
no_pay_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>income-class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1901</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>5361</th>
      <td>Never-worked</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>9257</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>10845</th>
      <td>Never-worked</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>14772</th>
      <td>Never-worked</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>15533</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>15695</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>16812</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>20073</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>20337</th>
      <td>Never-worked</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>21944</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>22215</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>23232</th>
      <td>Never-worked</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>24596</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>25500</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>27747</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>28829</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>29158</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32262</th>
      <td>Without-pay</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32304</th>
      <td>Never-worked</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>32314</th>
      <td>Never-worked</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>



From the table above, we can see that there are very few instances in data whose workclass are "Without-pay" or "Never-worked". All these instances earn less than 50K dollars.  
Thus, we create a dummy variable "no_pay_or_work" to indicate if a workclass is no pay or not.


```python
df['no_pay_or_work'] = 0
df.loc[df['workclass'].isin(['Never-worked', 'Without-pay']), 'no_pay_or_work'] = 1
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>hours-per-week</th>
      <th>income-class</th>
      <th>capital-gain-zero</th>
      <th>capital-loss-zero</th>
      <th>is_USA</th>
      <th>no_pay_or_work</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>40</td>
      <td>&lt;=50K</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>13</td>
      <td>&lt;=50K</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>40</td>
      <td>&lt;=50K</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>40</td>
      <td>&lt;=50K</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>40</td>
      <td>&lt;=50K</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Correlation Analysis (Bivariate)

Part 1: Qualitative Analysis
1. Qualitative correlation between two categorical variables
    - Contingency table  
2. Qualitative correlation between a categorical variable and a numeric variable
    - Histogram of the numeric variable per unique value of the categorical variable 
    
Part 2: Quantitative Analysis
1. Quantitative correlation between two categorical variables
    - Chi-square
    - Mutual information
2. Quantitative correlation between a categorical variable and a numeric variable
    - Student T-test
3. Quantitative correlation between two numeric variables
    - Pearson Correlation

## Qualitative Analysis

### Betweeen Categorical Variables


```python
# Investigate the correlation between education-num and income-class given different workclasses
pd.crosstab(df['education-num'], [df['workclass'], df['income-class']])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>workclass</th>
      <th colspan="2" halign="left">?</th>
      <th colspan="2" halign="left">Federal-gov</th>
      <th colspan="2" halign="left">Local-gov</th>
      <th>Never-worked</th>
      <th colspan="2" halign="left">Private</th>
      <th colspan="2" halign="left">Self-emp-inc</th>
      <th colspan="2" halign="left">Self-emp-not-inc</th>
      <th colspan="2" halign="left">State-gov</th>
      <th>Without-pay</th>
    </tr>
    <tr>
      <th>income-class</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>&lt;=50K</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>&lt;=50K</th>
    </tr>
    <tr>
      <th>education-num</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>131</td>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>259</td>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>15</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>27</td>
      <td>1</td>
      <td>1</td>
      <td>406</td>
      <td>18</td>
      <td>9</td>
      <td>5</td>
      <td>80</td>
      <td>14</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>50</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>0</td>
      <td>369</td>
      <td>18</td>
      <td>10</td>
      <td>0</td>
      <td>30</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>98</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>2</td>
      <td>648</td>
      <td>47</td>
      <td>16</td>
      <td>3</td>
      <td>60</td>
      <td>7</td>
      <td>11</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>118</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>34</td>
      <td>2</td>
      <td>1</td>
      <td>878</td>
      <td>45</td>
      <td>10</td>
      <td>4</td>
      <td>53</td>
      <td>7</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>38</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>17</td>
      <td>2</td>
      <td>0</td>
      <td>310</td>
      <td>23</td>
      <td>6</td>
      <td>1</td>
      <td>16</td>
      <td>3</td>
      <td>8</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>486</td>
      <td>46</td>
      <td>190</td>
      <td>73</td>
      <td>413</td>
      <td>90</td>
      <td>1</td>
      <td>6661</td>
      <td>1119</td>
      <td>160</td>
      <td>119</td>
      <td>687</td>
      <td>179</td>
      <td>219</td>
      <td>49</td>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>479</td>
      <td>35</td>
      <td>172</td>
      <td>82</td>
      <td>294</td>
      <td>93</td>
      <td>2</td>
      <td>4171</td>
      <td>923</td>
      <td>110</td>
      <td>116</td>
      <td>379</td>
      <td>107</td>
      <td>294</td>
      <td>31</td>
      <td>3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>48</td>
      <td>13</td>
      <td>23</td>
      <td>15</td>
      <td>61</td>
      <td>25</td>
      <td>0</td>
      <td>749</td>
      <td>256</td>
      <td>19</td>
      <td>19</td>
      <td>87</td>
      <td>21</td>
      <td>34</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>41</td>
      <td>6</td>
      <td>36</td>
      <td>19</td>
      <td>60</td>
      <td>28</td>
      <td>0</td>
      <td>559</td>
      <td>170</td>
      <td>17</td>
      <td>18</td>
      <td>53</td>
      <td>18</td>
      <td>35</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>128</td>
      <td>45</td>
      <td>117</td>
      <td>95</td>
      <td>315</td>
      <td>162</td>
      <td>0</td>
      <td>2056</td>
      <td>1495</td>
      <td>102</td>
      <td>171</td>
      <td>236</td>
      <td>163</td>
      <td>180</td>
      <td>90</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>30</td>
      <td>18</td>
      <td>20</td>
      <td>47</td>
      <td>169</td>
      <td>173</td>
      <td>0</td>
      <td>360</td>
      <td>534</td>
      <td>22</td>
      <td>57</td>
      <td>65</td>
      <td>59</td>
      <td>98</td>
      <td>71</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>10</td>
      <td>8</td>
      <td>6</td>
      <td>23</td>
      <td>10</td>
      <td>19</td>
      <td>0</td>
      <td>86</td>
      <td>171</td>
      <td>3</td>
      <td>78</td>
      <td>25</td>
      <td>106</td>
      <td>13</td>
      <td>18</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4</td>
      <td>11</td>
      <td>1</td>
      <td>15</td>
      <td>10</td>
      <td>17</td>
      <td>0</td>
      <td>49</td>
      <td>132</td>
      <td>6</td>
      <td>29</td>
      <td>19</td>
      <td>31</td>
      <td>18</td>
      <td>71</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



**Observation from above:**  
Education level seems to have influences on income. In some workclass, when education-num exceeds 13 (Bachelor), the number of people who earn >50K is more than that of people who earn <=50k.  
Thus, we consider creating a dummy variable to suggest whether a person's education level is above bachelor or not.


```python
# Investigate the correlation between occupation and income-class given different genders
table = pd.crosstab(df['occupation'], [df['sex'], df['income-class']])
# calculate percentages
table[('Female', '>50K percent')] = round(table[('Female', '>50K')] 
                                          / 
                                          (table[('Female', '<=50K')] + table[('Female', '>50K')]) * 100
                                          , 2)

table[('Male', '>50K percent')] = round(table[('Male', '>50K')] 
                                        / 
                                        (table[('Male', '<=50K')] + table[('Male', '>50K')]) * 100
                                        , 2)
table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>sex</th>
      <th colspan="2" halign="left">Female</th>
      <th colspan="2" halign="left">Male</th>
      <th>Female</th>
      <th>Male</th>
    </tr>
    <tr>
      <th>income-class</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>&lt;=50K</th>
      <th>&gt;50K</th>
      <th>&gt;50K percent</th>
      <th>&gt;50K percent</th>
    </tr>
    <tr>
      <th>occupation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>?</th>
      <td>789</td>
      <td>52</td>
      <td>863</td>
      <td>139</td>
      <td>6.18</td>
      <td>13.87</td>
    </tr>
    <tr>
      <th>Adm-clerical</th>
      <td>2325</td>
      <td>212</td>
      <td>938</td>
      <td>295</td>
      <td>8.36</td>
      <td>23.93</td>
    </tr>
    <tr>
      <th>Armed-Forces</th>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>NaN</td>
      <td>11.11</td>
    </tr>
    <tr>
      <th>Craft-repair</th>
      <td>202</td>
      <td>20</td>
      <td>2968</td>
      <td>909</td>
      <td>9.01</td>
      <td>23.45</td>
    </tr>
    <tr>
      <th>Exec-managerial</th>
      <td>879</td>
      <td>280</td>
      <td>1219</td>
      <td>1688</td>
      <td>24.16</td>
      <td>58.07</td>
    </tr>
    <tr>
      <th>Farming-fishing</th>
      <td>63</td>
      <td>2</td>
      <td>816</td>
      <td>113</td>
      <td>3.08</td>
      <td>12.16</td>
    </tr>
    <tr>
      <th>Handlers-cleaners</th>
      <td>160</td>
      <td>4</td>
      <td>1124</td>
      <td>82</td>
      <td>2.44</td>
      <td>6.80</td>
    </tr>
    <tr>
      <th>Machine-op-inspct</th>
      <td>530</td>
      <td>20</td>
      <td>1222</td>
      <td>230</td>
      <td>3.64</td>
      <td>15.84</td>
    </tr>
    <tr>
      <th>Other-service</th>
      <td>1749</td>
      <td>51</td>
      <td>1409</td>
      <td>86</td>
      <td>2.83</td>
      <td>5.75</td>
    </tr>
    <tr>
      <th>Priv-house-serv</th>
      <td>140</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>0.71</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Prof-specialty</th>
      <td>1130</td>
      <td>385</td>
      <td>1151</td>
      <td>1474</td>
      <td>25.41</td>
      <td>56.15</td>
    </tr>
    <tr>
      <th>Protective-serv</th>
      <td>66</td>
      <td>10</td>
      <td>372</td>
      <td>201</td>
      <td>13.16</td>
      <td>35.08</td>
    </tr>
    <tr>
      <th>Sales</th>
      <td>1175</td>
      <td>88</td>
      <td>1492</td>
      <td>895</td>
      <td>6.97</td>
      <td>37.49</td>
    </tr>
    <tr>
      <th>Tech-support</th>
      <td>303</td>
      <td>45</td>
      <td>342</td>
      <td>238</td>
      <td>12.93</td>
      <td>41.03</td>
    </tr>
    <tr>
      <th>Transport-moving</th>
      <td>81</td>
      <td>9</td>
      <td>1196</td>
      <td>311</td>
      <td>10.00</td>
      <td>20.64</td>
    </tr>
  </tbody>
</table>
</div>



**Observation from above:**  
Gender seems to have a big influence on income. On almost all occupations, male have higher income than female. For example, only 8.36% female make more than 50K as adm-clerical whereas 23.93% male make more than 50K in the same occupation.


```python
# Investigate the correlation between occupation and income-class given different races
table = pd.crosstab(df['occupation'], [df['race'], df['income-class']])
# calculate percentages
table[('White',  '>50K percent')] = round(table[('White', '>50K')]
                                          / (table[('White', '<=50K')] + table[('White', '>50K')]) * 100
                                          , 2)

table[('Other',  '>50K percent')] = round(table[('Other', '>50K')]
                                          / (table[('Other', '<=50K')] + table[('Other', '>50K')]) * 100
                                          , 2)

table[('Black',  '>50K percent')] = round(table[('Black', '>50K')]
                                          / (table[('Black', '<=50K')] + table[('Black', '>50K')]) * 100
                                          , 2)

table[('Asian-Pac-Islander',  '>50K percent')] = round(table[('Asian-Pac-Islander', '>50K')]
                                                       / 
                                                 (table[('Asian-Pac-Islander', '<=50K')]
                                                  + table[('Asian-Pac-Islander', '>50K')]) * 100
                                                       , 2)

table[('Amer-Indian-Eskimo',  '>50K percent')] = round(table[('Amer-Indian-Eskimo', '>50K')]
                                                       / 
                                                 (table[('Amer-Indian-Eskimo', '<=50K')]
                                                  + table[('Amer-Indian-Eskimo', '>50K')]) * 100
                                                       , 2)
```


```python
percent_cols = [col for col in table.columns if 'percent' in col[1]]
table[percent_cols]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th>race</th>
      <th>White</th>
      <th>Other</th>
      <th>Black</th>
      <th>Asian-Pac-Islander</th>
      <th>Amer-Indian-Eskimo</th>
    </tr>
    <tr>
      <th>income-class</th>
      <th>&gt;50K percent</th>
      <th>&gt;50K percent</th>
      <th>&gt;50K percent</th>
      <th>&gt;50K percent</th>
      <th>&gt;50K percent</th>
    </tr>
    <tr>
      <th>occupation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>?</th>
      <td>11.42</td>
      <td>8.70</td>
      <td>4.19</td>
      <td>7.69</td>
      <td>8.00</td>
    </tr>
    <tr>
      <th>Adm-clerical</th>
      <td>14.23</td>
      <td>3.85</td>
      <td>8.57</td>
      <td>15.83</td>
      <td>9.68</td>
    </tr>
    <tr>
      <th>Armed-Forces</th>
      <td>14.29</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Craft-repair</th>
      <td>22.85</td>
      <td>17.86</td>
      <td>20.08</td>
      <td>28.09</td>
      <td>13.64</td>
    </tr>
    <tr>
      <th>Exec-managerial</th>
      <td>49.86</td>
      <td>18.18</td>
      <td>34.43</td>
      <td>45.19</td>
      <td>10.00</td>
    </tr>
    <tr>
      <th>Farming-fishing</th>
      <td>12.35</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>12.50</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Handlers-cleaners</th>
      <td>6.53</td>
      <td>0.00</td>
      <td>6.15</td>
      <td>4.35</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Machine-op-inspct</th>
      <td>13.47</td>
      <td>2.56</td>
      <td>8.03</td>
      <td>16.95</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Other-service</th>
      <td>4.00</td>
      <td>0.00</td>
      <td>2.98</td>
      <td>13.28</td>
      <td>6.06</td>
    </tr>
    <tr>
      <th>Priv-house-serv</th>
      <td>0.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Prof-specialty</th>
      <td>46.07</td>
      <td>29.03</td>
      <td>27.62</td>
      <td>48.92</td>
      <td>33.33</td>
    </tr>
    <tr>
      <th>Protective-serv</th>
      <td>34.30</td>
      <td>20.00</td>
      <td>24.51</td>
      <td>33.33</td>
      <td>25.00</td>
    </tr>
    <tr>
      <th>Sales</th>
      <td>28.54</td>
      <td>12.00</td>
      <td>12.20</td>
      <td>19.44</td>
      <td>15.38</td>
    </tr>
    <tr>
      <th>Tech-support</th>
      <td>32.01</td>
      <td>0.00</td>
      <td>18.31</td>
      <td>27.27</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>Transport-moving</th>
      <td>21.62</td>
      <td>7.14</td>
      <td>10.59</td>
      <td>14.29</td>
      <td>12.00</td>
    </tr>
  </tbody>
</table>
</div>



**Observation from above:**  
Race seems to have an influence on income. White and Asian-Pac-Islanders have higher income than other races on almost all occupations.


```python
# Investigate the correlation between education-num and race
table = pd.crosstab(df['education-num'], df['race'])
table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>race</th>
      <th>Amer-Indian-Eskimo</th>
      <th>Asian-Pac-Islander</th>
      <th>Black</th>
      <th>Other</th>
      <th>White</th>
    </tr>
    <tr>
      <th>education-num</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>6</td>
      <td>5</td>
      <td>2</td>
      <td>38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>5</td>
      <td>16</td>
      <td>9</td>
      <td>134</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>18</td>
      <td>21</td>
      <td>13</td>
      <td>278</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>11</td>
      <td>56</td>
      <td>16</td>
      <td>551</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>9</td>
      <td>82</td>
      <td>7</td>
      <td>385</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15</td>
      <td>13</td>
      <td>124</td>
      <td>7</td>
      <td>634</td>
    </tr>
    <tr>
      <th>7</th>
      <td>12</td>
      <td>20</td>
      <td>138</td>
      <td>8</td>
      <td>817</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>9</td>
      <td>69</td>
      <td>13</td>
      <td>300</td>
    </tr>
    <tr>
      <th>9</th>
      <td>119</td>
      <td>225</td>
      <td>1171</td>
      <td>78</td>
      <td>8877</td>
    </tr>
    <tr>
      <th>10</th>
      <td>79</td>
      <td>206</td>
      <td>743</td>
      <td>51</td>
      <td>6195</td>
    </tr>
    <tr>
      <th>11</th>
      <td>19</td>
      <td>38</td>
      <td>112</td>
      <td>6</td>
      <td>1206</td>
    </tr>
    <tr>
      <th>12</th>
      <td>8</td>
      <td>29</td>
      <td>106</td>
      <td>8</td>
      <td>915</td>
    </tr>
    <tr>
      <th>13</th>
      <td>21</td>
      <td>286</td>
      <td>329</td>
      <td>33</td>
      <td>4645</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
      <td>88</td>
      <td>85</td>
      <td>7</td>
      <td>1520</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2</td>
      <td>36</td>
      <td>15</td>
      <td>2</td>
      <td>475</td>
    </tr>
    <tr>
      <th>16</th>
      <td>3</td>
      <td>28</td>
      <td>11</td>
      <td>2</td>
      <td>357</td>
    </tr>
  </tbody>
</table>
</div>



**Observation from above:**  
Asians have an unusually high number in ed level 13 (Bachelor).

### Betweeen Categorical Variables and Numeric Variables


```python
# Hours-per-week vs income-class

# seperate high income group and low income group
low_df = df.loc[df['income-class'] == '<=50K', 'hours-per-week']
high_df = df.loc[df['income-class'] == '>50K', 'hours-per-week']

sns.distplot(low_df, kde=False, color='y', label='<=50K')
sns.distplot(high_df, kde=False, color='k', label='>50K')
plt.legend()
```




    <matplotlib.legend.Legend at 0x12928fa20>




![png](output_44_1.png)


**Observation from above:**  
In high income group, the proportion of people who work more than 40 hours is higher than low income group.


```python
# Hours-per-week vs ge

# seperate high income group and low income group
low_df = df.loc[df['income-class'] == '<=50K', 'age']
high_df = df.loc[df['income-class'] == '>50K', 'age']

sns.distplot(low_df, kde=False, color='y', label='<=50K')
sns.distplot(high_df, kde=False, color='k', label='>50K')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1293727f0>




![png](output_46_1.png)



```python
low_income = df.loc[df['income-class'] == '<=50K', ['age', 'hours-per-week']]
high_income = df.loc[df['income-class'] == '>50K', ['age', 'hours-per-week']]
print('low income')
print(low_income.describe(), '\n')
print('high income')
print(high_income.describe())
```

    low income
                    age  hours-per-week
    count  24325.000000    24325.000000
    mean      37.104995       39.123947
    std       13.903084       12.147302
    min       18.000000        1.000000
    25%       26.000000       36.000000
    50%       34.000000       40.000000
    75%       46.000000       40.000000
    max       90.000000       99.000000 
    
    high income
                   age  hours-per-week
    count  7682.000000     7682.000000
    mean     44.206196       45.383494
    std      10.507283       10.964122
    min      19.000000        1.000000
    25%      36.000000       40.000000
    50%      43.000000       40.000000
    75%      51.000000       50.000000
    max      90.000000       99.000000


**Observation from above:**  
The median age and average age in high income group are both higher than low income group.

## Quantitative Analysis

### Between Categorical Variables


```python
# make a copy of the processed data
copy_df = df.copy()
```


```python
types = copy_df.dtypes
types
```




    age                   int64
    workclass            object
    fnlwgt                int64
    education-num         int64
    status               object
    occupation           object
    relationship         object
    race                 object
    sex                  object
    hours-per-week        int64
    income-class         object
    capital-gain-zero     int64
    capital-loss-zero     int64
    is_USA                int64
    no_pay_or_work        int64
    dtype: object




```python
# to do quantitative anlaysis, we need to transform non-numeric variables into numeric ones
oe = OrdinalEncoder()
obj_cols = types[types=='object'].index.tolist()
obj_cols.remove('income-class')
print(obj_cols)
copy_df[obj_cols] = oe.fit_transform(copy_df[obj_cols])

le = LabelEncoder()
copy_df['income-class'] = le.fit_transform(copy_df['income-class'])
```

    ['workclass', 'status', 'occupation', 'relationship', 'race', 'sex']



```python
# keep a record to map the original value to the encoded value
oe.categories_
```




    [array(['?', 'Federal-gov', 'Local-gov', 'Never-worked', 'Private',
            'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'],
           dtype=object),
     array(['Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
            'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'],
           dtype=object),
     array(['?', 'Adm-clerical', 'Armed-Forces', 'Craft-repair',
            'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners',
            'Machine-op-inspct', 'Other-service', 'Priv-house-serv',
            'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support',
            'Transport-moving'], dtype=object),
     array(['Husband', 'Not-in-family', 'Other-relative', 'Own-child',
            'Unmarried', 'Wife'], dtype=object),
     array(['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other',
            'White'], dtype=object),
     array(['Female', 'Male'], dtype=object)]




```python
copy_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>hours-per-week</th>
      <th>income-class</th>
      <th>capital-gain-zero</th>
      <th>capital-loss-zero</th>
      <th>is_USA</th>
      <th>no_pay_or_work</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>7.0</td>
      <td>77516</td>
      <td>13</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>6.0</td>
      <td>83311</td>
      <td>13</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>4.0</td>
      <td>215646</td>
      <td>9</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>4.0</td>
      <td>234721</td>
      <td>7</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>4.0</td>
      <td>338409</td>
      <td>13</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# calculate chi2 score between each categorical variable and income-class
cat_cols = obj_cols + ['education-num', 'capital-gain-zero', 'capital-loss-zero', 'is_USA']
print(cat_cols)
chi2, pval = chi2(copy_df[cat_cols], copy_df['income-class'])
```

    ['workclass', 'status', 'occupation', 'relationship', 'race', 'sex', 'education-num', 'capital-gain-zero', 'capital-loss-zero', 'is_USA']



```python
for i, chi2_score in enumerate(chi2):
    print('%s: %f' % (cat_cols[i], chi2_score))
```

    workclass: 47.508119
    status: 1123.469818
    occupation: 504.558854
    relationship: 3659.143125
    race: 33.031305
    sex: 502.439419
    education-num: 2401.421777
    capital-gain-zero: 192.123998
    capital-loss-zero: 29.218864
    is_USA: 4.029206



```python
for i, pv in enumerate(pval):
    print('%s: %f' % (cat_cols[i], pv))
```

    workclass: 0.000000
    status: 0.000000
    occupation: 0.000000
    relationship: 0.000000
    race: 0.000000
    sex: 0.000000
    education-num: 0.000000
    capital-gain-zero: 0.000000
    capital-loss-zero: 0.000000
    is_USA: 0.044719



```python
# calculate mutual information score between each categorical variable and income-class
mi = mutual_info_classif(copy_df[cat_cols], copy_df['income-class'])
```


```python
for i, v in enumerate(mi):
    print('%s: %f' % (cat_cols[i], v))
```

    workclass: 0.018099
    status: 0.111759
    occupation: 0.060578
    relationship: 0.116259
    race: 0.011263
    sex: 0.026260
    education-num: 0.067128
    capital-gain-zero: 0.034813
    capital-loss-zero: 0.011882
    is_USA: 0.007318


**Observation from above:**  
Chi2 score shows that all categorical variables are dependent with income-class with a significance level set at 0.05. Among them, is_USA is the least dependent one.  
Relationship, education-num and status are suggested to be the most relevant variables by both chi2 score and MI score. 

### Between Categorical Variables and Numeric Variables


```python
# do student T-test on numeric variables and income-class
num_cols = ['age', 'fnlwgt', 'hours-per-week']
high_income_df = copy_df.loc[copy_df['income-class']==1, num_cols]
low_income_df = copy_df.loc[copy_df['income-class']==0, num_cols]
```


```python
# check if the samples in high income group and low income group have equal mean, and equal (close) variance
mean_df = pd.concat([high_income_df.mean(), low_income_df.mean()], axis=1)\
            .rename(columns={0:'high_income', 1:'low_income'})
# N-1 to calculate variance by default
var_df = pd.concat([high_income_df.var(), low_income_df.var()], axis=1)\
            .rename(columns={0:'high_income', 1:'low_income'})
```


```python
mean_df 
# unequal mean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>high_income</th>
      <th>low_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>44.249841</td>
      <td>36.783738</td>
    </tr>
    <tr>
      <th>fnlwgt</th>
      <td>188005.000000</td>
      <td>190340.865170</td>
    </tr>
    <tr>
      <th>hours-per-week</th>
      <td>45.473026</td>
      <td>38.840210</td>
    </tr>
  </tbody>
</table>
</div>




```python
var_df 
# 1/2 < var1/var2 < 2, so very close variance
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>high_income</th>
      <th>low_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.106499e+02</td>
      <td>1.965629e+02</td>
    </tr>
    <tr>
      <th>fnlwgt</th>
      <td>1.051482e+10</td>
      <td>1.133847e+10</td>
    </tr>
    <tr>
      <th>hours-per-week</th>
      <td>1.212855e+02</td>
      <td>1.517576e+02</td>
    </tr>
  </tbody>
</table>
</div>




```python
# As we have unequal mean and almost equal variance in two samples for each numeric variables, 
# we can use the default settings in scipy ttest_ind()
for col in num_cols:
    print(col)
    t_stat, pval = ttest_ind(high_income_df[col], low_income_df[col])
    print('t statistic: %f' % t_stat)
    print('p value: %f' % pval, '\n')
```

    age
    t statistic: 43.436244
    p value: 0.000000 
    
    fnlwgt
    t statistic: -1.707511
    p value: 0.087737 
    
    hours-per-week
    t statistic: 42.583873
    p value: 0.000000 
    


**Observation from above:**   
Student T-test shows both age and hours-per-week are dependent with income-class with a significance level set at 0.05. This aligns with our findings in qualitative analysis.  
fnlwgt appears to be independent so we will remove it.

### Between Numeric Variables


```python
# correlation between numeric features
sns.pairplot(copy_df[num_cols])
```




    <seaborn.axisgrid.PairGrid at 0x1308d4860>




![png](output_70_1.png)



```python
copy_df[num_cols].corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>hours-per-week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.000000</td>
      <td>-0.079375</td>
      <td>0.039007</td>
    </tr>
    <tr>
      <th>fnlwgt</th>
      <td>-0.079375</td>
      <td>1.000000</td>
      <td>-0.021169</td>
    </tr>
    <tr>
      <th>hours-per-week</th>
      <td>0.039007</td>
      <td>-0.021169</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



**Observation from above:**  
No pair of numeric variables have strong correlations.

# Feature Engineering

Part 1: Feature selection
- based on the findings in correlation analysis  
- based on additional criteria

Part 2: Skewnewss

Part 3: Encoding

## Feature Selection


```python
# remove fnlwgt
copy_df.drop('fnlwgt', axis=1, inplace=True)
# create a dummy variable to indicate if a person's education level is above bachelor degree
copy_df['bachelor_above'] = 0
copy_df.loc[copy_df['education-num'] >= 13, 'bachelor_above'] = 1
copy_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education-num</th>
      <th>status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>hours-per-week</th>
      <th>income-class</th>
      <th>capital-gain-zero</th>
      <th>capital-loss-zero</th>
      <th>is_USA</th>
      <th>no_pay_or_work</th>
      <th>bachelor_above</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>7.0</td>
      <td>13</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>6.0</td>
      <td>13</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>4.0</td>
      <td>9</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>4.0</td>
      <td>7</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>4.0</td>
      <td>13</td>
      <td>2.0</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# remove variables where the variace is below a certain threshold
print('shape before: %s' % str(copy_df.shape))
selector = VarianceThreshold()
features = copy_df.drop('income-class', axis=1).columns
copy_df[features] = selector.fit_transform(copy_df[features])
print('shape after: %s' % str(copy_df.shape))
```

    shape before: (32561, 15)
    shape after: (32561, 15)


## Skewness


```python
# distribution of age is very skewed
sns.distplot(copy_df['age'], kde=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12a108c18>




![png](output_78_1.png)



```python
sns.distplot(np.log(copy_df['age']), kde=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12a28a9e8>




![png](output_79_1.png)



```python
# skewness is reduced by applying log transform even though it's still not a perfect bell shape
print(copy_df['age'].skew())
print(np.log(copy_df['age']).skew())
```

    0.5587433694130484
    -0.1317299194198282



```python
# apply log transform to age
copy_df['age'] = np.log(copy_df['age'])
```


```python
# distribution of hours-per-week is not as skewed
sns.distplot(copy_df['hours-per-week'], kde=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12a24ed30>




![png](output_82_1.png)



```python
copy_df['hours-per-week'].skew()
```




    0.22764253680450092




```python
# save a checkpoint
# ordinal encoded, imbalanced
#copy_df.to_csv('adult_oe_im.csv', index=False)
```

## Encoding


```python
# Apply one hot encoding for modeling purposes
ohe_cols = ['workclass', 'status', 'occupation', 'relationship', 'race', 'sex']
ohe_df = pd.get_dummies(data=copy_df, columns=ohe_cols)
ohe_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education-num</th>
      <th>hours-per-week</th>
      <th>income-class</th>
      <th>capital-gain-zero</th>
      <th>capital-loss-zero</th>
      <th>is_USA</th>
      <th>no_pay_or_work</th>
      <th>bachelor_above</th>
      <th>workclass_0.0</th>
      <th>...</th>
      <th>relationship_3.0</th>
      <th>relationship_4.0</th>
      <th>relationship_5.0</th>
      <th>race_0.0</th>
      <th>race_1.0</th>
      <th>race_2.0</th>
      <th>race_3.0</th>
      <th>race_4.0</th>
      <th>sex_0.0</th>
      <th>sex_1.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.663562</td>
      <td>13.0</td>
      <td>40.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.912023</td>
      <td>13.0</td>
      <td>13.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.637586</td>
      <td>9.0</td>
      <td>40.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.970292</td>
      <td>7.0</td>
      <td>40.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.332205</td>
      <td>13.0</td>
      <td>40.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>




```python
# save a checkpoint
# one hot encoded, imbalanced
#ohe_df.to_csv('adult_ohe_im.csv', index=False)
```
