![{842A45D2-0DAF-4AEB-8CB2-F4C8608573EF}](https://github.com/user-attachments/assets/987549c7-33f8-44e4-8306-c74287d36b77)# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![{FA80482A-9E5D-4EAF-A240-959A9FA26F9F}](https://github.com/user-attachments/assets/945839a3-214d-41d3-9e85-3e8114eb06fd)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![{7D91CF8D-7BB1-4698-AA7E-C1F2D4B05AB3}](https://github.com/user-attachments/assets/1d4c1fce-76c5-4ac8-a004-1b12fd23769e)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![{323CE491-A7EA-4D5B-A190-1965A62AFB17}](https://github.com/user-attachments/assets/19edb126-0732-4a79-ac6f-e27b943d652c)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![{9C161AB1-31C5-426C-9237-5B05698B17F2}](https://github.com/user-attachments/assets/00dfa798-cfde-4264-9921-f1dbadb5eeb7)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![{AB2B484C-E8EF-43B3-957F-20BE02DD440A}](https://github.com/user-attachments/assets/a973f707-caf0-4000-870a-2655c4c32610)
```
from sklearn.preprocessing import MaxAbsScaler
sc=MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![{ED8EE2AA-AD8A-42D7-A711-9DF27344BB1E}](https://github.com/user-attachments/assets/acc4365f-ea85-4f81-86c8-bf5381804da5)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head()
```
![{3F182863-26DB-4BF3-8275-FB780DD94668}](https://github.com/user-attachments/assets/f6d6f285-6e04-4ac9-af6c-254f34d88538)
```
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![{23153665-A0A5-4472-A8EA-020C0F442CA6}](https://github.com/user-attachments/assets/e17dde68-496f-427a-ab52-727db40bf962)
```
data.isnull().sum()
```
![{0B57A892-3414-4C7A-BDF9-2C0FD2176B65}](https://github.com/user-attachments/assets/6856f336-7b0a-4f2f-bd88-0a5e1a737793)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![{E52A360C-BE3E-4542-AC0A-F3F70D8FB46D}](https://github.com/user-attachments/assets/ecb8821e-2a60-4e6f-9748-72b59459955e)
```
data2=data.dropna(axis=0)
data2
```
![{D3965B07-FD31-4A45-9569-AA061D443688}](https://github.com/user-attachments/assets/0e09f1f5-f887-403e-8cf8-cb01f2e2cbdf)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![{4B0CB39E-A8BF-4C5B-839C-0A9E7FC2669F}](https://github.com/user-attachments/assets/7d4b9104-9cde-41a9-b0c3-a23ad15651f3)
```
data2
```
![{1B7231CD-5292-418E-AB2F-D4A761A6FD5B}](https://github.com/user-attachments/assets/9f5c7b03-aa9e-4536-914d-ab08512dcdf3)
```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![{D431BB22-820E-4E71-A74B-3DB4DBD9F092}](https://github.com/user-attachments/assets/cfa3b365-dfe3-4319-9885-dfd38517032e)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![{5CA86C59-BD8A-4F0A-B8DB-1ED5312973FB}](https://github.com/user-attachments/assets/be1de8c8-fe16-4e15-bd5d-465476f5ca0d)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![{14D959D1-E1C9-4981-9033-9C1B8DA97298}](https://github.com/user-attachments/assets/b3be4e67-8719-4661-910f-ce707f11a34a)
```
y=new_data['SalStat'].values
print(y)
x=new_data[features].values
print(x)
```
![{842A45D2-0DAF-4AEB-8CB2-F4C8608573EF}](https://github.com/user-attachments/assets/c7e36fc4-8b13-41f5-8ff2-30dc447bb5ea)
```
train_x, test_x, train_y, test_y=train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
```
![{76F2F6EF-E0FA-4CA6-9430-C5ADF6401FD3}](https://github.com/user-attachments/assets/e23ade43-8702-4c25-9426-2d2403ee07bb)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![{E088A815-47BA-4C8A-B76F-FAD30005D625}](https://github.com/user-attachments/assets/c666b78f-e320-4380-8968-39614d2f6614)
```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```
![{A82117F6-3CB3-458C-B9EF-F35854FB336E}](https://github.com/user-attachments/assets/fb249d12-c2b7-483f-8368-9971936cc129)
```
print('Misclassified samples: %d' % (test_y !=prediction).sum())
```
![{B13317C4-1878-45DC-8DCC-7AACB72CC4BA}](https://github.com/user-attachments/assets/3752ed01-229a-472f-bff9-f5231623c120)
```
data.shape
```
![{221322CB-55C8-4D6B-B6CD-E5ED83F0BEEF}](https://github.com/user-attachments/assets/c9ceb1c6-4e19-4853-884d-ce16cbafa74b)
```
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![{5C1A9762-5C4A-4A21-A1CD-F0C93631607A}](https://github.com/user-attachments/assets/56ab4f0d-8c5b-463f-9c5c-06fd89230c60)
```
contigency_table=pd.crosstab(tips['sex'],tips['time'])
print(contigency_table)
```
![{2313111E-250F-45C2-966E-FAA12E33DED8}](https://github.com/user-attachments/assets/a54dfe04-804f-4e77-96e1-bfc98b495fd4)
```
chi2, p, _, _ =chi2_contingency(contigency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![{5940B988-4249-4751-896B-90723C5822E3}](https://github.com/user-attachments/assets/523147b2-ef79-46d1-a1ca-48ab5de0f1ae)
```
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': ['A', 'B', 'C', 'A', 'B'],
    'Feature3': [0, 1, 1, 0, 1],
    'Target': [0, 1, 1, 0, 1]
}
df=pd.DataFrame(data)
X=df[['Feature1', 'Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new=selector.fit_transform(X,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print('Selected Features:')
print(selected_features)
```
![{EE94828E-C0F5-4215-A855-3F0E794860A2}](https://github.com/user-attachments/assets/3b8e48fa-0821-4057-b414-7a4e7f17395b)
# RESULT:
    we have performed Feature Scaling and Feature Selection process and save the data to a file.

