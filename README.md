# EXNO:4-DS
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
```
```
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![{D5C0709E-E892-4794-98A9-2599EF784D1F}](https://github.com/user-attachments/assets/2fc6df00-3ef7-4c06-8c91-d062299e5392)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![{A09BF65E-288A-485D-9006-F1F883611F9A}](https://github.com/user-attachments/assets/fb9dc29e-98b4-4a56-9581-bc559f24c53f)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals

```
![{93A2F1F3-6A54-4AFC-B320-8212A8D18391}](https://github.com/user-attachments/assets/ca51ac83-c399-46cd-83b0-952f89912242)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![{B8A7767B-A311-4D65-BC61-F4ACD5B23F83}](https://github.com/user-attachments/assets/177ff9fb-5371-4e76-8086-408c9fcc779d)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![{BD89CA17-6CC3-4775-9A6B-BB48180DEAFB}](https://github.com/user-attachments/assets/ebb4ec4d-1cc7-4a95-b6d4-d7e820953bce)
```
from sklearn.preprocessing import MaxAbsScaler
sc=MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![{90018A8F-E916-402E-9B8A-BEE0831D8424}](https://github.com/user-attachments/assets/5a12ff4a-5868-490e-a8a2-0748012b18df)




# RESULT:
       # INCLUDE YOUR RESULT HERE
