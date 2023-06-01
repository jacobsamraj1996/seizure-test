# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
warnings.filterwarnings('ignore')


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

ESR = pd.read_csv('data.csv')
ESR = ESR.drop(columns = ESR.columns[0])
ESR.head()
print(ESR.head())

cols = ESR.columns
tgt = ESR.y
tgt[tgt > 1] = 0
ax = sn.countplot(tgt,label="Count")
non_seizure, seizure = tgt.value_counts()
print('The number of trials for the non-seizure class is:', non_seizure)
print('The number of trials for the seizure class is:', seizure)
ESR.isnull().sum().sum()
ESR.describe()
print(ESR.describe())
Y = ESR.iloc[:,178].values
Y.shape
X = ESR.iloc[:,1:178].values
X.shape
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train) * 100, 2)
print("Accuracy is:",(str(acc_svc)+'%'))
new_input1 = [ESR.iloc[6, :177]]
new_input1
print(new_input1)
new_output = clf.predict(new_input1)
new_output
new_output
if new_output==[1]:
    print('"yes" you might get seizure be conscious about it')
else:
    print('You are safe no worries :)')