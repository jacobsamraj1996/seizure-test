import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

ESR = pd.read_csv('data.csv')
ESR.head()

print(ESR.head())

cols = ESR.columns
tgt = ESR.y
tgt[tgt > 1] = 0
ax = sn.countplot(tgt, label="Count")
non_seizure, seizure = tgt.value_counts()
print('The number of trials for the non-seizure class is:', non_seizure)
print('The number of trials for the seizure class is:', seizure)

ESR.isnull().sum()
print(ESR.isnull().sum())
ESR.info()
ESR.describe()
print(ESR.describe())

X = ESR.iloc[:, 1:179].values
X.shape
print(X.shape)

plt.figure(figsize=(10, 10))
plt.subplot(511)
plt.plot(X[1, :])
plt.title('Class 1')
plt.ylabel('uV')
plt.subplot(512)
plt.plot(X[7, :])
plt.title('Class 7')
plt.subplot(513)
plt.plot(X[12, :])
plt.title('Class 12')
plt.subplot(514)
plt.plot(X[0, :])
plt.title('Class 0')
plt.subplot(515)
plt.plot(X[2, :])
plt.title('Class 2')
plt.xlabel('Samples')

plt.tight_layout()
plt.show()

y = ESR.iloc[:, 179].values
y
print(y)
y[y > 1] = 0
y

from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_log_reg) + ' %')

# Support Vector Machine (SVM)
from sklearn.svm import SVC
clf = SVC()
clf.fit(X_train, y_train)
y_pred_svc = clf.predict(X_test)
acc_svc = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_svc) + '%')

# Linear SVM
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
acc_linear_svc = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_linear_svc) + '%',)

# K-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_knn) + '%')

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred_gnb = clf.predict(X_test)
acc_gnb = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_gnb) + '%')

# Artificial Neural Network
classifier = Sequential()
classifier.add(Dense(units=80, kernel_initializer='uniform', activation='relu', input_dim=178))
classifier.add(Dense(units=80, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
acc_ANN = round(clf.score(X_train, y_train) * 100, 2)
print(str(acc_ANN) + '%')

from sklearn.decomposition import PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
acc_PCA = round(pca.score(X_train, y_train))
print(str(acc_PCA) + '%')

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'ANN', 'KNN', 'Naive Bayes', 'Principal Component Analysis'],
    'Score': [acc_log_reg, acc_svc, acc_knn, acc_gnb, acc_ANN, acc_PCA]
})

sorted_models = models.sort_values(by='Score', ascending=False)
print(sorted_models)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_log_reg)
print('Confusion Matrix:')
print(cm)
