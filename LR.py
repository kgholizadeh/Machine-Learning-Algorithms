import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

shuttle = pd.read_csv('d:/shuttle/shuttle-trn.csv',
                      names=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'Class'])

model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(x, y)

# Step 4: Evaluate the model
p_pred = model.predict_proba(x)
y_pred = model.predict(x)
score_ = model.score(x, y)
conf_m = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)
f = open("d:/shuttle/shuttle-trn.txt")
f.readline() # skip the header
data = np.loadtxt(f)
x = data[:, 0:]
y = data[:, 1]


# f = open("D:\Shuttle\shuttle-trn.txt")
# g = open("D:\Shuttle\shuttle-tst.txt")
# f.readline() # skip the header
# g.readline() # skip the header
# dataf = np.loadtxt(f)
# datag = np.loadtxt(g)
# xf = dataf[:, 0:]
# yf = dataf[:, 1]
# xg = datag[:, 0:]
# yg = datag[:, 1]
#
# logmodel = LogisticRegression()
# logmodel.fit(xf, yf)
#
# predictions = logmodel.predict(xg)
# print(classification_report(yg, predictions))
# print(confusion_matrix(yg, predictions))
# print(accuracy_score(yg, predictions))

# from sklearn.model_selection import train_test_split
#
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.20)
# from sklearn.svm import SVC
# svclassifier = SVC(kernel='linear')
# svclassifier.fit(x_train,y_train)
#
# y_pred = svclassifier.predict(x_test)
#
# from sklearn.metrics import  classification_report, confusion_matrix
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))


