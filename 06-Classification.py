import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix

df=pd.read_csv('output/feature_full/total_feature.csv',sep=',')
df=df.drop('Unnamed: 0',axis=1)

#lọc dữ liệu Nan, inf,-inf
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
df=clean_dataset(df)

df_0 = df[df['mood']==0]
df_5 = df[df['mood']==5]
df_10 = df[df['mood']==10]
a=765000
luongtrain=76500
df_0_train = df_0[:luongtrain]
df_5_train = df_5[:luongtrain]
df_10_train = df_10[:luongtrain]
df_0_test = df_0[900000:]
df_5_test = df_5[900000:]
df_10_test = df_10[900000:]
df_train = pd.concat([df_0_train,df_5_train,df_10_train],ignore_index=True)
df_test = pd.concat([df_0_test,df_5_test,df_10_test],ignore_index=True)

X_train = df_train.drop(['mood','EAR','MAR','PUC','MOE'],axis=1)
y_train = df_train['mood']
X_test = df_test.drop(['mood','EAR','MAR','PUC','MOE'],axis=1)
y_test=df_test['mood']


'''
from sklearn.linear_model import LogisticRegression

print('start Logistic Regression')
clf = LogisticRegression(max_iter=500).fit(X_train, y_train)
y_pred_1 = clf.predict(X_test)
acc1 = accuracy_score(y_test, y_pred_1)
print('hiệu quả trên tập train:',clf.score(X_train,y_train))
print('hiệu quả trên tập test:',acc1)
print('ma trận nhầm lẫn:\n',confusion_matrix(y_test,y_pred_1))
'''


from sklearn.svm import SVC
print('Start SVM')
clf = SVC().fit(X_train,y_train)
y_predict_SVM = clf.predict(X_test)
acc1 = clf.score(X_train,y_train)
acc2 = accuracy_score(y_test, y_predict_SVM)
confusion_matrix_result = confusion_matrix(y_test,y_predict_SVM)
print('hiệu quả trên tập train:',acc1)
print('hiệu quả trên tập test:',acc2)
print('ma trận nhầm lẫn:\n',confusion_matrix_result)


with open('output/SVC.txt','a',encoding="utf-8") as SVC_report:
    SVC_report.write('result of SVC\n')
    SVC_report.write(f'số lượng train: {luongtrain*3}')
    SVC_report.write(f'hiệu quả trên tập train: {acc1}\n')
    SVC_report.write(f'hiệu quả trên tập test: {acc2}\n')
    SVC_report.write(f'ma trận nhầm lẫn: {confusion_matrix_result}\n')
    SVC_report.write('-' * 20)
pass
