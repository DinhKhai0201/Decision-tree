import pandas as pd # thư viện lấy dữ liệu train và test
from sklearn.tree import DecisionTreeClassifier # hàm decision tree
from sklearn.model_selection import train_test_split # chia dữ liệu thành tập train và test
from sklearn import metrics # cung cấp thư viện để so sánh số liệu giữa y_pred và y_test

pima=pd.read_csv("diabetes.csv") # đọc dữ liệu tù csv
X=pima.iloc[:,:8] # X lấy tất cả các hàng, cột lấy từ cột đầu tiên cho dến cột 7
y=pima.iloc[:,8] # lấy tất cả các hàng, cột lấy cột 8
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

clf=DecisionTreeClassifier()
clf=clf.fit(X_train,y_train) # train dữ liệu
y_pred=clf.predict(X_test) # tiên đoán
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # so sánh giống bao nhiêu %