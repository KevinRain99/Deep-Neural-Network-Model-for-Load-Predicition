import pandas as pd
import quandl,math
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression


xl = pd.ExcelFile("./data/data.xlsx")
df=xl.parse("Sheet1")
#df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df=df.reindex(columns=['AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ', 'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', 'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'BS', 'BT', 'BU', 'BV','Total','temp high','temp avg','temp low'])
print (df.columns)
print(df.head())

forecast_col = 'Total'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df))) # 1% forecast
print (forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out) 
df.dropna(inplace=True)

X=np.array(df.drop(['label'],1)) #features..except label(feature to forecast)
y=np.array(df['label']) #to forecast
print (X)
print (y)
X=preprocessing.scale(X)
y=np.array(df['label'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
X_lately = X[-forecast_out:]
clf = svm.SVR()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test) #ACCURACY
forecast_set = clf.predict(X_lately)
print (forecast_set)
print(confidence)
print ("SVM")
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)
print ("\n\nLinear Regression")
clf = LinearRegression()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)
    

