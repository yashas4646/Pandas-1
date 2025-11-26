import pandas as pd
from sklearn import linear_model
df=pd.read_csv(r"C:\Users\Dixit\Documents\cie.csv")
x=df[["CIE1","CIE2","CIE3","CIE"]]
y=df[["SEE"]]
regr=linear_model.LinearRegression()
regr.fit(x,y)
predicted1=regr.predict([[30,27,26]])
predicted2=regr.predict([[27,24,26]])
predicted3=regr.predict([[25,23,26]])
predicted4=regr.predict([[26,24,28]])
predicted5=regr.predict([[25,24,23]])
print(predicted1)
print(predicted2)
print(predicted3)
print(predicted4)
print(predicted5)# Pandas-1
