import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics



#reading data / creating of Data Frame
data = pd.read_csv('melbourne_house_data.csv')

#print(data.to_string())
#print()
print(data.info())


#removing duplicates
data.drop_duplicates(inplace = True)

#creating sub-dataset
data_1 = data.drop(['id','Address','Method','SellerG','Date','Suburb','Postcode','CouncilArea','Lattitude','Longtitude','Regionname'], axis = 1)

#dropping empty cells
data_1.dropna(inplace = True)

#cleaned data
print()
print(data_1.info())
print()
#print(data_1.to_string())

#simple statistics
print()
print(data_1.describe())

#correlation
print()
print(data_1.corr())

#visualization

data_1.plot(kind = "scatter", x = "Rooms", y = "Price", color = 'black')
plt.show()

data_1.plot(kind = "scatter", x = "YearBuilt", y = "Price", color = 'black')
plt.show()

data_1["Price"].plot(kind = "hist", color = 'black')
plt.show()


#linear regression with stats: uni-variable only

slope, intercept, r, p, std_err = stats.linregress(data_1["Rooms"], data_1["Price"])
x = data_1["Rooms"]
y = data_1["Price"]
plt.scatter(x,y, color = 'black')
plt.plot(x, slope*x+intercept, color = "blue")
plt.show()

slope, intercept, r, p, std_err = stats.linregress(data_1["Landsize"], data_1["Price"])
x = data_1["Landsize"]
y = data_1["Price"]
plt.scatter(x,y, color = 'black')
plt.plot(x, slope*x+intercept, color = "blue")
plt.show()


#001. LINEAR REGRESSION with sklearn: multi-variable

#dummy variables
data_1 = pd.get_dummies(data = data_1, drop_first = True)

#Dependent and independent variables
y = data_1["Price"]
X = data_1.drop(["Price"], axis = 1)

#Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Training
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Coefficients of predicted linear function
coeff_df = pd.DataFrame(regressor.coef_, X.columns)
print()
print(coeff_df.to_string)

#Prediction
y_pred = regressor.predict(X_test)
y_pred = y_pred.astype('int32')

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.to_string())

#Evaluation
print()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
