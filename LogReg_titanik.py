import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



#reading data / creating of Data Frame
data = pd.read_csv('titanic_data.csv')

print(data.to_string())
print()
print(data.info())

#removing duplicates
data.drop_duplicates(inplace = True)

#creating sub-dataset
data_1 = data.drop(['PassengerId','Name','Ticket','Fare','Cabin'], axis = 1)

#dropping empty cells
data_1.dropna(inplace = True)

#cleaned data
print()
print(data_1.info())
print()
print(data_1.to_string())

#simple statistics
print()
print(data_1.describe())

#correlation
print()
print(data_1.corr())


#visualization

sns.countplot(x = "Survived", data = data_1)
plt.show()

sns.countplot(x = "Survived", hue = "Sex", data = data_1)
plt.show()

sns.countplot(x = "Survived", hue = "Pclass", data = data_1)
plt.show()

data_1["Age"].plot.hist()
plt.show()

sns.boxplot(x = "Pclass", y = "Age", data = data_1)
plt.show()



#002. LOGISTIC REGRESSION

#dummy variables
data_1 = pd.get_dummies(data = data_1, drop_first = True)

#Dependent and independent variables
y = data_1["Survived"]
X = data_1.drop(["Survived"], axis = 1)

#Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Training
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Prediction
y_pred = classifier.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print()
print(df.to_string())

#Evaluation
print()
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print()
print(metrics.classification_report(y_test, y_pred))
print()
print(metrics.confusion_matrix(y_test, y_pred))

