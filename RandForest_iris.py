import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn import metrics



#reading data / creating of Data Frame
data = pd.read_csv('iris_data.csv')

print(data.to_string())
print()
print(data.info())

#removing duplicates
data.drop_duplicates(inplace = True)

#dropping empty cells
data.dropna(inplace = True)

#cleaned data
print()
print(data.info())
print()
print(data.to_string())

#simple statistics
print()
print(data.describe())

#correlation
print()
print(data.corr())


#004. RANDOM FOREST

#Dependent and independent variables
y = data["variety"]
X = data.drop(["variety", "sepal.length", "sepal.width"], axis = 1)
#X = data.drop(["variety"], axis = 1)

#Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Training
classifier = RandomForestClassifier()
#regressor = RandomForestRegressor()
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

"""
#Finding important features
feature_imp = pd.Series(classifier.feature_importances_,index=['sepal.length','sepal.width','petal.length','petal.width']).sort_values(ascending=False)
print()
print(feature_imp)
"""

#Visualizing a Decision Tree of Random Forest

#fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (1,1), dpi=800)
fig = plt.figure(figsize=(15,8))
_  = tree.plot_tree(classifier.estimators_[0],
               feature_names = ['petal.length', 'petal.width'],
               class_names= ["Setosa", "Versicolor", "Virginica"],
               filled = True)
plt.show()
