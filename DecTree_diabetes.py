import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn import metrics



#reading data / creating of Data Frame
data = pd.read_csv('diabetes_data.csv')

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


#003. DECISION TREE

#Dependent and independent variables
y = data["Outcome"]
X = data.drop(["Outcome"], axis = 1)

#Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#Training
#classifier = DecisionTreeClassifier(max_depth=4)
classifier = DecisionTreeClassifier(criterion = "entropy", max_depth=4)
#regressor = DecisionTreeRegressor()
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


#Visualizing Decision Tree

#text_representation = tree.export_text(classifier)
#print(text_representation)

fig = plt.figure(figsize=(15,8))
_ = tree.plot_tree(classifier,
                   feature_names=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"],
                   class_names=["1","0"],
                   filled=True)
plt.show()