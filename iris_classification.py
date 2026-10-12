CodeAlpha Task 1: Iris Flower Classification

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

Load the Iris dataset

iris = load_iris()
X = iris.data
y = iris.target

Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)

Train the Machine Learning model

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

Test the model

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Iris Flower Classification")
print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

Predict a flower species

sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("Predicted Flower Species:", iris.target_names[prediction[0]])
