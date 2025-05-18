from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import metrics
import matplotlib.pypot as plt
nnist = fetch_openml('mnist_784', version=1)
X = nnist ['data'] / 255.0
y = nnist['target'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=100000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = metircs.accuracy_score(y_test,y pred)
print(f "Test accuracy: {accuracy}")
for 1 in range(5):
    plt.inshow(X_test .iloc[i].values.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}")
    plt.show()