import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import os
import numpy as np
import matplotlib.pyplot as plt

# Pfad bestimmen
base_path = os.path.dirname(__file__)

# 1. Trainingsdaten laden
train_path = os.path.join(base_path, "iris_train.csv")
df_train = pd.read_csv(train_path)

# 2. Testdaten laden
test_path = os.path.join(base_path, "iris_test.csv")
df_test = pd.read_csv(test_path)

# 3. Features & Labels trennen
X_train = df_train[[
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]]
y_train = df_train['label']

X_test = df_test[[
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]]
y_test = df_test['label']

# 4. Modell erstellen (4-8-3)
model = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation='relu',     # ausprobieren: 'tanh', 'logistic', etc.
    max_iter=150,
    learning_rate_init=0.05,
    random_state=42
)

# 5. Trainieren
model.fit(X_train, y_train)

# 6. Vorhersage
y_pred = model.predict(X_test)

# 7. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)


# Confusion Matrix
# Namen der Iris-Klassen für die Beschriftung
target_names = ['Setosa', 'Versicolor', 'Virginica']

# Erstellung der Matrix mit Labels
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_pred, 
    display_labels=target_names,
    include_values=False 
)

# Colorbar anpassen
cbar = disp.im_.colorbar
cbar.set_ticks([0, disp.confusion_matrix.max()])
cbar.set_ticklabels(['Low', 'High'])

# Zusätzliche Beschriftungen
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

plt.figure(figsize=(8,6))

plt.scatter(
    X_test['petal length (cm)'],
    X_test['petal width (cm)'],
    c=y_test,
    cmap='viridis',
    edgecolors='k'
)

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Iris Data (Test Set)")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))

plt.scatter(
    X_test['sepal length (cm)'],
    X_test['sepal width (cm)'],
    c=y_test,
    cmap='viridis',
    edgecolors='k'
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Data (Sepal Features)")
plt.grid(True)
plt.show()


plt.figure(figsize=(6, 4))
plt.imshow(model.coefs_[0], interpolation='none', cmap='viridis')
plt.colorbar(label="Weight Magnitude")

plt.xticks(range(8), [f'N{i}' for i in range(8)])
plt.yticks(range(4), [
    'Sepal L',
    'Sepal W',
    'Petal L',
    'Petal W'
])

plt.title("Weights of Hidden Layer")
plt.show()