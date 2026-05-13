import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os
import numpy as np
import matplotlib.pyplot as plt

# Pfad bestimmen
base_path = os.path.dirname(__file__)

# 1. Trainingsdaten laden
train_path = os.path.join(base_path, "concentric_circles_train.csv")
df_train = pd.read_csv(train_path)

# 2. Testdaten laden
test_path = os.path.join(base_path, "concentric_circles_test.csv")
df_test = pd.read_csv(test_path)

# 3. Features & Labels trennen
X_train = df_train[['x', 'y']]
y_train = df_train['label']

X_test = df_test[['x', 'y']]
y_test = df_test['label']

# 4. Modell erstellen (Baseline mit AF)
model = MLPClassifier(
    hidden_layer_sizes=(8,),
    activation='relu',   # hier AF ändern
    max_iter=100,
    learning_rate_init=0.005, #Lernrate
    random_state=42
)

# 5. Trainieren
model.fit(X_train, y_train)

# 6. Vorhersage auf TEST-DATEN
y_pred = model.predict(X_test)

# 7. Accuracy berechnen
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)


# Visualisierung

# 1. Meshgrid erstellen (Raum aufspannen)
x_min, x_max = X_train['x'].min() - 0.5, X_train['x'].max() + 0.5
y_min, y_max = X_train['y'].min() - 0.5, X_train['y'].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# 2. Modell auf jedem Punkt anwenden
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid)
Z = Z.reshape(xx.shape)

# 3. Plot
plt.figure(figsize=(8, 8))

# Entscheidungsfläche
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Originalpunkte (Testdaten)
plt.scatter(
    X_test['x'], X_test['y'],
    c=y_test,
    cmap='coolwarm',
    edgecolors='k'
)

plt.title("Decision Boundary")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()


# Gewichtung

plt.figure(figsize=(6, 4))
plt.imshow(model.coefs_[0], interpolation='none', cmap='viridis')
plt.colorbar(label="Weight Magnitude")
plt.xticks(range(8), [f'N{i}' for i in range(8)])
plt.yticks([0, 1], ['Input X', 'Input Y'])
plt.title("Weights of the hidden layer neurons")
plt.show()