import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# Pfad bestimmen
base_path = os.path.dirname(__file__)

# 1. Trainingsdaten laden
train_path = os.path.join(base_path, "crossing_spirals_train.csv")
df_train = pd.read_csv(train_path)

# 2. Testdaten laden
test_path = os.path.join(base_path, "crossing_spirals_test.csv")
df_test = pd.read_csv(test_path)

# 3. Feature Engineering (sin/cos hinzufügen)
def add_features(df):
    df['x_sin'] = np.sin(df['x'])
    df['y_sin'] = np.sin(df['y'])
    df['x_cos'] = np.cos(df['x'])
    df['y_cos'] = np.cos(df['y'])
    return df

df_train = add_features(df_train)
df_test = add_features(df_test)

# 4. Features & Labels trennen (6 Inputs!)
features = ['x', 'y', 'x_sin', 'y_sin', 'x_cos', 'y_cos']

X_train = df_train[features]
y_train = df_train['label']

X_test = df_test[features]
y_test = df_test['label']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Modell erstellen (6-32-32-1)
model = MLPClassifier(
    hidden_layer_sizes=(32, 32),
    activation='relu',        # AF ausprobieren: 'tanh', 'logistic', 'identity', etc.
    max_iter=500,
    learning_rate_init=0.001, #Lernrate
    random_state=42
)

# 6. Trainieren
model.fit(X_train, y_train)

# 7. Vorhersage
y_pred = model.predict(X_test)

# 8. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)


# =========================
# CONFUSION MATRIX
# =========================
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Red Spiral (0)", "Blue Spiral (1)"],
    include_values=True
)

plt.title("Confusion Matrix - Crossing Spirals")
plt.show()


# =========================
# PLOT: TEST DATA
# =========================
plt.figure(figsize=(6,6))

plt.scatter(
    df_test['x'],
    df_test['y'],
    c=y_test,
    cmap='coolwarm',
    edgecolors='k',
    alpha=0.7
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Crossing Spirals - True Labels")
plt.grid(True)
plt.show()


# =========================
# PLOT: PREDICTIONS
# =========================
plt.figure(figsize=(6,6))

plt.scatter(
    df_test['x'],
    df_test['y'],
    c=y_pred,
    cmap='coolwarm',
    edgecolors='k',
    alpha=0.7
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Crossing Spirals - Predictions")
plt.grid(True)
plt.show()


# =========================
# DECISION BOUNDARY (optional, aber sehr hilfreich!)
# =========================
xx, yy = np.meshgrid(
    np.linspace(df_test['x'].min(), df_test['x'].max(), 300),
    np.linspace(df_test['y'].min(), df_test['y'].max(), 300)
)

grid = pd.DataFrame({
    'x': xx.ravel(),
    'y': yy.ravel()
})

grid = add_features(grid)

Z = model.predict(grid[features])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

plt.scatter(
    df_test['x'],
    df_test['y'],
    c=y_test,
    cmap='coolwarm',
    edgecolors='k'
)

plt.title("Decision Boundary")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# =========================
# WEIGHTS VISUALIZATION
# =========================
plt.figure(figsize=(8,6))
plt.imshow(model.coefs_[0], interpolation='none')
plt.colorbar(label="Weight Magnitude")


plt.yticks(range(6), features)

plt.title("Weights Input for Hidden Layer")
plt.show()