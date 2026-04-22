import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_two_spirals(n_samples=1000, noise=0.2):
    n = n_samples // 2
    
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi
    
    r_a = 2 * theta
    x_a = r_a * np.cos(theta)
    y_a = r_a * np.sin(theta)
    
    r_b = -2 * theta
    x_b = r_b * np.cos(theta)
    y_b = r_b * np.sin(theta)
    
    X = np.vstack([
        np.column_stack([x_a, y_a]),
        np.column_stack([x_b, y_b])
    ])
    
    X += np.random.normal(0, noise, X.shape)
    
    y = np.hstack([np.zeros(n), np.ones(n)])
    
    df = pd.DataFrame(X, columns=['x', 'y'])
    df['target'] = y.astype(int)
    
    return df


# Dataset generieren
df = generate_two_spirals()

# CSV speichern
df.to_csv("two_spirals.csv", index=False)
print("Saved two_spirals.csv")

# PNG erzeugen
plt.figure(figsize=(8, 8))
plt.scatter(
    df['x'],
    df['y'],
    c=df['target'],
    cmap='Spectral',
    edgecolors='k',
    alpha=0.7
)

plt.title("Two Spirals Dataset")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig("two_spirals.png")
print("Saved two_spirals.png")