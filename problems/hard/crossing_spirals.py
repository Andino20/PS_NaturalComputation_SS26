import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_crossing_spirals(n_samples=1200, noise=0.05):
    n_per_spiral = n_samples // 2
    
    theta = np.linspace(0, 4 * 2 * np.pi, n_per_spiral)
    
    # Spirale A
    r = theta / (4 * 2 * np.pi)
    x_a = r * np.cos(theta)
    y_a = r * np.sin(theta)
    
    # Spirale B (kreuzt sich)
    x_b = r * np.cos(theta + np.pi) * (1 + 0.3 * np.sin(theta * 2))
    y_b = r * np.sin(theta + np.pi) * (1 + 0.3 * np.cos(theta * 3))
    
    X = np.vstack([
        np.column_stack([x_a, y_a]),
        np.column_stack([x_b, y_b])
    ])
    
    # Noise
    noise_a = np.random.normal(0, noise, (n_per_spiral, 2))
    noise_b = np.random.normal(0, noise * 1.2, (n_per_spiral, 2))
    
    X[:n_per_spiral] += noise_a
    X[n_per_spiral:] += noise_b
    
    y = np.hstack([np.zeros(n_per_spiral), np.ones(n_per_spiral)])
    
    df = pd.DataFrame(X, columns=['x', 'y'])
    df['target'] = y.astype(int)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    df = generate_crossing_spirals(n_samples=1200, noise=0.05)
    
    filename = "crossing_spirals.csv"
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")
    
    plt.figure(figsize=(8, 8))
    plt.scatter(df['x'], df['y'], c=df['target'], cmap='Spectral', edgecolors='k', alpha=0.7)
    plt.title("Crossing Spirals")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('crossing_spirals.png')