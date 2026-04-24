import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def generate_concentric_circles(n_samples=1000, noise=0.05, factor=0.4):
    n_per_circle = n_samples // 2
    
    theta = np.random.uniform(0, 2 * np.pi, n_per_circle)
    
    r_outer = 1.0 + np.random.normal(0, noise, n_per_circle)
    x_outer = r_outer * np.cos(theta)
    y_outer = r_outer * np.sin(theta)
    
    theta_inner = np.random.uniform(0, 2 * np.pi, n_per_circle)
    r_inner = factor + np.random.normal(0, noise, n_per_circle)
    x_inner = r_inner * np.cos(theta_inner)
    y_inner = r_inner * np.sin(theta_inner)
    
    X = np.vstack([
        np.column_stack([x_outer, y_outer]),
        np.column_stack([x_inner, y_inner])
    ])
    
    y = np.hstack([np.zeros(n_per_circle), np.ones(n_per_circle)])

    df = pd.DataFrame(X, columns=['x', 'y'])
    df['label'] = y.astype(int)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    df = generate_concentric_circles(n_samples=1000, noise=0.1, factor=0.5)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_df.to_csv("concentric_circles_train.csv", index=False)
    test_df.to_csv("concentric_circles_test.csv", index=False)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(df['x'], df['y'], c=df['label'], cmap='Spectral', edgecolors='k', alpha=0.7)
    plt.title("Concentric Circles")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('concentric_circles.png')