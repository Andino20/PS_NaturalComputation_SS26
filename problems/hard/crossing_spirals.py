import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def generate_crossing_spirals(n_samples=1000, noise=0.05):
    n_per_spiral = n_samples // 2
    
    # Range from 0 to ~25 radians for 4 full rotations
    theta = np.linspace(0, 4 * 2 * np.pi, n_per_spiral)
    r = theta / (4 * 2 * np.pi) # Normalize radius to [0, 1]
    
    # Spiral A (Standard)
    x_a = r * np.cos(theta)
    y_a = r * np.sin(theta)
    
    # Spiral B (Warped Intersections)
    # The modulation makes it cross A multiple times in non-obvious ways
    x_b = r * np.cos(theta + np.pi) * (1 + 0.15 * np.sin(theta * 3))
    y_b = r * np.sin(theta + np.pi) * (1 + 0.15 * np.cos(theta * 2.0))
    
    X = np.vstack([np.column_stack([x_a, y_a]), np.column_stack([x_b, y_b])])
    y = np.hstack([np.zeros(n_per_spiral), np.ones(n_per_spiral)])

    # Add Gaussian Noise
    X += np.random.normal(0, noise, X.shape)
    
    # Final DataFrame
    df = pd.DataFrame(X, columns=['x', 'y'])
    df['label'] = y.astype(int)
    return df.sample(frac=1).reset_index(drop=True)

if __name__ == "__main__":
    df = generate_crossing_spirals(n_samples=1000, noise=0.025)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_df.to_csv("crossing_spirals_train.csv", index=False)
    test_df.to_csv("crossing_spirals_test.csv", index=False)