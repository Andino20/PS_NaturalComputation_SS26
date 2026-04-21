from sklearn.datasets import load_breast_cancer

# Load dataset as pandas DataFrame
breast_cancer = load_breast_cancer(as_frame=True)

# Combine features and target in one table
df = breast_cancer.frame.rename(columns={"target": "label"})

# Save as CSV
df.to_csv("breast_cancer.csv", index=False)

print("breast_cancer.csv created")
print(f"Shape: {df.shape}")
print("Label meanings:")
print("0 = malignant")
print("1 = benign")