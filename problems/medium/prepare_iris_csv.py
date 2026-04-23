from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)
df = iris.frame.rename(columns={"target": "label"})

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_df.to_csv("iris_train.csv", index=False)
test_df.to_csv("iris_test.csv", index=False)

print("iris.csv created")