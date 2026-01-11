import pandas as pd
df = pd.read_csv('EVAL_target01_29.csv')

print("--- SUBMISSION SNAPSHOT ---")
print(f"Total Rows: {len(df)}")
print(f"Mean Value: {df['target'].mean():.4f}")
print(f"Value Range: [{df['target'].min():.4f}, {df['target'].max():.4f}]")
print("\nFirst 5 Predictions:")
print(df.head())