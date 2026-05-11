import pandas as pd
import os

print("Reading the original 150MB dataset...")
try:
    df = pd.read_csv('creditcard.csv')
    
    # Get all frauds (492 rows)
    frauds = df[df['Class'] == 1]
    print(f"Found {len(frauds)} fraud cases.")
    
    # Sample 5000 normal transactions
    normals = df[df['Class'] == 0].sample(n=5000, random_state=42)
    print(f"Sampled {len(normals)} normal cases.")
    
    # Combine and shuffle
    df_mini = pd.concat([frauds, normals]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save as temp_data.csv (which the API uses as fallback on live server)
    output_file = 'temp_data.csv'
    df_mini.to_csv(output_file, index=False)
    
    # Check file size
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\nSuccess! Compressed dataset saved as '{output_file}'.")
    print(f"New File Size: {size_mb:.2f} MB")
    print(f"Total Rows: {len(df_mini)}")
    
except Exception as e:
    print(f"Error: {e}")
