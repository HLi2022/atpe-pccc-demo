import pandas as pd
import numpy as np

print("--- Checking Semiconducting ---")
try:
    df_semi = pd.read_csv("data/swcnt_info_semiconducting.csv", header=None)
    print("Shape:", df_semi.shape)
    print("First 2 rows:")
    print(df_semi.head(2))
    
    # Check if first row is header or data (numeric check)
    is_numeric = np.issubdtype(df_semi.iloc[0, 0], np.number) or isinstance(df_semi.iloc[0, 0], (int, float))
    print(f"Row 0 Col 0 is numeric? {is_numeric} (Value: {df_semi.iloc[0,0]})")
except Exception as e:
    print(f"Error reading semi: {e}")

print("\n--- Checking Metallic ---")
try:
    df_met = pd.read_csv("data/swcnt_info_metallic.csv")
    print("Shape:", df_met.shape)
    print("Columns:", df_met.columns.tolist())
    print("First 2 rows:")
    print(df_met.head(2))
except Exception as e:
    print(f"Error reading met: {e}")
