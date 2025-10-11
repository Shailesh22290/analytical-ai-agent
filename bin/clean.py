import pandas as pd
import re

# Load the CSV
df = pd.read_csv("data/input/ftf_low_energy_table.csv")

# Function to clean numeric columns with units
def clean_numeric_column(series):
    """Remove units (Hz, etc.) and convert to float"""
    def extract_number(val):
        if pd.isna(val) or val == '-' or val == '':
            return None
        # Extract number from string like "18.75 Hz"
        match = re.search(r'[\d.]+', str(val))
        if match:
            return float(match.group())
        return None
    
    return series.apply(extract_number)

# Clean the amplitude columns
df['FTF as per data sheet'] = clean_numeric_column(df['FTF as per data sheet'])
df['Amplitude of FTF as per data sheet'] = clean_numeric_column(df['Amplitude of FTF as per data sheet'])
df['Adjacent frequencies within 2%'] = clean_numeric_column(df['Adjacent frequencies within 2%'])
df['Amplitude of Adjacent frequencies within 2%'] = clean_numeric_column(df['Amplitude of Adjacent frequencies within 2%'])

# Save the cleaned data
df.to_csv("data/input/ftf_low_energy_table_numeric.csv", index=False)

print("✅ Cleaned CSV saved to: data/input/ftf_low_energy_table_numeric.csv")
print("\nCleaned columns:")
print(df.dtypes)
print("\nSample data:")
print(df.head())