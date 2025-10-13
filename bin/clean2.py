import pandas as pd

def clean_preserve_bearing_csv(path, output_path):
    # Load CSV
    df = pd.read_csv(path)
    
    # Replace NaN with empty string and strip whitespace
    df = df.fillna('').astype(str).applymap(lambda x: x.strip())

    # Drop rows where *all* columns are empty (but keep partial rows)
    df = df[df.apply(lambda row: ''.join(row.values).strip() != '', axis=1)]

    # Optionally, combine columns for embedding context — but don’t drop originals
    df['combined_text'] = df.apply(lambda row: ' '.join(v for v in row if v.strip()), axis=1)

    # Save cleaned version
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned (no data loss): {output_path} ({len(df)} rows)")

    return df.head(5)  # preview

# Clean all three without data loss
clean_preserve_bearing_csv("13/MDE_Bearing_Condition_Assessment.csv", "13/MDE_cleaned_full.csv")
clean_preserve_bearing_csv("13/MND_Bearing_Condition_Assessment.csv", "13/MND_cleaned_full.csv")
clean_preserve_bearing_csv("13/PND_Bearing_Condition_Assessment.csv", "13/PND_cleaned_full.csv")

