import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from flirimageextractor import FlirImageExtractor
from tqdm import tqdm

# --- Configuration ---
base_cv_dir = Path("/content/drive/MyDrive/Thesis/data/splits_predict_cv")

# נגדיר את כל התיקיות שבהן יש metadata.csv שצריך לעדכן
folders_to_process = [
    "test",
    "external_test_season2"
]
for i in range(1, 6):
    folders_to_process.append(f"fold_{i:02d}/train")
    folders_to_process.append(f"fold_{i:02d}/val")

# Initialize the FLIR extractor once
te = FlirImageExtractor()

def add_thermal_stats_to_cv():
    for folder in folders_to_process:
        print(f"\n" + "="*50)
        print(f"--- Processing: {folder} ---")
        print("="*50)
        
        split_dir = base_cv_dir / folder
        csv_path = split_dir / "metadata.csv"
        
        if not csv_path.exists():
            print(f"Skipping {folder}: {csv_path} not found.")
            continue
            
        df = pd.read_csv(csv_path)
        
        # איפוס עמודות או יצירה אם לא קיימות
        df['temps_mean_new'] = np.nan
        df['temps_std_new'] = np.nan
        
        for idx in tqdm(df.index, desc=f"Updating {folder}"):
            row = df.loc[idx]
            
            # לוקחים את הנתיבים ישירות מהעמודות בטבלה
            th_path = str(row['thermal_path'])
            mask_path = str(row['seg_thermal_path'])
            
            # אם הנתיב הוא יחסי (ולא אבסולוטי מלא), נחבר אותו לתיקיית ה-split
            if not os.path.exists(th_path):
                th_path = os.path.join(split_dir, th_path)
            if not os.path.exists(mask_path):
                mask_path = os.path.join(split_dir, mask_path)
                
            # מדלג אם התמונות עדיין לא קיימות
            if not os.path.exists(th_path) or not os.path.exists(mask_path):
                continue

            try:
                # --- Processing Logic ---
                te.process_image(th_path)
                raw_map = te.get_thermal_np().astype(np.float32)

                # Load mask (1=fruit, 0=bg)
                mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.float32)
                
                # Safety: Ensure mask matches thermal image size
                if mask.shape != raw_map.shape:
                    mask = cv2.resize(mask, (raw_map.shape[1], raw_map.shape[0]), interpolation=cv2.INTER_NEAREST)

                if mask.sum() == 0:
                    continue # Skip empty masks (values remain NaN)

                # Mask background as NaN so it won't affect stats
                proc = raw_map.copy()
                proc[mask == 0] = np.nan
                
                # Calculate Stats (Fruit Only)
                fruit_mean = np.nanmean(proc)
                fruit_std = np.nanstd(proc)
                
                # Update the DataFrame
                df.at[idx, 'temps_mean_new'] = fruit_mean
                df.at[idx, 'temps_std_new'] = fruit_std

            except Exception as e:
                print(f"Error on index {idx} ({th_path}): {e}")

        # שמירת הקובץ המעודכן 
        df.to_csv(csv_path, index=False)
        print(f"Saved updated metadata to: {csv_path}")

if __name__ == "__main__":
    add_thermal_stats_to_cv()