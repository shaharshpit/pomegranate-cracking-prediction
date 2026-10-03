from google.colab import drive
drive.mount('/content/drive')

import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# ============================================================
# 1. PATHS SETUP (הגדרת נתיבים)
# ============================================================
META_S1_CSV       = '/content/drive/MyDrive/Thesis/results/metadata_identify_new.csv'
META_SEASON2_CSV  = '/content/drive/MyDrive/Thesis/results/metadata_season2/metadata_seq_season2_new.csv'
OUTPUT_BASE       = '/content/drive/MyDrive/Thesis/data/splits_identify_cv_with_test'

# ============================================================
# 2. CONFIGURATION & SETTINGS (הגדרות פייפליין)
# ============================================================
TEST_SIZE    = 0.15          
N_SPLITS     = 5             
RANDOM_SEED  = 42
SHUFFLE      = True
GROUP_COL    = 'pom_id'      # מזהה הרימון הייחודי למניעת זליגה אקדמית

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ============================================================
# 3. EXECUTION (הרצת הפיצול הוירטואלי המהיר)
# ============================================================
print("⏳ טוען את קובץ ה-Metadata ומכין את הפיצולים הוירטואליים...")
df = pd.read_csv(META_S1_CSV)

required_cols = ['Crack', 'seg_rgb_path', 'thermal_path', 'seg_thermal_path', GROUP_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"חסרות עמודות חובה ב-CSV: {missing}!")

# 🧹 ניקוי ואיפוס מוחלט של תיקיית הפלט הישנה
if os.path.exists(OUTPUT_BASE):
    print(f"🧹 מנקה תיקיית פלט ישנה בנתיב: {OUTPUT_BASE}")
    shutil.rmtree(OUTPUT_BASE)

# --- שלב א': חלוקת ה-Internal Test הכללי (15%) ---
unique_pomegranates = df[GROUP_COL].unique()
np.random.seed(RANDOM_SEED)
np.random.shuffle(unique_pomegranates)

test_group_count = int(len(unique_pomegranates) * TEST_SIZE)
test_groups = unique_pomegranates[:test_group_count]

test_df = df[df[GROUP_COL].isin(test_groups)].copy()
cv_df = df[~df[GROUP_COL].isin(test_groups)].copy().reset_index(drop=True)

print(f"🚀 מייצר קובץ מטא-דאטה ל-Internal Test...")
ensure_dir(os.path.join(OUTPUT_BASE, 'test'))
test_df.to_csv(os.path.join(OUTPUT_BASE, 'test', 'metadata.csv'), index=False)

# --- שלב ב': חלוקת 5 Folds בעזרת StratifiedGroupKFold ---
y_cv = cv_df['Crack'].astype(int).values
groups_cv = cv_df[GROUP_COL].values

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_SEED)

print("🚀 מייצר קבצי מטא-דאטה ל-5 ה-Cross Validation Folds...")
for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(cv_df, y_cv, groups_cv), start=1):
    fold_name = f"fold_{fold_idx:02d}"
    
    train_fold_df = cv_df.iloc[train_idx].copy()
    val_fold_df   = cv_df.iloc[val_idx].copy()

    # יצירת התיקיות ושמירת ה-CSV בלבד (לוקח מילישניות)
    ensure_dir(os.path.join(OUTPUT_BASE, fold_name, "train"))
    ensure_dir(os.path.join(OUTPUT_BASE, fold_name, "val"))
    
    train_fold_df.to_csv(os.path.join(OUTPUT_BASE, fold_name, "train", "metadata.csv"), index=False)
    val_fold_df.to_csv(os.path.join(OUTPUT_BASE, fold_name, "val", "metadata.csv"), index=False)
    print(f"   ✅ {fold_name} נוצר וירטואלית בהצלחה.")

# --- שלב ג': טיפול בעונה 2 (External Test) ---
if META_SEASON2_CSV is not None and os.path.exists(META_SEASON2_CSV):
    print("⏳ מייצר קובץ מטא-דאטה לעונה 2...")
    df_s2 = pd.read_csv(META_SEASON2_CSV)
    
    ensure_dir(os.path.join(OUTPUT_BASE, 'external_test_season2'))
    df_s2.to_csv(os.path.join(OUTPUT_BASE, 'external_test_season2', 'metadata.csv'), index=False)
else:
    print("💡 שים לב: קובץ המטא-דאטה של עונה 2 לא נמצא, דילגתי על שלב זה.")

print("\n🎉 כל קבצי ה-Cross Validation ועונה 2 נוצרו בהצלחה מטורפת ובשנייה אחת!")