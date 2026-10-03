import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold

# ============================================================
# CONFIGURATION - התאמי את הנתיבים לאלו שאצלך בדרייב
# ============================================================
CONFIG = {
    "full_metadata_path": "/content/drive/MyDrive/Thesis/results/metadata_seq_new.csv",
    # כאן נמצא ה-Test של העונה הראשונה (כדי שנזהה את הרימונים שיש שם)
    "old_splits_dir": "/content/drive/MyDrive/Thesis/data/splits_identify_cv_with_test",
    # כאן נשמור את החלוקה החדשה לחיזוי
    "new_splits_dir": "/content/drive/MyDrive/Thesis/data/splits_predict_cv",
    "n_splits": 5,
    "target_col": "cracked_next_session",
    "group_col": "pom_id" 
}

def create_predict_cv_splits():
    print("🚀 מתחיל ביצירת ה-Splits לחיזוי...")
    
    # 1. טעינת הנתונים המלאים
    df_full = pd.read_csv(CONFIG["full_metadata_path"])
    print(f"סה\"כ שורות בקובץ המקורי: {len(df_full)}")
    
    # סינון שורות חסרות ב-target (למשל, סשן אחרון שאין לו עתיד)
    df_full = df_full.dropna(subset=[CONFIG["target_col"]]).reset_index(drop=True)
    print(f"לאחר ניקוי שורות ללא target: {len(df_full)}")

    # 2. טעינת ה-Test Set הישן ואיתור הרימונים שיש להוציא
    test_csv_path = Path(CONFIG["old_splits_dir"]) / "test" / "metadata.csv"
    if not test_csv_path.exists():
        test_csv_path = Path(CONFIG["old_splits_dir"]) / "test_metadata.csv"
        
    if test_csv_path.exists():
        df_test_old = pd.read_csv(test_csv_path)
        # בהנחה שגם שם יש עמודת ID. אם לא, צריך לחלץ אותו מהשם של התמונה.
        test_pomegranates = set(df_test_old[CONFIG["group_col"]].unique())
        print(f"נמצאו {len(test_pomegranates)} רימונים שונים ב-Test Set.")
    else:
        print("⚠️ קובץ ה-Test הישן לא נמצא. נא לוודא נתיב. התהליך ייעצר כדי למנוע דלף.")
        return

    # 3. סינון הרימונים של ה-Test מהדאטה הכללי ויצירת קובץ Test חדש
    # כל השורות של הרימונים שבטסט
    df_test_new = df_full[df_full[CONFIG["group_col"]].isin(test_pomegranates)].copy()
    # כל השאר (הם יעברו ל-CV)
    df_cv = df_full[~df_full[CONFIG["group_col"]].isin(test_pomegranates)].copy()
    
    print(f"שורות שהוקצו ל-Test החדש: {len(df_test_new)}")
    print(f"שורות שהוקצו ל-Cross Validation: {len(df_cv)}")

    # שמירת תיקיית ה-Test החדשה
    out_dir = Path(CONFIG["new_splits_dir"])
    os.makedirs(out_dir / "test", exist_ok=True)
    df_test_new.to_csv(out_dir / "test" / "metadata.csv", index=False)

    # 4. חלוקת ה-CV (GroupKFold)
    gkf = GroupKFold(n_splits=CONFIG["n_splits"])
    
    groups = df_cv[CONFIG["group_col"]].values
    X = df_cv.drop(columns=[CONFIG["target_col"]]).values
    y = df_cv[CONFIG["target_col"]].values

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
        fold_name = f"fold_{fold_idx:02d}"
        print(f"\nמעבד את {fold_name}...")
        
        df_train = df_cv.iloc[train_idx].copy()
        df_val = df_cv.iloc[val_idx].copy()
        
        # וידוא שאין דלף של רימונים בין train ל-val
        train_groups = set(df_train[CONFIG["group_col"]])
        val_groups = set(df_val[CONFIG["group_col"]])
        leakage = train_groups.intersection(val_groups)
        if leakage:
            print(f"❌ אזהרה: דלף זוהה! רימונים חופפים: {leakage}")
        else:
            print("✅ חלוקה נקייה. אין חפיפת רימונים.")
            print(f"   Train: {len(df_train)} שורות. חיוביים (Cracked): {df_train[CONFIG['target_col']].sum()}")
            print(f"   Val:   {len(df_val)} שורות. חיוביים (Cracked): {df_val[CONFIG['target_col']].sum()}")

        # שמירת קבצי ה-Fold
        fold_dir = out_dir / fold_name
        os.makedirs(fold_dir / "train", exist_ok=True)
        os.makedirs(fold_dir / "val", exist_ok=True)
        
        df_train.to_csv(fold_dir / "train" / "metadata.csv", index=False)
        df_val.to_csv(fold_dir / "val" / "metadata.csv", index=False)

    print(f"\n🎉 החלוקה הסתיימה בהצלחה! הנתונים נשמרו ב:\n{out_dir}")

if __name__ == "__main__":
    create_predict_cv_splits()