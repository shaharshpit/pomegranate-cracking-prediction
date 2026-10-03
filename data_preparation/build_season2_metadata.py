import os
import pandas as pd
from datetime import timedelta
from PIL import Image, ExifTags
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# 1. SETUP & PATHS (INPUTS עונה 2)
# ============================================================
MASTER_XLSX_S2     = "/content/drive/MyDrive/Thesis/data/Pomegrante master file season2.xlsx"
THERMAL_DIR_S2     = "/content/drive/MyDrive/Thesis/data/Season2/Thermal_photos"
SEG_THERMAL_DIR_S2 = "/content/drive/MyDrive/Thesis/results/sam_thermal_season2"
SEG_RGB_DIR_S2     = "/content/drive/MyDrive/Thesis/results/sam_rgb_season2/RGB_SEG_preprocess"

OUTPUT_DIR         = "/content/drive/MyDrive/Thesis/results/metadata_season2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH        = os.path.join(OUTPUT_DIR, "metadata_seq_season2_new2.csv")

# הגדרת רשימת העמודות המדויקת של עונה 1 (כפי שמופיעה בתמונה 2 ששלחת)
# זה ישמש כעוגן ליישור המבנה בסוף התהליך
SEASON1_COLUMNS_ORDER = [
    'Crack', 'pom_id', 'thermal_path', 'seg_thermal_path', 'seg_rgb_path', 'temp',
    'Treatment_Blue', 'Treatment_Yellow', 
    'Session_2', 'Session_3', 'Session_4', 'Session_5',
    'Loaded branch_1', 'Old branch_1', 'temp_scaled'
]

# ============================================================
# 2. HELPERS
# ============================================================
TAG_REVERSE = {v: k for k, v in ExifTags.TAGS.items()}

def get_capture_dt_pil(path: str) -> pd.Timestamp:
    img = Image.open(path)
    exif = img._getexif() or {}
    for name in ("DateTimeOriginal", "DateTime", "CreateDate"):
        tag_id = TAG_REVERSE.get(name)
        if tag_id and tag_id in exif:
            return pd.to_datetime(exif[tag_id], format="%Y:%m:%d %H:%M:%S")
    raise KeyError(f"No EXIF datetime in {path}")

def keep_until_first_crack(df: pd.DataFrame) -> pd.DataFrame:
    def _slice_group(g):
        mask = g["cracked_next_session"] == 1
        if mask.any():
            first_pos = mask.values.argmax()
            return g.iloc[: first_pos + 1]
        return g
    return df.groupby("ID", group_keys=False).apply(_slice_group).reset_index(drop=True)

def parse_id(id_str: str, session_int: int):
    # ניקוי המזהה והפיכתו למספר שלם נקי כמו בעונה 1 (למשל P10 הופך ל-10)
    pom_id = int(str(id_str).lstrip("P"))
    session = str(int(session_int))
    return pom_id, session

def thermal_name(session, pom_id):
    return f"Session{session}_P{pom_id}_thermal.jpg"

def seg_thermal_name(session, pom_id):
    return f"Session{session}_P{pom_id}_thermal_segmented.jpg"

def seg_rgb_name(session, pom_id):
    return f"Session{session}_P{pom_id}_rgb_segmented.jpg"

# ============================================================
# 3. LOAD EXCEL & CLEANING
# ============================================================
SHEET_MAIN = "final table"
SHEET_DT   = "date time"

df_full = pd.read_excel(MASTER_XLSX_S2, sheet_name=SHEET_MAIN)
df_datetime = pd.read_excel(MASTER_XLSX_S2, sheet_name=SHEET_DT)

# התאמה מדויקת לשמות העמודות בקובץ האקסל שלך
df = df_full[['ID','Session','Treatment','Loaded branch','Old branch','Date  in Thermal camera','Date  in RGB ','Num in Thermal camera','Num in RGB','Crack']].copy()

if "Date  in Thermal camera" in df.columns:
    df["Date  in Thermal camera"] = pd.to_datetime(df["Date  in Thermal camera"], errors="coerce")
    df["Date  in Thermal camera"] = df["Date  in Thermal camera"].dt.date

df["Session"] = pd.to_numeric(df["Session"], errors="coerce").astype("Int64")
df["ID"] = df["ID"].astype(str)
df = df.dropna(subset=["Session", "ID"]).reset_index(drop=True)

# תיקון חריגות של 24:00 בלילה בטבלאות הזמנים
df_datetime["date"] = df_datetime["date"].astype(str)
is_24 = df_datetime["date"].str.endswith("24:00")
df_datetime.loc[~is_24, "date"] = pd.to_datetime(df_datetime.loc[~is_24, "date"], errors="coerce")
df_datetime.loc[is_24, "date"] = (
    pd.to_datetime(df_datetime.loc[is_24, "date"].str.replace("24:00", "00:00"), errors="coerce")
    + timedelta(days=1)
)
df_datetime["capture_interval"] = pd.to_datetime(df_datetime["date"], errors="coerce")
df_temps = df_datetime[["capture_interval", "temp"]].copy()

# ============================================================
# 4. PATHS BUILDING
# ============================================================
df["pom_id"], df["session_str"] = zip(*df.apply(lambda r: parse_id(r["ID"], r["Session"]), axis=1))

df["thermal_path"] = df.apply(lambda r: os.path.join(THERMAL_DIR_S2, thermal_name(r["session_str"], r["pom_id"])), axis=1)
df["seg_thermal_path"] = df.apply(lambda r: os.path.join(SEG_THERMAL_DIR_S2, seg_thermal_name(r["session_str"], r["pom_id"])), axis=1)
df["seg_rgb_path"] = df.apply(lambda r: os.path.join(SEG_RGB_DIR_S2, seg_rgb_name(r["session_str"], r["pom_id"])), axis=1)

# ============================================================
# 5. EXIF EXTRACTION & MERGE TEMP
# ============================================================
df["capture_dt"] = pd.NaT
df["capture_interval"] = pd.NaT

for idx, row in df.iterrows():
    if not os.path.isfile(row["seg_rgb_path"]):
        df.at[idx, "seg_rgb_path"] = "Missing"
    if not os.path.isfile(row["seg_thermal_path"]):
        df.at[idx, "seg_thermal_path"] = "Missing"

    tpath = row["thermal_path"]
    try:
        dt = get_capture_dt_pil(tpath)
    except (FileNotFoundError, KeyError, OSError):
        df.at[idx, "thermal_path"] = "Missing"
        continue

    df.at[idx, "capture_dt"] = dt
    df.at[idx, "capture_interval"] = dt.floor("15T")

df["capture_interval"] = pd.to_datetime(df["capture_interval"], errors="coerce")
df = df.merge(df_temps, on="capture_interval", how="left")

temp_lookup = df_temps.set_index("capture_interval")["temp"]
df["temp"] = df["temp"].fillna(df["capture_interval"].sub(pd.Timedelta(minutes=15)).map(temp_lookup))

# ============================================================
# 6. CRACK LABELS & TIMESTEP SHIFTING
# ============================================================
df = df.sort_values(["ID", "Session"]).reset_index(drop=True)

# קידוד עמודת הסדק המקורית
df["Crack"] = df["Crack"].eq("X").astype(int)

# יצירת לייבל המטרה (העתיד)
df["cracked_next_session"] = df.groupby("ID")["Crack"].shift(-1)
df = df.dropna(subset=["cracked_next_session"]).reset_index(drop=True)
df = df[df["Crack"] == 0].reset_index(drop=True)
df = keep_until_first_crack(df)

# ============================================================
# 7. ONE-HOT ENCODING & FEATURE ALIGNMENT (יישור עמודות סופי)
# ============================================================
# טיפול בעמודות הטיפול (Treatment) - הפיכה לבוליאני לפי עונה 1
df['Treatment_Blue'] = df['Treatment'].astype(str).str.lower().eq('blue')
df['Treatment_Yellow'] = df['Treatment'].astype(str).str.lower().eq('yellow')

# טיפול בעמודות הסשנים (Sessions) - יצירת סשן 4 ו-5 חסרים כאפסים
df['Session_2'] = df['Session'].eq(2)
df['Session_3'] = df['Session'].eq(3)
df['Session_4'] = df['Session'].eq(4) 
df['Session_5'] = df['Session'].eq(5) 

# טיפול מיושר בעמודות הענפים (Loaded/Old branch)
# הקוד תומך בסימוני X וגם בסימוני מספר 1 (או 1.0) וממיר אותם לפורמט בוליאני אחיד
df['Loaded branch_1'] = df['Loaded branch'].astype(str).str.upper().isin(['X', '1', '1.0'])
df['Old branch_1'] = df['Old branch'].astype(str).str.upper().isin(['X', '1', '1.0'])

# סקיילינג לטמפרטורה
scaler = MinMaxScaler(feature_range=(0, 1))
if not df[["temp"]].isna().all().all():
    df["temp_scaled"] = scaler.fit_transform(df[["temp"]])
else:
    df["temp_scaled"] = 0.0

# ניקוי שורות חסרות (במידה ויש קבצים שלא נמצאו פיזית בדיסק)
df = df[~df.isin(["Missing"]).any(axis=1)].reset_index(drop=True)

# הוספת עמודת cracked_next_session זמנית לרשימת הסדר (כדי שלא תימחק בסינון ה-reindex)
final_columns = SEASON1_COLUMNS_ORDER.copy()
if 'cracked_next_session' not in final_columns:
    idx_to_insert = final_columns.index('seg_rgb_path') + 1
    final_columns.insert(idx_to_insert, 'cracked_next_session')

# סינון ועריכת סדר העמודות בצורה אגרסיבית לפי התבנית של עונה 1
df = df.reindex(columns=final_columns, fill_value=False)

# המרת עמודות ה-One-Hot לטיפוס בוליאני (כך שיציגו TRUE/FALSE באותיות גדולות כמו בעונה 1)
bool_cols = [c for c in df.columns if c.startswith('Treatment_') or c.startswith('Session_') or 'branch_' in c]
for col in bool_cols:
    df[col] = df[col].astype(bool)

# שמירה לקובץ הסופי
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ קובץ ה-Metadata של עונה 2 נוצר והותאם במאת האחוזים לעונה 1!")
print(f"נתיב שמירה: {OUTPUT_PATH}")
print("מבנה טבלה סופי:", df.shape)
print("רשימת עמודות בסדר החדש:\n", list(df.columns))