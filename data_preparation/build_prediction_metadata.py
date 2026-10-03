# -------- STEP 4 : Prepare dataframe with t+1 data
import pandas as pd
from datetime import timedelta
import pandas as pd
from flirimageextractor import FlirImageExtractor
import os
from PIL import Image, ExifTags
from sklearn.preprocessing import MinMaxScaler

TAG_REVERSE = {v:k for k,v in ExifTags.TAGS.items()}

flir = FlirImageExtractor()

def get_capture_dt_pil(path: str) -> pd.Timestamp:
    img = Image.open(path)
    exif = img._getexif() or {}
    # look for one of the standard capture‑time tags
    for name in ("DateTimeOriginal", "DateTime", "CreateDate"):
        tag_id = TAG_REVERSE.get(name)
        if tag_id and tag_id in exif:
            # “YYYY:MM:DD HH:MM:SS”
            return pd.to_datetime(exif[tag_id], format="%Y:%m:%d %H:%M:%S")
    raise KeyError(f"No EXIF datetime in {path}")


def keep_until_first_crack(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each fruit ID, keeps all rows up to and including
    the first row where cracked_next_session == 1.
    If a fruit never cracks, keeps all its rows.
    """
    def _slice_group(g):
        mask = g["cracked_next_session"] == 1
        if mask.any():
            first_pos = mask.values.argmax()
            return g.iloc[: first_pos + 1]
        else:
            return g

    return (
        df
        .groupby("ID", group_keys=False)
        .apply(_slice_group)
        .reset_index(drop=True)
    )

def parse_id(sess_str):
    pom_part, sess_part = sess_str.split("_", 1)
    pom_id  = pom_part.lstrip("P")
    session = sess_part
    return pom_id, session

df_full = pd.read_excel("/content/drive/MyDrive/Thesis/data/Pomegrantes master file final.xlsx", sheet_name="final table")
df_datetime = pd.read_excel("/content/drive/MyDrive/Thesis/data/Pomegrantes master file final.xlsx", sheet_name="date time")
keep_idx = [0, 1] + list(range(3, 9)) + [10]+[14]
df = df_full.iloc[:, keep_idx]
print(df)
df['Date  in Thermal camera'] = pd.to_datetime(df['Date  in Thermal camera'])
df['Date  in Thermal camera'] = df['Date  in Thermal camera'].dt.date
#df = df.dropna(subset=["Date  in Thermal camera"]).reset_index(drop=True)
df["Session"] = df["Session"].astype(int)       
df["ID"] = df["ID"].astype(str)
df["ID_session"] = df["ID"].astype(str) + "_" + df["Session"].astype(str)
df_datetime['date'] = df_datetime['date'].astype(str)
is_24 = df_datetime['date'].str.endswith('24:00')
df_datetime.loc[~is_24, 'date'] = pd.to_datetime(df_datetime.loc[~is_24, 'date'])
df_datetime.loc[is_24, 'date'] = pd.to_datetime(
    df_datetime.loc[is_24, 'date'].str.replace('24:00', '00:00')
) + timedelta(days=1)
df_datetime['date_time'] = pd.to_datetime(df_datetime['date']) 

output_path = "/content/drive/MyDrive/Thesis/results/metadata_seq_new.csv"
thermal_dir      = "/content/drive/MyDrive/Thesis/data/Thermal_photos/"
seg_thermal_dir  = "/content/drive/MyDrive/Thesis/results/final_seg_thermal/"
seg_rgb_dir      = "/content/drive/MyDrive/Thesis/results/final_seg_RGB/RGB_SEG_preprocess"

df[["pom_id", "session"]] = df["ID_session"].apply(lambda x: pd.Series(parse_id(x)))
df["thermal_path"]     = df.apply(
    lambda row: os.path.join(
        thermal_dir,
        f"Session{row.session}_P{row.pom_id}_thermal.jpg"
    ),
    axis=1
)
df["seg_thermal_path"] = df.apply(
    lambda row: os.path.join(
        seg_thermal_dir,
        f"Session{row.session}_P{row.pom_id}_thermal_segmented.jpg"
    ),
    axis=1
)
df["seg_rgb_path"]     = df.apply(
    lambda row: os.path.join(
        seg_rgb_dir,
        f"Session{row.session}_P{row.pom_id}_rgb_segmented.jpg"
    ),
    axis=1
)
df['capture_dt']       = pd.NaT
df['capture_interval'] = pd.NaT

for idx, row in df.iterrows():
    rpath = row.get('seg_rgb_path', None)
    if not rpath or not os.path.isfile(rpath):
        print(f"Warning [rgb missing]: {rpath}")
        df.at[idx, 'seg_rgb_path'] = 'Missing'
    tpath = row.get('seg_thermal_path', None)
    if not tpath or not os.path.isfile(tpath):
        print(f"Warning [thermal missing]: {tpath}")
        df.at[idx, 'seg_thermal_path'] = 'Missing'
    path = row['thermal_path']
    try:
        dt = get_capture_dt_pil(path)
    except (FileNotFoundError, KeyError, OSError) as e:
        # skip files that can’t be opened or lack EXIF datetime
        print(f"Warning [{path}]: {e}")
        df.at[idx, 'thermal_path'] = 'Missing'
        continue
    # 3) floor to 10‑minute interval
    interval = dt.floor('10T')

    # 4) write back into df_photos
    df.at[idx, 'capture_dt'] = dt
    df.at[idx, 'capture_interval'] = interval


# 3) Force both columns to datetime dtype
df['capture_interval'] = pd.to_datetime(df['capture_interval'])
df_temps = (
    df_datetime
    .rename(columns={"date": "capture_interval"})
    .assign(capture_interval=lambda d: pd.to_datetime(d["capture_interval"]))
)

# 4) Merge on that floored‐to‐10min timestamp
df = df.merge(
    df_temps[['capture_interval', 'temp']],
    on='capture_interval',
    how='left'
)

temp_lookup = df_temps.set_index('capture_interval')['temp']

# 3) remember which rows were missing after the first merge
was_missing = df['temp'].isna()

# 4) fill from capture_interval – 10 min
df['temp'] = df['temp'].fillna(
    df['capture_interval']
      .sub(pd.Timedelta(minutes=10))
      .map(temp_lookup)
)


df = df.sort_values(["ID", "Session"])
df["Crack"] = df["Crack"].eq("X").astype(int)
df["cracked_next_session"] = (
    df
    .groupby("ID")["Crack"]    
    .shift(-1)
)

df = df.dropna(subset=["cracked_next_session"]).reset_index(drop=True)
df = df[df["Crack"] == 0].reset_index(drop=True)
df = keep_until_first_crack(df)
cat_cols = ["Treatment","Session","Loaded branch","Old branch"]
for c in cat_cols:
    df[c] = df[c].astype("category")
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
scaler = MinMaxScaler(feature_range=(0, 1))
df['temp_scaled'] = scaler.fit_transform(df[['temp']])
df = df[~df.isin(['Missing']).any(axis=1)].reset_index(drop=True)
df = df.drop(columns=["Crack","Date  in Thermal camera","ID","plot","Tree","Date added to the field","capture_dt","capture_interval"])
#df = df.drop(columns=["Crack","ID","Date  in Thermal camera","pom_id","session","plot","Tree","Date added to the field","ID_session","capture_dt","capture_interval"])
df.to_csv(output_path, index=False)