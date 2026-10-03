
import os
import shutil
import pandas as pd

class PomegranateImageExtractor:
    def __init__(self, excel_file, start_folder, output_folder):
        self.excel_file = excel_file
        self.start_folder = start_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self.df = pd.read_excel(self.excel_file)

        # Column definitions (camera columns not used anymore)
        self.thermal_num_col = "Num in Thermal camera"
        self.thermal_date_col = "Date  in Thermal camera"
        self.thermal_camera_col = "Thermal Camera"  # kept for compatibility, not used
        self.rgb_num_col = "Num in RGB"
        self.rgb_camera_col = "RGB Camera"          # kept for compatibility, not used
        self.rgb_date_col = "Date in RGB"
        self.session_col = "Session"
        self.id_col = "ID"  # or "Pomegranate ID"

        # Output folders
        self.rgb_folder = os.path.join(output_folder, "RGB_photos")
        self.thermal_folder = os.path.join(output_folder, "Thermal_photos")
        os.makedirs(self.rgb_folder, exist_ok=True)
        os.makedirs(self.thermal_folder, exist_ok=True)

        # Missing log list
        self.missing_log = []

        # Acceptable image extensions
        self._img_exts = ('.jpg', '.jpeg', '.png')

    def _iter_session_dirs(self):
        """
        Yield full paths of session directories under self.start_folder.
        Assumes children of start_folder are session-like folders.
        """
        for name in os.listdir(self.start_folder):
            p = os.path.join(self.start_folder, name)
            if os.path.isdir(p):
                yield p

    def find_rgb_path(self, photo_num):
        """
        Search for RGB image inside <Session...>/RGB that contains the zero-padded photo number.
        """
        if pd.isna(photo_num):
            return None

        photo_str = str(int(photo_num)).zfill(4)
        for session_path in self._iter_session_dirs():   
            # rgb_dir = os.path.join(session_path, "RGB")
            # print(rgb_dir)
            if not os.path.isdir(session_path):
                continue
            
            for file in os.listdir(session_path):
                fname_lower = file.lower()
                # Matches files that contain the photo number and have an image extension
                if photo_str in fname_lower and fname_lower.endswith(self._img_exts):
                    return os.path.join(session_path, file)

        return None

    def find_thermal_path(self, photo_num, photo_date):
        """
        Search for Thermal image inside <Session...>/Thermal/Thermal (or fallback to <Session...>/Thermal)
        with exact name pattern IR_dd-mm-YYYY_####.<ext> (case-insensitive).
        """
        if pd.isna(photo_num) or pd.isna(photo_date):
            return None

        photo_str = str(int(photo_num)).zfill(4)
        date_str = pd.to_datetime(photo_date, dayfirst=True).strftime("%d-%m-%Y")
        base_expected = f"ir_{date_str}_{photo_str}"

        for session_path in self._iter_session_dirs():
            # Prefer nested Thermal/Thermal; fall back to Thermal if needed
            thermal_dir = os.path.join(session_path, "Thermal", "Thermal")
            if not os.path.isdir(thermal_dir):
                thermal_dir = os.path.join(session_path, "Thermal")
            if not os.path.isdir(thermal_dir):
                continue

            for file in os.listdir(thermal_dir):
                fname_lower = file.lower()
                # Must start with IR_<date>_<num> and end with an image extension
                if fname_lower.startswith(base_expected) and fname_lower.endswith(self._img_exts):
                    return os.path.join(thermal_dir, file)

        return None

    def log_missing(self, pomegranate_id, session, image_type, reason):
        self.missing_log.append({
            "Pomegranate ID": pomegranate_id,
            "Session": session,
            "Image Type": image_type,
            "Reason": reason
        })

    def save_log(self, session_number):
        if self.missing_log:
            log_path = os.path.join(self.output_folder, f"missing_images_log_{session_number}.csv")
            pd.DataFrame(self.missing_log).to_csv(log_path, index=False)
            print(f"[i] Missing image log saved to: {log_path}")
        else:
            print("[i] No missing images to log!")

    def extract_images(self, session_number):
        session_df = self.df[self.df[self.session_col] == session_number]
        session_str = f"Session{session_number}"

        for _, row in session_df.iterrows():
            pomegranate_id = row.get(self.id_col)

            # === Thermal ===
            # thermal_num = row.get(self.thermal_num_col)
            # thermal_date = row.get(self.thermal_date_col)
            # if pd.isna(thermal_num):
            #     self.log_missing(pomegranate_id, session_number, "Thermal", "Missing number")
            # elif pd.isna(thermal_date):
            #     self.log_missing(pomegranate_id, session_number, "Thermal", "Missing date")
            # else:
            #     thermal_path = self.find_thermal_path(thermal_num, thermal_date)
            #     print(thermal_path)
            #     if thermal_path:
            #         new_name = f"{session_str}_{pomegranate_id}_thermal.jpg"
            #         shutil.copy(thermal_path, os.path.join(self.thermal_folder, new_name))
            #     else:
            #         self.log_missing(pomegranate_id, session_number, "Thermal", "File not found")

            # === RGB ===
            rgb_num = row.get(self.rgb_num_col)

            if pd.isna(rgb_num):
                self.log_missing(pomegranate_id, session_number, "RGB", "Missing number")
            else:
                rgb_path = self.find_rgb_path(rgb_num)
                if rgb_path:
                    new_name = f"{session_str}_{pomegranate_id}_rgb.jpg"
                    shutil.copy(rgb_path, os.path.join(self.rgb_folder, new_name))
                else:
                    self.log_missing(pomegranate_id, session_number, "RGB", "File not found")

        self.save_log(session_number)
