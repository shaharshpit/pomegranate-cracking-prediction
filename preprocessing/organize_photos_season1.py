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

        # Column definitions
        self.thermal_num_col = "Num in Thermal camera"
        self.thermal_date_col = "Date in Thermal camera"
        self.thermal_camera_col = "Thermal Camera"

        self.rgb_num_col = "Num in RGB"
        self.rgb_camera_col = "RGB Camera"
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

    def find_rgb_path(self, photo_num, camera=None):
        if pd.isna(photo_num):
            return None
        camera_str = str(int(camera))
        photo_str = str(int(photo_num)).zfill(4)

        for day_folder in os.listdir(self.start_folder):
            day_path = os.path.join(self.start_folder, day_folder)
            if not os.path.isdir(day_path):
                continue

            rgb_folder = os.path.join(day_path, "RGB")
            if not os.path.exists(rgb_folder):
                continue

            search_dirs = [rgb_folder]
            if camera and str(camera).strip():
                cam_dir = os.path.join(rgb_folder, f"Camera {camera_str}")
                if os.path.exists(cam_dir):
                    search_dirs.insert(0, cam_dir)

            for dir_path in search_dirs:
                for file in os.listdir(dir_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')) and photo_str in file:
                        return os.path.join(dir_path, file)
        return None

    def find_thermal_path(self, photo_num, photo_date, camera=None):
        if pd.isna(photo_num) or pd.isna(photo_date):
            return None
        camera_str = str(int(camera))
        photo_str = str(int(photo_num)).zfill(4)
        date_str = pd.to_datetime(photo_date, dayfirst=True).strftime("%d-%m-%Y")
        expected_name = f"IR_{date_str}_{photo_str}.jpg".lower()
        for day_folder in os.listdir(self.start_folder):
            day_path = os.path.join(self.start_folder, day_folder)
            if not os.path.isdir(day_path):
                continue
            thermal_folder = os.path.join(day_path, "Thermal")
            if not os.path.exists(thermal_folder):
                continue
            search_dirs = [thermal_folder]
            if camera and str(camera).strip():
                cam_dir = os.path.join(thermal_folder, f"Camera {camera_str}")
                if os.path.exists(cam_dir):
                    search_dirs.insert(0, cam_dir)
            for dir_path in search_dirs:
                for file in os.listdir(dir_path):
                    if file.lower() == expected_name:
                        return os.path.join(dir_path, file)
        return None

    def log_missing(self, pomegranate_id, session, image_type, reason):
        self.missing_log.append({
            "Pomegranate ID": pomegranate_id,
            "Session": session,
            "Image Type": image_type,
            "Reason": reason
        })

    def save_log(self,session_number):
        if self.missing_log:
            log_path = os.path.join(self.output_folder, f"missing_images_log_{session_number}.csv")
            pd.DataFrame(self.missing_log).to_csv(log_path, index=False)
            print(f"[i] Missing image log saved to: {log_path}")
        else:
            print("[i] No missing images to log!")

    def extract_images(self, session_number):
        session_df = self.df[self.df[self.session_col] == session_number]
        for idx, row in session_df.iterrows():
            pomegranate_id = row.get(self.id_col)
            session_str = f"Session{session_number}"

            # === Thermal
            thermal_num = row.get(self.thermal_num_col)
            thermal_date = row.get(self.thermal_date_col)
            thermal_camera = row.get(self.thermal_camera_col)
            if pd.isna(thermal_num):
                self.log_missing(pomegranate_id, session_number, "Thermal", "Missing number")
            elif pd.isna(thermal_date):
                self.log_missing(pomegranate_id, session_number, "Thermal", "Missing date")
            else:
                thermal_path = self.find_thermal_path(thermal_num, thermal_date, thermal_camera)
                if thermal_path:
                    new_name = f"{session_str}_{pomegranate_id}_thermal.jpg"
                    shutil.copy(thermal_path, os.path.join(self.thermal_folder, new_name))
                else:
                    self.log_missing(pomegranate_id, session_number, "Thermal", "File not found")

            # === RGB
            rgb_num = row.get(self.rgb_num_col)
            rgb_camera = row.get(self.rgb_camera_col)

            if pd.isna(rgb_num):
                self.log_missing(pomegranate_id, session_number, "RGB", "Missing number")
            else:
                rgb_path = self.find_rgb_path(rgb_num, rgb_camera)
                if rgb_path:
                    new_name = f"{session_str}_{pomegranate_id}_rgb.jpg"
                    shutil.copy(rgb_path, os.path.join(self.rgb_folder, new_name))
                else:
                    self.log_missing(pomegranate_id, session_number, "RGB", "File not found")

        self.save_log(session_number)
