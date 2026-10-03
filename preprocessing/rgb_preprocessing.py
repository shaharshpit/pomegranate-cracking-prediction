import cv2
import numpy as np
import os

def remove_specular_highlights(rgb_image: np.ndarray, threshold: int = 245, 
                               dilate_k: int = 7, inpaint_radius: int = 5) -> np.ndarray:
    """הסרת כתמי בהק ועיבוד בשיטת inpainting."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((dilate_k, dilate_k), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    inpainted = cv2.inpaint(bgr, mask, inpaint_radius, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

def normalize_illumination_masked(rgb_image: np.ndarray, clip_limit: float = 2.0, grid_size: tuple = (8, 8)) -> np.ndarray:
    """הפעלת CLAHE רק על האובייקט ולא על הרקע השחור."""
    # יצירת מסכת אובייקט (כל מה שהוא לא שחור מוחלט)
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    _, obj_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    L_enhanced = clahe.apply(L)
    
    # שילוב בין המקור למשופר לפי המסכה
    mask_normalized = obj_mask / 255.0
    L_final = (L_enhanced * mask_normalized + L * (1 - mask_normalized)).astype(np.uint8)
    
    lab_final = cv2.merge([L_final, A, B])
    return cv2.cvtColor(lab_final, cv2.COLOR_LAB2RGB)

def process_folder(input_folder):
    output_folder = os.path.join(input_folder, 'RGB_SEG_preprocess')
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        # סינון: לעבד רק קבצי תמונה ולא מסכות קיימות
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            if '_mask' in filename.lower():
                continue
            
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)
            if img is None: continue
            
            # עיבוד
            rgb_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = remove_specular_highlights(rgb_np)
            result = normalize_illumination_masked(result)
            
            # שמירה
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result_bgr)
            print(f"Saved: {output_path}")

# הרצה
input_folder = r"/content/drive/MyDrive/Thesis/results/sam_rgb_season2"
process_folder(input_folder)