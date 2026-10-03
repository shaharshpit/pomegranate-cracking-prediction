import json
import os
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from segment_anything import sam_model_registry, SamPredictor
import torch
from scipy import stats

class CrossSeasonSAMSegmenter:
    def __init__(self, config):
        # מזינים ישירות את הקופסה מעונה 1
        self.median_box = np.array(config["s1_train_median_box"])
        
        self.s2_image_dir = config["s2_image_dir"]
        self.s2_annotation_file = config["s2_annotation_file"]
        
        self.sam_checkpoint = config["sam_checkpoint"]
        self.model_type = config["model_type"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"✅ Using predefined Median BBox: {self.median_box.tolist()}")
        
        self.s2_coco = COCO(self.s2_annotation_file)
        self._load_s2_data()
        self._load_model()

    def _load_s2_data(self):
        self.s2_image_filenames = [
            filename for filename in os.listdir(self.s2_image_dir)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        self.s2_image_id_map = {
            img["file_name"]: img["id"]
            for img in self.s2_coco.dataset["images"]
            if img["file_name"] in self.s2_image_filenames
        }

    def _load_model(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

    def _compute_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union != 0 else 0

    def run(self):
        ious = []
        files_to_process = list(self.s2_image_id_map.keys())
        
        for filename in tqdm(files_to_process, desc="Running Validation"):
            image_path = os.path.join(self.s2_image_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue

            # המרה ל-RGB (OpenCV קורא כ-BGR כברירת מחדל)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_rgb)

            # שימוש בקופסה שהזנו מראש מעונה 1
            box = self.median_box
            masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
            best_idx = np.argmax(scores)
            pred_mask = masks[best_idx]

            # השוואה ל-GT של עונה 2
            image_id = self.s2_image_id_map[filename]
            ann_ids = self.s2_coco.getAnnIds(imgIds=image_id)
            anns = self.s2_coco.loadAnns(ann_ids)

            for ann in anns:
                gt_mask = self.s2_coco.annToMask(ann)
                iou = self._compute_iou(pred_mask, gt_mask)
                ious.append(iou)

        # הדפסת תוצאות
        if ious:
            average_iou = np.mean(ious)
            iou_std = np.std(ious, ddof=1)
            confidence_interval = stats.norm.interval(
                0.95, loc=average_iou, scale=iou_std / np.sqrt(len(ious))
            )
            print(f"\n📊 --- VALIDATION RESULTS --- 📊")
            print(f"✅ Average IoU: {average_iou:.4f}")
            print(f"📉 IoU Standard Deviation: {iou_std:.4f}")
            print(f"📏 95% Confidence Interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]\n")
        else:
            print("\n⚠️ No annotations found to compute IoU.\n")

# ==========================================
# הרצה על עונה 2 - תרמי ו-RGB
# ==========================================

print("🔥 --- Starting Season 2 Validation for THERMAL --- 🔥")
thermal_config = {
    "s1_train_median_box": [237, 96, 777, 702], # הקופסה שקיבלת מהתרמי
    "s2_annotation_file": "/content/drive/MyDrive/Thesis/annotations/instances_default_thermal_season2.json", 
    "s2_image_dir": "/content/drive/MyDrive/Thesis/data/Season2/Thermal_photos", 
    "sam_checkpoint": "/content/drive/MyDrive/Thesis/models/sam_vit_h_4b8939-002.pth",
    "model_type": "vit_h"
}

thermal_segmenter = CrossSeasonSAMSegmenter(thermal_config)
thermal_segmenter.run()


print("🌈 --- Starting Season 2 Validation for RGB --- 🌈")
rgb_config = {
    "s1_train_median_box": [2182, 721, 5866, 4786], # הקופסה שקיבלת מה-RGB
    "s2_annotation_file": "/content/drive/MyDrive/Thesis/annotations/instances_default_rgb_season2.json", 
    "s2_image_dir": "/content/drive/MyDrive/Thesis/data/Season2/RGB_photos", 
    "sam_checkpoint": "/content/drive/MyDrive/Thesis/models/sam_vit_h_4b8939-002.pth",
    "model_type": "vit_h"
}

rgb_segmenter = CrossSeasonSAMSegmenter(rgb_config)
rgb_segmenter.run()