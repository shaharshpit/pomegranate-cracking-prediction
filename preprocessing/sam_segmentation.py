import json
import os
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO
from segment_anything import sam_model_registry, SamPredictor
import torch
from scipy import stats
import random

class SAMSegmenter:
    def __init__(self, config):
        self.image_dir = config["image_dir"]
        self.annotation_file = config["annotation_file"]
        self.output_mask_dir = config["output_mask_dir"]
        self.sam_checkpoint = config["sam_checkpoint"]
        self.model_type = config["model_type"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_size = config.get("train_size", 150)
        os.makedirs(self.output_mask_dir, exist_ok=True)
        self._load_data()
        self._load_model()
        self._split_data()
        self.avg_box = self._compute_average_bbox()
        self.median_box = self._compute_median_bbox()

    def _load_data(self):
        self.image_filenames = [
            filename for filename in os.listdir(self.image_dir)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        self.coco = COCO(self.annotation_file)
        self.image_id_map = {
            img["file_name"]: img["id"]
            for img in self.coco.dataset["images"]
            if img["file_name"] in self.image_filenames
        }

    def _split_data(self):
        random.seed(42)  # for reproducibility
        all_files = list(self.image_id_map.keys())
        random.shuffle(all_files)
        self.train_files = all_files[:self.train_size]
        self.test_files = all_files[self.train_size:]
        print(f"✅ Split: {len(self.train_files)} train images, {len(self.test_files)} test images")

    def _load_model(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

    def _compute_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union != 0 else 0

    def _compute_average_bbox(self):
        bboxes = []
        for filename in self.train_files:
            image_id = self.image_id_map[filename]
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                bbox = ann['bbox']
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = x_min + bbox[2]
                y_max = y_min + bbox[3]
                bboxes.append([x_min, y_min, x_max, y_max])

        if not bboxes:
            print("[!] No bounding boxes found in training annotations!")
            return None

        bboxes = np.array(bboxes)
        avg_box = np.mean(bboxes, axis=0)
        avg_box = avg_box.astype(int)
        return avg_box


    def _compute_median_bbox(self):
        bboxes = []
        for filename in self.train_files:
            image_id = self.image_id_map[filename]
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                bbox = ann['bbox']
                x_min = bbox[0]
                y_min = bbox[1]
                x_max = x_min + bbox[2]
                y_max = y_min + bbox[3]
                bboxes.append([x_min, y_min, x_max, y_max])
        if not bboxes:
            print("[!] No bounding boxes found in training annotations!")
            return None

        bboxes = np.array(bboxes)
        median_box = np.median(bboxes, axis=0)
        median_box = median_box.astype(int)
        print(f"\n✅ Median bounding box (computed on train set): {median_box.tolist()}")
        return median_box

    def run(self):
        ious = []
        for filename in tqdm(self.test_files):
            image_path = os.path.join(self.image_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[!] Could not read image {filename}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_rgb)
            mask_output = np.zeros_like(image_rgb)

            # if self.avg_box is not None:
            #     box = self.avg_box
            # else:
            #     height, width, _ = image_rgb.shape
            #     box = np.array([0, 0, width, height])

            # masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
            # best_idx = np.argmax(scores)
            # pred_mask = masks[best_idx]

            if self.median_box is not None:
                box = self.median_box
            else:
                height, width, _ = image_rgb.shape
                box = np.array([0, 0, width, height])

            masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
            best_idx = np.argmax(scores)
            pred_mask = masks[best_idx]

            # Compute IoU
            image_id = self.image_id_map[filename]
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)

            for ann in anns:
                gt_mask = self.coco.annToMask(ann)
                iou = self._compute_iou(pred_mask, gt_mask)
                ious.append(iou)

            # Save segmented image
            # masked_region = image_rgb.copy()
            # masked_region[~pred_mask] = 0
            # mask_output = masked_region

            # output_filename = f"{os.path.splitext(filename)[0]}_segmented.jpg"
            # output_path = os.path.join(self.output_mask_dir, output_filename)
            # cv2.imwrite(output_path, cv2.cvtColor(mask_output, cv2.COLOR_RGB2BGR))

        # Summary Statistics
        if ious:
            average_iou = np.mean(ious)
            iou_std = np.std(ious, ddof=1)
            confidence_interval = stats.norm.interval(
                0.95, loc=average_iou, scale=iou_std / np.sqrt(len(ious))
            )
            print(f"\n✅ Average IoU (on test set): {average_iou:.4f}")
            print(f"📊 IoU Standard Deviation: {iou_std:.4f}")
            print(f"95% Confidence Interval: [{confidence_interval[0]:.4f}, {confidence_interval[1]:.4f}]")
        else:
            print("\n⚠️ No annotations found in any image. Skipped IoU computation.")
    
    def run_inference_on_real_data(self, real_image_dir, real_output_dir):
        # Make sure output dir exists
        os.makedirs(real_output_dir, exist_ok=True)

        # List all images in real-life directory
        real_image_filenames = [
            filename for filename in os.listdir(real_image_dir)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        print(f"\n✅ Running inference on {len(real_image_filenames)} real-life images...")

        for filename in tqdm(real_image_filenames):
            image_path = os.path.join(real_image_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"[!] Could not read image {filename}")
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image_rgb)
            mask_output = np.zeros_like(image_rgb)

            # # Use same average box as set by the annotations
            # if self.avg_box is not None:
            #     box = self.avg_box
            # else:
            #     height, width, _ = image_rgb.shape
            #     box = np.array([0, 0, width, height])

            # Use same average box as set by the annotations
            if self.median_box is not None:
                box = self.median_box
            else:
                height, width, _ = image_rgb.shape
                box = np.array([0, 0, width, height])

            masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
            best_idx = np.argmax(scores)
            pred_mask = masks[best_idx]

            # Apply mask for saving
            masked_region = image_rgb.copy()
            masked_region[~pred_mask] = 0
            mask_output = masked_region

            output_filename = f"{os.path.splitext(filename)[0]}_segmented.jpg"
            output_path = os.path.join(real_output_dir, output_filename)
            cv2.imwrite(output_path, cv2.cvtColor(mask_output, cv2.COLOR_RGB2BGR))


        

