import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import warnings

# ביטול אזהרות מיותרות של ספריות
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. CONFIGURATION (הגדרות פרודקשן)
# ============================================================
CONFIG = {
    "data_root": "/content/drive/MyDrive/Thesis/data/splits_predict_cv",
    "out_base_dir": "/content/drive/MyDrive/Thesis/results/models_resnet_pro_predict",
    "epochs": 15,
    "batch_size": 16,
    "lr": 1e-4,                      
    "weight_decay": 1e-4,
    "freeze_epochs": 2,              # הקפאת ה-Backbone להתייצבות ראשונית באפוקים 1 ו-2
    "seed": 42,
    "num_workers": 2,
    "img_h": 512,                    # שמירת פרופורציה 3:2 בדיוק לחצי מהמקור
    "img_w": 768       
    #"img_h": 768,                    # שמירת פרופורציה 3:2 בדיוק לחצי מהמקור
    #"img_w": 1024                  
}

# ============================================================
# 2. CUSTOM DATASET
# ============================================================
class PomegranateCSVDataset(Dataset):
    def __init__(self, csv_path: str, modality: str, transform=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing metadata file: {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.path_col = 'seg_rgb_path' if modality == "rgb" else 'seg_thermal_path'
        
        # סינון שורות חסרות או לא תקינות
        self.df = self.df[self.df[self.path_col].notna() & (self.df[self.path_col] != "Missing")].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row[self.path_col])
        
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception:
            # גיבוי למקרה של קובץ פגום כדי לא להקריס את הריצה בלילה
            image = Image.new("RGB", (CONFIG["img_w"], CONFIG["img_h"]), (0, 0, 0))

        # תווית מסוג Float עבור סיווג בינארי (BCE)
        label = torch.tensor([float(row['cracked_next_session'])], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# ============================================================
# 3. HELPERS & METRICS
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # להבטיח רפרודוקטיביות מושלמת

def plot_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=15)
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

def find_best_threshold_f1(y_true, y_prob):
    """
    סורק את כל הספים האפשריים על סמך הולידציה ומוצא את ה-Threshold שממקסם את ה-F1 Score
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = int(np.nanargmax(f1s[:-1])) if len(thresholds) > 0 else 0
    best_t = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    best_f1 = float(f1s[best_idx]) if len(f1s) > 0 else 0.0
    return best_t, best_f1

def freeze_unfreeze_backbone(model, epoch, freeze_until):
    """ מקפיא את הרשת כדי להגן על המשקלים בתחילת האימון, ומשחרר בהמשך """
    if epoch <= freeze_until:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

# ============================================================
# 4. MODEL & ENGINE (WITH AMP)
# ============================================================
def build_model(arch: str):
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, 1) # נוירון פלט בודד לסיווג בינארי
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return model

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool, scaler=None):
    model.train() if is_train else model.eval()
    running_loss = 0.0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train: optimizer.zero_grad(set_to_none=True) # אופטימיזציית זיכרון מהירה
        
        with torch.set_grad_enabled(is_train):
            # שימוש ב-AMP (Automatic Mixed Precision) לחסכון בזיכרון והאצת ריצה
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, labels)
            
            if is_train:
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        running_loss += loss.item() * images.size(0)
        
        # המרת ה-Logits להסתברויות ושיטוח בטוח ללא תלות בגודל ה-Batch
        probs = torch.sigmoid(logits).view(-1)
        labels_flat = labels.view(-1)

        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_labels.extend(labels_flat.cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    try: auc = roc_auc_score(all_labels, all_probs)
    except: auc = float("nan")

    return {"loss": epoch_loss, "auc": auc, "labels": all_labels, "probs": all_probs}

# ============================================================
# 5. MAIN PIPELINE
# ============================================================
def main():
    cfg = CONFIG.copy()
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(cfg["data_root"])
    
    # הפעלת סקיילר רק אם אנחנו רצים על ה-GPU (CUDA)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # הגדרת המודלים (להרצה רק של RGB אפשר לשים סולמית # בתחילת השורה של thermal)
    modalities_setup = {
        "rgb": {"arch": "resnet50", "use_color_jitter": True},
        #"thermal": {"arch": "resnet50", "use_color_jitter": False}
    }

    img_h, img_w = cfg["img_h"], cfg["img_w"]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # ולידציה: מראים את כל התמונה בדיוק ביחס 3:2. בלי שום חיתוך שמעלים סדקים!
    eval_tfms = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        normalize
    ])

    for modality, setup in modalities_setup.items():
        print(f"\n" + "="*75)
        print(f"🎬 STARTING EXPERIMENT FOR MODALITY: {modality.upper()} (Architecture: {setup['arch']})")
        print("="*75)
        
        # אימון: תזוזות גיאומטריות עדינות ששומרות על הסדק בתוך הפריים
        base_train_tfms = [
            transforms.Resize((img_h, img_w)),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.25)
        ]
        
        if setup["use_color_jitter"]:
            base_train_tfms.extend([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
            ])
            
        base_train_tfms.extend([transforms.ToTensor(), normalize])
        train_tfms = transforms.Compose(base_train_tfms)

        modality_out_dir = Path(cfg["out_base_dir"]) / f"{modality}_cv_results"
        os.makedirs(modality_out_dir, exist_ok=True)
        cv_records = []

        for fold_idx in range(2, 6):
            fold_name = f"fold_{fold_idx:02d}"
            fold_out_dir = modality_out_dir / fold_name
            os.makedirs(fold_out_dir, exist_ok=True)
            
            print(f"\n   ⏳ [{modality.upper()}] Training {fold_name.upper()}...")
            
            train_csv = root / fold_name / "train" / "metadata.csv"
            val_csv   = root / fold_name / "val"   / "metadata.csv"
            
            train_ds = PomegranateCSVDataset(str(train_csv), modality, transform=train_tfms)
            val_ds   = PomegranateCSVDataset(str(val_csv), modality, transform=eval_tfms)
            
            # drop_last=True באימון שומר עלינו מקריסות שכבת ה-BatchNorm במקרה של batch קטן בסוף
            train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
            val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
            
            model = build_model(setup["arch"]).to(device)
            
            # חישוב דינמי של הפיצוי לחוסר איזון (pos_weight) עבור הפולד הנוכחי
            num_pos = max(1, sum(1 for label in train_ds.df['cracked_next_session'] if label == 1))
            num_neg = len(train_ds.df) - num_pos
            pos_weight_val = torch.tensor([num_neg / num_pos]).to(device)
            
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
            optimizer = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
            
            best_fold_f1 = -1.0
            best_fold_metrics = {}
            
            for epoch in tqdm(range(1, cfg["epochs"] + 1), desc=f"      {fold_name} Progress", leave=False):
                # ניהול ההקפאה והשחרור של שכבות הבסיס
                freeze_unfreeze_backbone(model, epoch, cfg["freeze_epochs"])
                
                tr = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True, scaler=scaler)
                vl = run_epoch(model, val_loader, criterion, optimizer=None, device=device, is_train=False, scaler=scaler)
                
                scheduler.step()
                
                # אופטימיזציה לסף ה-F1 על סמך תוצאות הולידציה באפוק הנוכחי
                best_t, best_f1_epoch = find_best_threshold_f1(vl["labels"], vl["probs"])
                
                # גזירת כל המדדים בפועל לפי הסף האופטימלי שנמצא
                val_preds_best_t = (np.array(vl["probs"]) >= best_t).astype(int)
                epoch_acc = accuracy_score(vl["labels"], val_preds_best_t)
                epoch_prec, epoch_rec, _, _ = precision_recall_fscore_support(vl["labels"], val_preds_best_t, average="binary", zero_division=0)
                
                # הדפסה מפורטת ומסודרת של סטטוס האפוק הנוכחי מעל ה-tqdm בשבילך
                tqdm.write(f"        Epoch [{epoch:02d}/{cfg['epochs']}] -> Val Loss: {vl['loss']:.4f} | F1: {best_f1_epoch:.4f} | Prec: {epoch_prec:.4f} | Rec: {epoch_rec:.4f} | Best Thresh: {best_t:.2f}")
                
                if best_f1_epoch > best_fold_f1:
                    best_fold_f1 = best_f1_epoch
                    best_fold_metrics = {
                        "Fold": fold_name, "Best_Epoch": epoch,
                        "Val_Loss": vl["loss"], "Val_Acc": epoch_acc,
                        "Val_Precision": epoch_prec, "Val_Recall": epoch_rec,
                        "Val_F1": best_f1_epoch, "Val_AUC": vl["auc"],
                        "Best_Threshold": best_t
                    }
                    # שמירת המודל והמטריצה האופטימליים לפולד
                    torch.save(model.state_dict(), fold_out_dir / "best_model.pt")
                    cm_val = confusion_matrix(vl["labels"], val_preds_best_t)
                    plot_confusion_matrix(cm_val, ["healthy", "cracked"], fold_out_dir / "val_confusion_matrix.png")
                    
                    # ============================================================
                    # תוספת חדשה: שמירת קובץ ה-CSV עם התחזיות והנתיבים של כל הולידציה
                    # ============================================================
                    val_paths = val_ds.df[val_ds.path_col].tolist()
                    
                    # המרה למילים מפורשות לנוחות הקריאה
                    preds_text = ["Crack" if p == 1 else "Not Crack" for p in val_preds_best_t]
                    actual_text = ["Crack" if a == 1 else "Not Crack" for a in vl["labels"]]
                    
                    predictions_df = pd.DataFrame({
                        "Image_Path": val_paths,
                        "Actual": actual_text,
                        "Predicted": preds_text,
                        "Probability": vl["probs"]
                    })
                    predictions_df.to_csv(fold_out_dir / "best_epoch_predictions.csv", index=False)
            
            print(f"   ✅ [{modality.upper()} - {fold_name.upper()}] Finished! Best F1: {best_fold_metrics['Val_F1']:.3f} | Recall: {best_fold_metrics['Val_Recall']:.3f} | Precision: {best_fold_metrics['Val_Precision']:.3f} | Threshold: {best_fold_metrics['Best_Threshold']:.2f}")
            cv_records.append(best_fold_metrics)
            
            # ניקוי זיכרון ה-GPU במעבר בין Folds למניעת קריסות Out of Memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # יצירת דוח אקסל (CSV) מסודר עם התוצאות של כל ה-folds כולל ממוצע וסטיית תקן
        summary_df = pd.DataFrame(cv_records)
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
        
        mean_row = summary_df[numeric_cols].mean().to_dict()
        mean_row["Fold"] = "MEAN"
        std_row = summary_df[numeric_cols].std().to_dict()
        std_row["Fold"] = "STD"
        
        summary_df = pd.concat([summary_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
        summary_path = modality_out_dir / "cv_summary_report.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n📊 Summary Report for {modality.upper()} saved to: {summary_path}")
        print(summary_df[['Fold', 'Val_Acc', 'Val_F1', 'Val_Precision', 'Val_Recall', 'Best_Threshold']].to_string(index=False))

    print("\n" + "="*75)
    print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY! YOUR EXCEL REPORTS AND MODELS ARE READY.")
    print("="*75)

if __name__ == "__main__":
    main()