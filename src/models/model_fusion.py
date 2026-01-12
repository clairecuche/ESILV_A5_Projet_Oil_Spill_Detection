import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from transformers import SegformerForSemanticSegmentation
import cv2
from tqdm import tqdm
import sys
import json

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import des constantes de votre projet
from config import *
from metrics import SegmentationMetrics
from src.data.data_loaders import get_dataloaders

class LADOSModelFusion:
    def __init__(self, yolo_weights_path, segformer_weights_path):
        """
        Initialise la fusion des modèles LADOS.
        """
        self.device = torch.device(DEVICE)
        
        # 1. Chargement de YOLOv11
        print(f"🔄 Chargement de YOLOv11 depuis {yolo_weights_path}...")
        self.yolo_model = YOLO(yolo_weights_path)
        
        # 2. Chargement de SegFormer
        print(f"🔄 Chargement de SegFormer depuis {segformer_weights_path}...")
        self.seg_model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME_SEG, 
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        # Chargement des poids entraînés (gestion de l'état du dictionnaire)
        checkpoint = torch.load(segformer_weights_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        # Nettoyage si le modèle a été sauvegardé avec 'torch.compile'
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        self.seg_model.load_state_dict(state_dict)
        self.seg_model.eval()

    def predict(self, image_np):
        H, W = image_np.shape[:2]
        
        # --- A. SegFormer ---
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img_input = (image_np / 255.0 - mean) / std
        img_input = torch.from_numpy(img_input).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.seg_model(pixel_values=img_input)
            logits = F.interpolate(outputs.logits, size=(H, W), mode='bilinear', align_corners=False)
            seg_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # --- B. YOLO ---
        results = self.yolo_model.predict(image_np, conf=0.15, iou=0.4, verbose=False)[0]
        yolo_mask = np.zeros((H, W), dtype=np.uint8) # Initialisation du masque YOLO seul
        
        fusion_mask = seg_mask.copy()
        
        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for mask, yolo_cls in zip(masks, classes):
                semantic_cls = yolo_cls + 1
                mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
                
                # On remplit le masque YOLO pour la visualisation/debug
                yolo_mask[mask_resized] = semantic_cls
                
                # Logique de fusion LADOS
                if semantic_cls in [4, 5]: # Ship et Platform
                    fusion_mask[mask_resized] = semantic_cls
                elif semantic_cls in [1, 2, 3]: # Oil, Emulsion, Sheen
                    fusion_mask[(mask_resized) & (fusion_mask == 0)] = semantic_cls

        # ✅ On retourne bien les 3 masques
        return seg_mask, yolo_mask, fusion_mask
    
    def evaluate_fusion(self):
            """Évalue la fusion avec matrice de confusion détaillée."""
            metrics = SegmentationMetrics(num_classes=NUM_CLASSES)
            test_loader = get_dataloaders()['test']
            print(f"\n🔍 Évaluation sur {len(test_loader.dataset)} images de test...")

            with torch.no_grad():
                for images, masks in tqdm(test_loader, desc="Fusion Inference"):
                    for i in range(images.shape[0]):
                        img_np = images[i].permute(1, 2, 0).cpu().numpy()
                        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
                        img_rgb = ((img_np * std + mean) * 255).clip(0, 255).astype(np.uint8)
                        
                        gt_mask = masks[i].cpu().numpy()
                        _, _, pred_fusion = self.predict(img_rgb)
                        metrics.update(pred_fusion, gt_mask)

            results = metrics.get_results()
            
            print("\n" + "="*50)
            print("🏆 RÉSULTATS FINAUX DE LA FUSION")
            print("="*50)
            print(f"✅ mIoU Global : {results['mIoU']:.4f}")
            print(f"✅ mAcc Global  : {results['mAcc']:.4f}")
            
            print("\n📋 DÉTAILS PAR CLASSE (IoU) :")
            for i, iou in enumerate(results['class_iou']):
                # Utilisation de l'indexation de liste pour CLASS_NAMES_SEG
                name = CLASS_NAMES_SEG[i]
                print(f"  {name:15s}: {iou:.4f}")

            # --- AJOUT : MATRICE DE CONFUSION ---
            print("\n📊 MATRICE DE CONFUSION (Répartition des prédictions par pixel) :")
            conf_matrix = metrics.confusion_matrix # Récupération de la matrice brute
            
            for i in range(NUM_CLASSES):
                true_class = CLASS_NAMES_SEG[i]
                total_pixels = conf_matrix[i].sum()
                
                if total_pixels > 0:
                    print(f"\nClasse réelle : {true_class} ({total_pixels:.0f} pixels)")
                    # On affiche les 3 prédictions les plus fréquentes pour cette classe
                    percentages = (conf_matrix[i] / total_pixels) * 100
                    for j in range(NUM_CLASSES):
                        if percentages[j] > 1.0: # On affiche seulement si > 1%
                            pred_name = CLASS_NAMES_SEG[j]
                            marker = "⭐" if i == j else "→"
                            print(f"  {marker} {pred_name:15s}: {percentages[j]:5.1f}%")

            # Sauvegarde complète
            output_path = project_root / "output" / "fusion_results_detailed.json"
            with open(output_path, 'w') as f:
                json.dump({
                    'mIoU': float(results['mIoU']),
                    'mAcc': float(results['mAcc']),
                    'class_iou': [float(x) for x in results['class_iou']],
                    'confusion_matrix': conf_matrix.tolist()
                }, f, indent=2)

if __name__ == "__main__":
    YOLO_PATH = project_root / "output" / "yolo_best.pt"
    SEG_PATH = project_root / "output" / "segformer_best.pt"
    
    engine = LADOSModelFusion(YOLO_PATH, SEG_PATH)
    
    # Recherche d'images (Correction du chemin vers /test/images)
    test_dir = project_root / "LADOS-2" / "test" / "images"
    image_files = list(test_dir.glob('*.jpg'))
    
    if image_files:
        img_rgb = cv2.cvtColor(cv2.imread(str(image_files[0])), cv2.COLOR_BGR2RGB)
        seg, yolo, fused = engine.predict(img_rgb)
        
        # Affichage Segformer, YOLO, Fusion
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_rgb); axes[0].set_title("Original")
        axes[1].imshow(seg, cmap='tab10', vmin=0, vmax=5); axes[1].set_title("SegFormer (Fluides)")
        axes[2].imshow(yolo, cmap='tab10', vmin=0, vmax=5); axes[2].set_title("YOLO (Instances)")
        axes[3].imshow(fused, cmap='tab10', vmin=0, vmax=5); axes[3].set_title("Fusion LADOS")
        for ax in axes: ax.axis('off')
        plt.show()

    if YOLO_PATH.exists() and SEG_PATH.exists():
        engine.evaluate_fusion()