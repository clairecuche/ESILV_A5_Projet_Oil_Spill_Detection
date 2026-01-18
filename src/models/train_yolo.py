import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2

# Ajoute la racine du projet au chemin de recherche de Python
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loaders import get_dataloaders
from src.data.preprocessing import calculate_class_weights, convert_coco_to_yolo_segmentation
from config import *
from metrics import SegmentationMetrics


class YOLOv11Trainer:
    """
    Entraîneur YOLOv11 pour la segmentation d'instance selon les paramètres du paper LADOS.
    Utilise weighted BCE loss et les mêmes métriques (IoU, mAcc) que SegFormer.
    """
    
    def __init__(self):
        self.device = torch.device(DEVICE)
        
        # Clarifier les num_classes
        # YOLO entraîne sur 5 classes (sans Background)
        # Mais on évalue sur 6 classes (avec Background) pour comparer avec SegFormer
        self.num_classes_yolo = NUM_CLASSES_YOLO  # 5 (pour YOLO)
        self.num_classes_eval = NUM_CLASSES       # 6 (pour évaluation)
        
        # 1. Modèle YOLOv11 (pré-entraîné sur COCO)
        self.model = self._initialize_model()
        
        # 2. Poids de classe pour weighted BCE loss
        weights_path = DATA_DIR / 'yolo_class_weights.pt'
        if weights_path.exists():
            self.class_weights = torch.load(weights_path)
        else:
            self.class_weights = calculate_class_weights(split='train', use_paper_method=True)
            if len(self.class_weights) == 6:
                self.class_weights = self.class_weights[1:]
            torch.save(self.class_weights, weights_path)
        
        # 3. Métriques et Suivi
        self.val_metrics = SegmentationMetrics(num_classes=self.num_classes_eval)
        self.best_miou = -1.0
        self.best_map = -1.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 
            'val_loss': [], 
            'train_miou': [], 
            'val_miou': [],
            'val_map50': [],
            'val_map50_95': []
        }
        
        # 4. Directories
        self.checkpoint_dir = OUTPUT_DIR_YOLO / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"YOLOv11 Trainer initialisé sur {self.device}.")
        print(f"Classes YOLO (entraînement): {self.num_classes_yolo}")
        print(f"Classes évaluation (avec BG): {self.num_classes_eval}")
        print(f"Poids de classe: {self.class_weights}")

    def _initialize_model(self):
        """Initialise YOLOv11m-seg avec poids COCO."""
        model = YOLO('yolo11m-seg.pt')  
        return model
    
    def _prepare_yolo_format(self):
        """
        Prépare les données au format YOLO (conversion des masques en polygones).
        Utilise le module yolo_data_converter.
        """
        yolo_data_dir = DATA_DIR / 'yolo_format'
        yaml_path = yolo_data_dir / 'data.yaml'
        
        # Vérifier si la conversion est déjà faite
        if yaml_path.exists():
            print(f"✅ Données YOLO déjà converties: {yolo_data_dir}")
            return str(yaml_path)
        
        # Sinon, lancer la conversion
        print("🔄 Conversion COCO → YOLO en cours...")
        convert_coco_to_yolo_segmentation(DATA_DIR, yolo_data_dir)
        
        return str(yaml_path)
    
    def train(self):
        """
        Entraînement YOLOv11 avec les paramètres du paper LADOS:
        - Batch size: 32
        - Epochs: max 80 (avec early stopping patience=10)
        - Optimizer: AdamW implicite dans YOLO
        - Augmentations: mosaic, HSV, flip, translate, scale
        - Early stopping basé sur weighted average mAP (0.1*mAP@50 + 0.9*mAP@50-95)
        """
        print("\n🚀 Début de l'entraînement YOLOv11...")
        start_time = datetime.now()
        
        # Préparer le format YOLO
        data_yaml_path = self._prepare_yolo_format()
        
        # Configuration d'entraînement selon le paper
        results = self.model.train(
            data=data_yaml_path,
            epochs=NUM_EPOCHS_YOLO,  
            batch=BATCH_SIZE_YOLO,   
            imgsz=TARGET_SIZE[0],
            device=[0, 1],
            
            # Optimizer
            optimizer='AdamW',
            lr0=LEARNING_RATE_YOLO,
            weight_decay=WEIGHT_DECAY,
            
            # Early stopping
            patience=PATIENCE,
            
            # Augmentations 
            mosaic=1.0,   # Mosaic augmentation
            hsv_h=0.015,  # HSV-Hue augmentation
            hsv_s=0.7,    # HSV-Saturation
            hsv_v=0.4,    # HSV-Value
            degrees=0.0,  # Rotation (pas mentionné dans paper)
            translate=0.1,  # Translation
            scale=0.5,    # Scaling
            fliplr=0.5,   # Horizontal flip
            
            # Loss weights (BCE pour masks)
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # Sauvegarde
            project=str(OUTPUT_DIR_YOLO),
            name='yolo_training',
            exist_ok=True,
            save=True,
            save_period=5,
            
            # Validation
            val=True,
            plots=True,
            verbose=True
        )
        
        training_time = (datetime.now() - start_time).total_seconds() / 3600
        print(f"\n--- ENTRAÎNEMENT TERMINÉ ---")
        print(f"Temps total : {training_time:.2f} heures.")
        
        # Évaluation finale sur test set
        self._evaluate_with_metrics()
        
        return results

    def _evaluate_with_metrics(self):
        """
        Évalue le modèle avec dénormalisation et conversion sémantique CORRIGÉE.
        """
        print("\n" + "="*50)
        print("📊 ÉVALUATION FINALE SUR TEST SET")
        print("="*50)
        
        # 1. Charger le meilleur modèle sauvegardé
        # ✅ Essayer plusieurs chemins possibles
        possible_paths = [
            OUTPUT_DIR_YOLO / 'yolo_training' / 'weights' / 'yolo_best.pt',
            Path("/kaggle/working/ESILV_A5_Projet_Oil_Spill_Detection/outputs/yolo/yolo_training/weights/yolo_best.pt"),
            Path(r"C:/Users/benoi/OneDrive-DeVinci/A5ESILV/CV/Project/ESILV_A5_Projet_Oil_Spill_Detection/outputs/yolo/yolo_training/weights/yolo_best.pt"),
            Path("/kaggle/working/yolo/yolo_training/weights/yolo_best.pt"),
            Path("outputs/yolo/yolo_training/weights/yolo_best.pt"),
            Path("output/yolo_best.pt")
        ]
        
        best_model_path = None
        for path in possible_paths:
            if path.exists():
                best_model_path = path
                break
        
        if best_model_path is None:
            raise FileNotFoundError("❌ Aucun modèle YOLO trouvé dans les chemins possibles!")
            
        print(f"🔄 Chargement des poids : {best_model_path}")
        model = YOLO(str(best_model_path))

        # 2. Paramètres de dénormalisation (ImageNet standards)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # 3. Charger le test loader
        test_loader = get_dataloaders()['test']
        
        print(f"\n🔍 Vérification du split:")
        print(f"   Nombre d'images test: {len(test_loader.dataset)}")
        print(f"   Attendu (paper LADOS): 343 images")
        if len(test_loader.dataset) != 343:
            print(f"   ⚠️ WARNING: Le nombre d'images ne correspond pas au paper!")
        
        self.val_metrics.reset()
        
        total_predictions = 0
        total_background_pixels = 0
        total_non_background_pixels = 0
        
        print("\n🔍 Prédiction et conversion sémantique...")
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
                for i in range(images.shape[0]):
                    # On récupère l'image du batch (déjà chargée en 0-1 par le DataLoader)
                    img_np = images[i].permute(1, 2, 0).cpu().numpy()
                    # On annule Normalize : img = (img_norm * std) + mean
                    img_rescaled = (img_np * std) + mean
                    # On repasse en 0-255 proprement
                    img_uint8 = (np.clip(img_rescaled, 0, 1) * 255).astype(np.uint8)
                            
                    # === PRÉDICTION YOLO AVEC SEUILS OPTIMISÉS ===
                    results = model.predict(
                        img_uint8,
                        verbose=False,
                        device=self.device,
                        imgsz=TARGET_SIZE[0],
                        conf=0.5,  
                        iou=0.45    
                    )
                    
                    # === CONVERSION INSTANCE → SÉMANTIQUE AVEC MAPPING ===
                    gt_mask = masks[i].cpu().numpy()
                    pred_semantic = self._convert_instance_to_semantic(
                        results[0],
                        gt_mask.shape
                    )
                    
                    # Debug: Compter les détections
                    if results[0].boxes is not None:
                        total_predictions += len(results[0].boxes)
                    
                    bg_pixels = (pred_semantic == 0).sum()
                    non_bg_pixels = (pred_semantic > 0).sum()
                    total_background_pixels += bg_pixels
                    total_non_background_pixels += non_bg_pixels
                    
                    # Debug première image
                    if batch_idx == 0 and i == 0:
                        print(f"\n[DEBUG] Première prédiction:")
                        print(f"  Image shape: {img_uint8.shape}")
                        print(f"  GT mask shape: {gt_mask.shape}")
                        print(f"  GT classes: {np.unique(gt_mask)}")
                        print(f"  Pred classes: {np.unique(pred_semantic)}")
                        if results[0].boxes is not None:
                            print(f"  Nb détections YOLO: {len(results[0].boxes)}")
                            print(f"  Classes YOLO (0-4): {results[0].boxes.cls.cpu().numpy()[:5]}")
                            print(f"  Confidences: {results[0].boxes.conf.cpu().numpy()[:5]}")
                        print(f"  Pixels Background: {bg_pixels} ({bg_pixels/(gt_mask.size)*100:.1f}%)")
                        print(f"  Pixels détectés: {non_bg_pixels} ({non_bg_pixels/(gt_mask.size)*100:.1f}%)")
                    
                    # Mise à jour métriques
                    self.val_metrics.update(pred_semantic, gt_mask)
        
        # 4. Statistiques globales
        total_pixels = total_background_pixels + total_non_background_pixels
        print(f"\n📊 Statistiques globales:")
        print(f"   Total détections YOLO: {total_predictions}")
        print(f"   Pixels Background: {total_background_pixels} ({total_background_pixels/total_pixels*100:.1f}%)")
        print(f"   Pixels détectés: {total_non_background_pixels} ({total_non_background_pixels/total_pixels*100:.1f}%)")
        
        # 5. Résultats finaux
        final_results = self.val_metrics.get_results()
        
        print(f"\n✅ mIoU Final (excl. Background) : {final_results['mIoU']:.4f}")
        print(f"✅ mAcc Final : {final_results['mAcc']:.4f}")
        
        print("\n📋 IoU par classe:")
        for i, iou in enumerate(final_results['class_iou']):
            print(f"  {CLASS_NAMES_SEG[i]:15s}: {iou:.4f}")
        
        # 6. Matrice de confusion (optionnel mais utile)
        print("\n📊 Matrice de Confusion (premières lignes):")
        conf_matrix = self.val_metrics.confusion_matrix
        for i in range(min(6, self.num_classes_eval)):
            true_class = CLASS_NAMES_SEG[i]
            total = conf_matrix[i].sum()
            if total > 0:
                print(f"\n{true_class} (GT) - Total pixels: {total:.0f}")
                for j in range(min(6, self.num_classes_eval)):
                    pred_class = CLASS_NAMES_SEG[j]
                    count = conf_matrix[i, j]
                    percentage = (count / total * 100) if total > 0 else 0
                    if percentage > 5:  # Afficher si > 5%
                        print(f"  → {pred_class:15s}: {percentage:5.1f}%")
        
        # 7. Sauvegarde JSON
        results_path = OUTPUT_DIR_YOLO / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'mIoU': float(final_results['mIoU']),
                'mAcc': float(final_results['mAcc']),
                'class_iou': [float(x) for x in final_results['class_iou']],
                'class_names': [CLASS_NAMES_SEG[i] for i in range(self.num_classes_eval)],
                'conf_threshold': 0.15,
                'iou_threshold': 0.4,
                'total_detections': int(total_predictions),
                'background_ratio': float(total_background_pixels / total_pixels)
            }, f, indent=2)
            
        print(f"\n💾 Résultats validés sauvegardés dans : {results_path}")

    def _convert_instance_to_semantic(self, result, img_shape):
        H, W = img_shape
        semantic_mask = np.zeros((H, W), dtype=np.int64)
        
        if result.masks is None:
            return semantic_mask
        
        # Ordre de priorité : on dessine d'abord les nappes, 
        # puis les objets par-dessus pour qu'ils ne soient pas écrasés.
        # 1: Oil, 2: Emulsion, 3: Sheen, 4: Ship, 5: Oil-platform
        priority_order = [3, 2, 1, 4, 5] 
        
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        # On trie les détections par l'ordre de priorité défini ci-dessus
        detections = sorted(zip(masks, classes, confidences), 
                            key=lambda x: priority_order.index(x[1] + 1) if (x[1]+1) in priority_order else 0)

        for mask, yolo_cls, conf in detections:
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
            mask_binary = mask_resized > 0.6
            
            gt_cls = yolo_cls + 1 # Votre mapping actuel
            
            # On remplace systématiquement si c'est une classe prioritaire
            semantic_mask[mask_binary] = gt_cls
            
        return semantic_mask
    

if __name__ == '__main__':
    trainer = YOLOv11Trainer()
    trainer.train()
    