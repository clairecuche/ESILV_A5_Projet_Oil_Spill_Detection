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
        self.num_classes = NUM_CLASSES
        
        # 1. Modèle YOLOv11 (pré-entraîné sur COCO)
        self.model = self._initialize_model()
        
        # 2. Poids de classe pour weighted BCE loss
        weights_path = DATA_DIR / 'class_weights.pt'
        if weights_path.exists():
            self.class_weights = torch.load(weights_path)
        else:
            self.class_weights = calculate_class_weights()
            torch.save(self.class_weights, weights_path)
        
        # 3. Métriques et Suivi
        self.train_metrics = SegmentationMetrics(NUM_CLASSES)
        self.val_metrics = SegmentationMetrics(NUM_CLASSES)
        self.best_miou = -1.0
        self.best_map = -1.0  # Pour mAP@50-95 (utilisé pour early stopping)
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
        print(f"Poids de classe: {self.class_weights}")

    def _initialize_model(self):
        """Initialise YOLOv11m-seg avec poids COCO."""
        model = YOLO('yolo11m-seg.pt')  # YOLOv11m segmentation
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
            epochs=NUM_EPOCHS_YOLO,  # Max epochs (early stopping activé)
            batch=BATCH_SIZE_YOLO,  # 32 selon paper
            imgsz=TARGET_SIZE[0],
            device=self.device,
            
            # Optimizer (AdamW implicite)
            optimizer='AdamW',
            lr0=LEARNING_RATE_YOLO,
            weight_decay=WEIGHT_DECAY,
            
            # Early stopping (patience=10 selon paper)
            patience=PATIENCE,
            
            # Augmentations (selon paper section 4.3)
            mosaic=1.0,  # Mosaic augmentation
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
        Évalue le modèle avec dénormalisation et conversion sémantique.
        """
        print("\n" + "="*50)
        print("📊 ÉVALUATION FINALE SUR TEST SET (CORRIGÉE)")
        print("="*50)
        
        # 1. Charger le meilleur modèle sauvegardé (Chemin basé sur ta config)
        # Comme OUTPUT_DIR_YOLO = Path('./outputs/yolo') et name = 'yolo_training'
        best_model_path = Path("/kaggle/working/outputs/yolo/yolo_training/weights/best.pt")
        
        if not best_model_path.exists():
            print(f"⚠️ Modèle non trouvé à {best_model_path}")
            # Fallback : on cherche dans le dossier project par défaut d'Ultralytics si besoin
            best_model_path = Path("/kaggle/working/yolo/yolo_training/weights/best.pt")
            
        print(f"🔄 Chargement des poids : {best_model_path}")
        model = YOLO(str(best_model_path))

        # 2. Paramètres de dénormalisation (ImageNet standards)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        test_loader = get_dataloaders()['test']
        self.metrics.reset()
        
        print("\n🔍 Prédiction et conversion sémantique...")
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Testing"):
                for i in range(images.shape[0]):
                    # --- ÉTAPE CRITIQUE : DÉNORMALISATION ---
                    # Convertir Tenseur (C, H, W) -> Numpy (H, W, C)
                    img_np = images[i].permute(1, 2, 0).cpu().numpy()
                    # Inverser la normalisation du DataLoader
                    img_raw = (img_np * std + mean) * 255
                    img_raw = np.clip(img_raw, 0, 255).astype(np.uint8)
                    # YOLO travaille mieux en BGR (OpenCV format)
                    img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
                    
                    # --- PRÉDICTION ---
                    results = model.predict(img_bgr, verbose=False, device=self.device, imgsz=TARGET_SIZE[0])
                    
                    # --- CONVERSION INSTANCE -> SÉMANTIQUE ---
                    # On assigne le pixel à la classe avec le plus haut score de confiance (Paper LADOS Section 4.4)
                    pred_semantic = self._convert_instance_to_semantic(results[0], TARGET_SIZE)
                    
                    # Mise à jour de la matrice de confusion
                    self.metrics.update(pred_semantic, masks[i].cpu().numpy())
        
        # 3. Récupération et affichage des résultats
        final_results = self.metrics.get_results()
        
        print(f"\n✅ mIoU Final (excl. Background) : {final_results['mIoU']:.4f}")
        print(f"✅ mAcc Final : {final_results['mAcc']:.4f}")
        
        # Sauvegarde dans le JSON pour compare_models.py
        results_path = OUTPUT_DIR_YOLO / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'mIoU': float(final_results['mIoU']),
                'mAcc': float(final_results['mAcc']),
                'class_iou': [float(x) for x in final_results['class_iou']],
                'class_names': [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
            }, f, indent=2)
            
        print(f"💾 Résultats validés sauvegardés dans : {results_path}")
    
    
    def _convert_instance_to_semantic(self, result, img_shape):
        """
        Convertit les masques d'instance YOLO en masque sémantique.
        
        Selon le paper (section 4.4):
        "we assigned each predicted pixel the class with the highest confidence score"
        
        Args:
            result: Résultat YOLO (contient masks, boxes, cls, conf)
            img_shape: (H, W) de l'image
            
        Returns:
            semantic_mask: np.ndarray de shape (H, W) avec les class IDs
        """
        H, W = img_shape
        semantic_mask = np.zeros((H, W), dtype=np.int64)
        confidence_map = np.zeros((H, W), dtype=np.float32)
        
        if result.masks is None:
            return semantic_mask
        
        # Récupérer les masques, classes et confidences
        masks = result.masks.data.cpu().numpy()  # (N, H, W)
        classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
        confidences = result.boxes.conf.cpu().numpy()  # (N,)
        
        # Redimensionner les masques à la taille de l'image
        for mask, cls, conf in zip(masks, classes, confidences):
            # Resize mask
            mask_resized = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
            mask_binary = mask_resized > 0.5
            
            # Pour chaque pixel du masque, garder la classe avec la plus haute confiance
            update_mask = (mask_binary) & (conf > confidence_map)
            semantic_mask[update_mask] = cls
            confidence_map[update_mask] = conf
        
        return semantic_mask

if __name__ == '__main__':
    trainer = YOLOv11Trainer()
    trainer.train()