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

from src.data.data_loaders import get_dataloaders
from src.data.preprocessing import calculate_class_weights
from src.data.yolo_data_converter import convert_coco_to_yolo_segmentation
from config import *


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
        Évalue le modèle sur le test set avec les métriques du paper:
        - IoU par classe
        - mIoU (excluant background)
        - mAcc
        
        Note: YOLO prédit des masques d'instance, nous devons les convertir en 
        masques sémantiques en assignant chaque pixel à la classe avec le plus 
        haut score de confiance (comme mentionné dans le paper section 4.4).
        """
        print("\n" + "="*50)
        print("📊 ÉVALUATION FINALE SUR TEST SET")
        print("="*50)
        
        # Charger le meilleur modèle
        best_model_path = OUTPUT_DIR_YOLO / 'yolo_training' / 'weights' / 'best.pt'
        if not best_model_path.exists():
            print("⚠️ Meilleur modèle non trouvé, utilisation du modèle actuel.")
            model = self.model
        else:
            model = YOLO(str(best_model_path))
        
        # Récupérer le test loader
        loaders = get_dataloaders()
        test_loader = loaders['test']
        
        # Métriques
        test_metrics = SegmentationMetrics(NUM_CLASSES)
        
        model.model.eval()
        print("\n🔍 Prédiction sur le test set...")
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                images, masks = batch
                
                # Conversion pour YOLO (numpy format)
                for i in range(images.shape[0]):
                    img = images[i].cpu().numpy().transpose(1, 2, 0)
                    img = (img * 255).astype(np.uint8)
                    gt_mask = masks[i].cpu().numpy()
                    
                    # Prédiction YOLO
                    results = model.predict(img, verbose=False, device=self.device)
                    
                    # Conversion instance -> semantic mask
                    # (selon paper section 4.4: pixel assigné à la classe avec highest confidence)
                    pred_mask = self._convert_instance_to_semantic(
                        results[0], 
                        img.shape[:2]
                    )
                    
                    # Mise à jour des métriques
                    test_metrics.update(pred_mask, gt_mask)
        
        # Calcul des résultats finaux
        results = test_metrics.get_results()
        
        print(f"\n✅ Résultats finaux (mIoU): {results['mIoU']:.4f}")
        print(f"   mAcc: {results['mAcc']:.4f}")
        print("\n📋 IoU par classe:")
        for class_id, iou in enumerate(results['class_iou']):
            class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
            print(f"  {class_name:15s}: {iou:.4f}")
        
        # Sauvegarder les résultats
        results_path = OUTPUT_DIR_YOLO / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'mIoU': float(results['mIoU']),
                'mAcc': float(results['mAcc']),
                'class_iou': [float(x) for x in results['class_iou']],
                'class_names': [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
            }, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés dans: {results_path}")
    
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


class SegmentationMetrics:
    """
    Calcule mIoU et mAcc selon les formules du paper (Section 4.4).
    Identique à l'implémentation SegFormer.
    """
    
    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Réinitialise les compteurs"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, pred, target):
        """Met à jour la matrice de confusion"""
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        
        pred = pred.flatten()
        target = target.flatten()
        
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            pred = pred[mask]
            target = target[mask]
        
        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[int(t), int(p)] += 1
    
    def compute_iou(self):
        """Calcule IoU par classe et mIoU (excluant le background, classe 0)"""
        TP = np.diag(self.confusion_matrix)
        FP = self.confusion_matrix.sum(axis=0) - TP
        FN = self.confusion_matrix.sum(axis=1) - TP
        
        denominator = TP + FP + FN
        iou = np.divide(TP, denominator, out=np.zeros_like(TP, dtype=float), 
                       where=denominator != 0)
        
        # Exclure background (classe 0) pour mIoU selon le paper
        valid_classes = iou[1:]
        miou = np.nanmean(valid_classes)
        
        return {'class_iou': iou, 'mIoU': miou, 'mIoU_with_bg': np.nanmean(iou)}
    
    def compute_accuracy(self):
        """Calcule mAcc"""
        TP = np.diag(self.confusion_matrix)
        FN = self.confusion_matrix.sum(axis=1) - TP
        
        denominator = TP + FN
        accuracy = np.divide(TP, denominator, out=np.zeros_like(TP, dtype=float), 
                           where=denominator != 0)
        
        macc = np.nanmean(accuracy)
        
        return {'class_acc': accuracy, 'mAcc': macc}

    def get_results(self):
        """Retourne tous les résultats"""
        return {**self.compute_iou(), **self.compute_accuracy()}


if __name__ == '__main__':
    trainer = YOLOv11Trainer()
    trainer.train()