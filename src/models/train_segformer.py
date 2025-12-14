import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt

from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig
)
from src.data.data_loaders import get_dataloaders
from src.data.preprocessing import calculate_class_weights
from config import *



class SegFormerTrainer:
    def __init__(self):
        self.device = torch.device(DEVICE)
        self.num_classes = NUM_CLASSES
        
        # 1. Modèle
        self.model = self._initialize_model().to(self.device)
        
        # 2. Loss Function (avec Poids de classe)
        weights_path = DATA_DIR / 'class_weights.pt'
        if weights_path.exists():
            class_weights = torch.load(weights_path)
        else:
            # Assurez-vous que calculate_class_weights peut être appelée ici
            class_weights = calculate_class_weights()
            torch.save(class_weights, weights_path)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        # 3. Optimiseur
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                            lr=LEARNING_RATE, 
                                            weight_decay=WEIGHT_DECAY)
        
        # 4. DataLoaders (Récupère train, valid, test)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders()
        
        # 5. Métriques et Suivi (État)
        self.train_metrics = SegmentationMetrics(NUM_CLASSES)
        self.val_metrics = SegmentationMetrics(NUM_CLASSES)
        self.best_miou = -1.0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'train_miou': [], 'val_miou': []}
        self.checkpoint_dir = CHECKPOINT_DIR_SEG
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Trainer initialisé sur {self.device}. Poids de classe appliqués.")


    def _initialize_model(self):
        """Initialise le modèle SegFormer."""
        config_hf = SegformerConfig.from_pretrained(MODEL_NAME_SEG, num_labels=NUM_CLASSES)
        model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME_SEG,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True
        )
        return model
    def _run_one_epoch(self, dataloader, metrics: 'SegmentationMetrics', is_training: bool, epoch: int):
        """Logique générique pour une epoch (entraînement ou validation)."""
        if is_training:
            self.model.train()
            pbar_desc = f"Epoch {epoch+1}/{NUM_EPOCHS} [TRAIN]"
        else:
            self.model.eval()
            pbar_desc = "Validating"
        
        metrics.reset()
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=pbar_desc)
        
        context_manager = torch.enable_grad() if is_training else torch.no_grad()
        
        with context_manager:
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                if is_training:
                    self.optimizer.zero_grad()
                
                # Forward pass SegFormer
                # Pour l'entraînement (si is_training=True), on passe les labels pour obtenir outputs.loss
                outputs = self.model(pixel_values=images, labels=masks)
                logits = outputs.logits
                
                # 💡 Logique d'Upsampling du Code 2
                logits_upsampled = nn.functional.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                
                # Calcul de la perte avec la weighted loss
                loss = self.criterion(logits_upsampled, masks)
                
                if is_training:
                    loss.backward()
                    self.optimizer.step()
                
                running_loss += loss.item()
                
                # Calcul des prédictions pour les métriques
                preds = logits_upsampled.argmax(dim=1)
                for pred, target in zip(preds, masks):
                    metrics.update(pred, target)
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 
                                  'avg_loss': f"{running_loss / (batch_idx + 1):.4f}"})

        # Retourner les résultats de l'epoch
        results = metrics.get_results()
        return {'loss': running_loss / len(dataloader), 
                'mIoU': results['mIoU'], 
                'mAcc': results['mAcc'],
                'class_iou': results['class_iou']}

    # ------------------------------------------------------------------------
    
    def train(self):
        """Boucle principale d'entraînement (Intègre Early Stopping et Sauvegarde)."""
        print("\n🚀 Début de l'entraînement...")
        start_time = datetime.now()
        
        for epoch in range(NUM_EPOCHS):
            # 1. Entraînement
            train_results = self._run_one_epoch(self.train_loader, self.train_metrics, is_training=True, epoch=epoch)
            
            # 2. Validation
            val_results = self._run_one_epoch(self.valid_loader, self.val_metrics, is_training=False, epoch=epoch)
            
            # Mise à jour de l'historique
            self.history['train_loss'].append(train_results['loss'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['train_miou'].append(train_results['mIoU'])
            self.history['val_miou'].append(val_results['mIoU'])
            
            # Affichage
            print(f"\n📊 Epoch {epoch+1} | Train Loss: {train_results['loss']:.4f} | Val Loss: {val_results['loss']:.4f} | Val mIoU: {val_results['mIoU']:.4f}")
            
            # 3. Early Stopping & Sauvegarde
            if val_results['mIoU'] > self.best_miou:
                self.best_miou = val_results['mIoU']
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_results)
            else:
                self.patience_counter += 1
                print(f"  ⏳ Patience: {self.patience_counter}/{PATIENCE}")
                
                if self.patience_counter >= PATIENCE:
                    print(f"\n🛑 Early stopping déclenché après {epoch+1} epochs.")
                    break
        
        # 4. Évaluation Finale
        training_time = (datetime.now() - start_time).total_seconds() / 3600
        print(f"\n--- ENTRAÎNEMENT TERMINÉ ---")
        print(f"Temps total : {training_time:.2f} heures.")
        self._evaluate_test_set()
        # Ajoutez l'appel à plot_training_curves(self.history, self.config.OUTPUT_DIR) ici.

    def _save_checkpoint(self, epoch, val_results):
        """Sauvegarde l'état complet du modèle."""
        checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': val_results['mIoU'],
            'history': self.history
        }, checkpoint_path)
        print(f"  ✅ Nouveau meilleur modèle sauvegardé (mIoU: {val_results['mIoU']:.4f})")

    def _evaluate_test_set(self):
        """Charge le meilleur modèle et évalue sur le set de test."""
        print("\n" + "="*50)
        print("📊 ÉVALUATION FINALE SUR TEST SET")
        print("="*50)

        # 1. Charger le meilleur modèle
        checkpoint_path = self.checkpoint_dir / 'best_model.pt'
        if not checkpoint_path.exists():
            print("Impossible d'évaluer : aucun meilleur modèle trouvé.")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 2. Évaluer (en utilisant la même fonction d'exécution d'epoch, mais sur le test set)
        test_metrics = SegmentationMetrics(NUM_CLASSES)
        test_results = self._run_one_epoch(self.test_loader, test_metrics, is_training=False, epoch=-1)

        print(f"\nRésultats finaux (mIoU) : {test_results['mIoU']:.4f}")
        print("\nIoU par classe :")
        for class_id, iou in enumerate(test_results['class_iou']):
            class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
            print(f"  {class_name:15s}: {iou:.4f}")

class SegmentationMetrics:
    """
    Calcule mIoU et mAcc selon les formules du paper (Section 4.4)
    
    """
    
    def __init__(self, ignore_index=None):
        self.num_classes = NUM_CLASSES
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
        iou = np.divide(TP, denominator, out=np.zeros_like(TP, dtype=float), where=denominator != 0)
        
        valid_classes = iou[1:] # Exclure background (classe 0) pour mIoU selon le paper
        miou = np.nanmean(valid_classes)
        
        return {'class_iou': iou, 'mIoU': miou, 'mIoU_with_bg': np.nanmean(iou)}
    
    def compute_accuracy(self):
        """Calcule mAcc"""
        TP = np.diag(self.confusion_matrix)
        FN = self.confusion_matrix.sum(axis=1) - TP
        
        denominator = TP + FN
        accuracy = np.divide(TP, denominator, out=np.zeros_like(TP, dtype=float), where=denominator != 0)
        
        macc = np.nanmean(accuracy)
        
        return {'class_acc': accuracy, 'mAcc': macc}

    def get_results(self):
        """Retourne tous les résultats"""
        return {**self.compute_iou(), **self.compute_accuracy()}
    

if __name__ == '__main__':
    trainer = SegFormerTrainer()
    trainer.train()