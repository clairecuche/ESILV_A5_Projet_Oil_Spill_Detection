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
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig
)
from src.data.data_loaders import get_dataloaders
from src.data.preprocessing import calculate_class_weights
from config import *
from metrics import SegmentationMetrics



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

        self.scaler = GradScaler("cuda") # Pour AMP
        
        # 4. DataLoaders (Récupère train, valid, test)
        loaders = get_dataloaders()
        self.train_loader = loaders['train']
        self.valid_loader = loaders['valid']
        self.test_loader = loaders['test']
        
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
        
        # 6. OPTIMISATION : Compilation du modèle PyTorch 2.0+
        if self.device.type == 'cuda' and torch.__version__ >= '2.0':
            print("🚀 Compilation du modèle SegFormer (PyTorch 2.0+)...")
            # Utiliser 'reduce-overhead' ou 'compile' pour l'entraînement
            model = torch.compile(model, mode='reduce-overhead') 

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
        
        # Contexte de PyTorch pour l'entraînement (enable_grad) ou la validation (no_grad)
        context_manager_grad = torch.enable_grad() if is_training else torch.no_grad()
        
        with context_manager_grad: 
            for batch_idx, batch in enumerate(pbar):
                # Le batch est un tuple (images, masks)
                images, masks = batch 
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                if is_training:
                    self.optimizer.zero_grad()
                
                # 4. OPTIMISATION : Début du contexte Mixed Precision (autocast)
                # Active autocast uniquement si le GPU est utilisé
                with autocast("cuda", enabled=self.device.type == "cuda"):
                    
                    # Forward pass SegFormer
                    # On n'utilise pas 'labels' ici, car nous calculons la perte manuellement après upsampling
                    outputs = self.model(pixel_values=images)
                    logits = outputs.logits 
                    
                    # 💡 Logique d'Upsampling vers la taille du masque (TARGET_SIZE)
                    logits_upsampled = F.interpolate(
                        logits,
                        size=masks.shape[-2:], # Utilise H, W du masque cible (640 ou 512)
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    # Calcul de la perte avec la weighted loss (la perte est mise à l'échelle si en mode AMP)
                    loss = self.criterion(logits_upsampled, masks)
                
                if is_training:
                    # 4. OPTIMISATION : Backward pass et optimisation avec GradScaler
                    # Mise à l'échelle de la perte
                    self.scaler.scale(loss).backward() 
                    
                    # Met à jour les poids (si les gradients ne sont pas trop grands)
                    self.scaler.step(self.optimizer) 
                    
                    # Met à jour le facteur d'échelle pour le prochain tour
                    self.scaler.update() 
                
                # Le .item() est utilisé pour les statistiques (même si la perte est mise à l'échelle)
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
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 2. Évaluer (en utilisant la même fonction d'exécution d'epoch, mais sur le test set)
        test_metrics = SegmentationMetrics(NUM_CLASSES)
        test_results = self._run_one_epoch(self.test_loader, test_metrics, is_training=False, epoch=-1)

        print(f"\nRésultats finaux (mIoU) : {test_results['mIoU']:.4f}")
        print("\nIoU par classe :")
        for class_id, iou in enumerate(test_results['class_iou']):
            class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
            print(f"  {class_name:15s}: {iou:.4f}")

if __name__ == '__main__':
    trainer = SegFormerTrainer()
    trainer.train()