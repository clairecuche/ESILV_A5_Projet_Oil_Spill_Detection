"""
Script principal pour l'entraînement YOLOv11 sur LADOS.
Suit exactement les paramètres du paper LADOS (Section 4).
"""

import sys
from pathlib import Path


from src.data.yolo_data_converter import convert_coco_to_yolo_segmentation, verify_conversion
from src.models.train_yolo import YOLOv11Trainer
from config import DATA_DIR


def main():
    """
    Pipeline complet d'entraînement YOLOv11:
    1. Conversion des données COCO → YOLO
    2. Vérification de la conversion
    3. Entraînement du modèle
    4. Évaluation finale
    """
    
    print("="*70)
    print("🚀 PIPELINE D'ENTRAÎNEMENT YOLOv11 - DATASET LADOS")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # ÉTAPE 1: Conversion des données
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("📦 ÉTAPE 1/4: CONVERSION DES DONNÉES (COCO → YOLO)")
    print("="*70)
    
    yolo_data_path = DATA_DIR / 'yolo_format'
    
    if not yolo_data_path.exists() or not (yolo_data_path / 'data.yaml').exists():
        print("🔄 Conversion du format COCO vers YOLO...")
        yolo_data_path = convert_coco_to_yolo_segmentation(DATA_DIR)
    else:
        print(f"✅ Données YOLO déjà existantes: {yolo_data_path}")
    
    # -------------------------------------------------------------------------
    # ÉTAPE 2: Vérification
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("🔍 ÉTAPE 2/4: VÉRIFICATION DE LA CONVERSION")
    print("="*70)
    
    verify_conversion(yolo_data_path, num_samples=5)
    
    # -------------------------------------------------------------------------
    # ÉTAPE 3: Entraînement
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("🏋️ ÉTAPE 3/4: ENTRAÎNEMENT DU MODÈLE YOLOv11")
    print("="*70)
    print("\n📋 Configuration d'entraînement (selon paper LADOS):")
    print("   - Modèle: YOLOv11m-seg (pré-entraîné sur COCO)")
    print("   - Batch size: 32")
    print("   - Max epochs: 80 (avec early stopping)")
    print("   - Patience: 10 epochs")
    print("   - Optimizer: AdamW")
    print("   - Augmentations: Mosaic, HSV, Flip, Translate, Scale")
    print("   - Early stopping metric: weighted mAP (0.1*mAP@50 + 0.9*mAP@50-95)")
    print("")
    
    # Initialiser et lancer l'entraînement
    trainer = YOLOv11Trainer()
    results = trainer.train()
    
    # -------------------------------------------------------------------------
    # ÉTAPE 4: Résumé
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("✅ ÉTAPE 4/4: RÉSUMÉ FINAL")
    print("="*70)
    
    print("\n📊 Résultats d'entraînement YOLOv11:")
    print(f"   - Meilleur mIoU (validation): {trainer.best_miou:.4f}")
    print(f"   - Meilleur mAP@50-95: {trainer.best_map:.4f}")
    print(f"\n💾 Modèles sauvegardés dans: {trainer.checkpoint_dir}")
    print(f"📈 Résultats détaillés dans: {trainer.checkpoint_dir.parent / 'test_results.json'}")
    
    print("\n" + "="*70)
    print("🎉 ENTRAÎNEMENT YOLOV11 TERMINÉ AVEC SUCCÈS!")
    print("="*70)
    
    return trainer, results


if __name__ == '__main__':
    trainer, results = main()