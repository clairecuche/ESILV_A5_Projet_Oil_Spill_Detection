# main.py

import os
from src.data.preprocessing import convert_coco_to_masks, calculate_class_weights, TARGET_SIZE
from src.data.data_loaders import get_dataloaders, get_training_augmentation, visualize_augmentation_example
import torch

DATASET_PATH = 'LADOS-2'
BATCH_SIZE = 8
NUM_WORKERS = 4
# ------------------------------------

def run_data_pipeline():
    """
    Exécute le pipeline complet de data loading et preprocessing.
    """
    print("--- 1. ÉTAPE DE PRÉ-TRAITEMENT (Conversion et Poids) ---")
    
    # 1. Conversion des annotations COCO en masques PNG
    print(f"Lancement de la conversion COCO -> Masques pour {DATASET_PATH}")
    convert_coco_to_masks(DATASET_PATH, target_size=TARGET_SIZE)

    # 2. Calcul des poids de classe pour la Loss Function
    class_weights_tensor = calculate_class_weights(DATASET_PATH, split='train')
    
    # Sauvegarde des poids pour une utilisation ultérieure
    torch.save(class_weights_tensor, 'class_weights.pt')
    print(f"✓ Poids de classe sauvegardés dans class_weights.pt : {class_weights_tensor}")

    print("\n--- 2. ÉTAPE DE CHARGEMENT DES DONNÉES (DataLoaders) ---")
    
    # 3. Création des DataLoaders PyTorch
    dataloaders = get_dataloaders(
        DATASET_PATH, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )

    print(f"✓ DataLoaders créés :")
    print(f"  - Train: {len(dataloaders['train'].dataset)} images (Batch size: {BATCH_SIZE})")
    print(f"  - Valid: {len(dataloaders['valid'].dataset)} images")
    print(f"  - Test: {len(dataloaders['test'].dataset)} images")


    visualize_augmentation_example(
        dataset_path=DATASET_PATH, 
        index=75, 
        n_versions=3
    )


    visualize_augmentation_example(
        dataset_path=DATASET_PATH, 
        index=125, 
        n_versions=3
    )
    # --- Exemple d'utilisation du premier batch ---
    if 'train' in dataloaders:
        sample_batch = next(iter(dataloaders['train']))
        images, masks = sample_batch
        print(f"\nExemple de batch d'entraînement :")
        print(f"  - Images Tensor Shape: {images.shape}") # [B, C, H, W]
        print(f"  - Masks Tensor Shape: {masks.shape}")   # [B, H, W]

    print("\nLe pipeline de données est prêt pour l'entraînement du modèle.")
    return dataloaders, class_weights_tensor

if __name__ == '__main__':
    # Cette fonction lance tout
    run_data_pipeline()