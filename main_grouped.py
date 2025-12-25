# main.py
import torch
from pathlib import Path
from src.data.preprocessing import convert_coco_to_masks, calculate_class_weights, convert_coco_to_yolo_segmentation, verify_conversion
from src.data.data_loaders import get_dataloaders, visualize_augmentation_example
from config import TARGET_SIZE


from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS

def run_universal_pipeline():
    print("--- 1. PRÉ-TRAITEMENT COMMUN (Masques et Poids) ---")
    
    # 1. Conversion pour SegFormer (Masques PNG)
    convert_coco_to_masks(DATA_DIR, target_size=TARGET_SIZE)

    # 2. Conversion pour YOLO (Polygones .txt normalisés)
    yolo_data_path = DATA_DIR / 'yolo_format'
    if not yolo_data_path.exists():
        convert_coco_to_yolo_segmentation(DATA_DIR, yolo_data_path)
    
    # 3. Calcul et sauvegarde des poids de classe communs
    class_weights_tensor = calculate_class_weights(DATA_DIR, split='train')
    torch.save(class_weights_tensor, 'class_weights.pt')
    print(f"✅ Poids de classe sauvegardés dans 'class_weights.pt'")
    
    print("\n--- 2. ÉTAPES DE VÉRIFICATION VISUELLE ---")
    
    # 4. Vérification SegFormer : Visualisation des augmentations
    # Affiche l'image originale et les transformations appliquées (Flip, Bruit, etc.)
    print(f"🔍 Vérification SegFormer : Génération de l'aperçu des augmentations...")
    visualize_augmentation_example(dataset_path=DATA_DIR, index=75, n_versions=3)

    # 5. Vérification YOLO : Alignement des polygones
    # Génère un fichier 'conversion_verification.png' dans le dossier yolo_format
    print(f"🔍 Vérification YOLO : Génération de 'conversion_verification.png'...")
    verify_conversion(yolo_data_path, num_samples=5)
    
    print("\n--- 3. CHARGEMENT DES DATALOADERS ---")
    dataloaders = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    print(f"\n🚀 Pipeline terminé avec succès !")
    print(f"   - SegFormer : Masques PNG prêts dans {DATA_DIR}/*/masks/")
    print(f"   - YOLO : Format .txt prêt dans {yolo_data_path}")
    print(f"   - Poids : Fichier 'class_weights.pt' généré.")
    
    return dataloaders

if __name__ == '__main__':
    run_universal_pipeline()