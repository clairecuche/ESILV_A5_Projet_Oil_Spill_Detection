# src/data/lados_dataset.py
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Dict, Optional
from config import TARGET_SIZE, DATA_DIR, BATCH_SIZE, NUM_CLASSES, NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR

# --- Fonctions d'Augmentation (Robustes et Stables) ---

def get_training_augmentation(target_size: Tuple[int, int] = TARGET_SIZE) -> A.Compose:
    """
    Pipeline d'augmentation pour l'entraînement (géométrique, chromatique, normalisation).
    """
    height, width = target_size[0], target_size[1]
    
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5), 
        
        # A.Affine remplace ShiftScaleRotate/Rotate
        A.Affine(
            scale=(0.8, 1.2), 
            translate_percent=(-0.1, 0.1), 
            rotate=(-15, 15), 
            shear=(-5, 5), 
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        
        # Recadrage aléatoire
        A.RandomCrop(height=int(height * 0.9), width=int(width * 0.9), p=0.5),
        
        # Redimensionnement final pour fixer la taille (crucial après RandomCrop)
        A.Resize(height=height, width=width), 
        
        # Transformations de couleur et bruit
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(p=0.3), 
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # Normalisation et conversion en Tensor
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ]
    # Assurer que le masque utilise une interpolation 'Nearest' pour les pixels discrets
    )  


def get_validation_augmentation(target_size: Tuple[int, int] = TARGET_SIZE) -> A.Compose:
    """
    Pipeline pour validation/test (seulement resize et normalisation).
    """
    height, width = target_size[0], target_size[1]
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def visualize_augmentation_example(dataset_path: str, index: int = 0, n_versions: int = 3) -> None:
    """
    Charge, augmente et affiche l'image originale et N versions augmentées 
    pour une vérification rapide.

    Args:
        dataset_path: Chemin racine des données (ex: 'data/LADOS-2').
        index: Index de l'image 'train' à visualiser.
        n_versions: Nombre de versions augmentées à afficher.
    """
    try:
        # Création du Dataset avec les augmentations d'entraînement
        # LADOSDataset doit être défini dans ce fichier !
        train_transform = get_training_augmentation()
        train_dataset = LADOSDataset(dataset_path, 'train', train_transform)
    except Exception as e:
        print(f"Erreur lors de l'initialisation du Dataset. Avez-vous exécuté la conversion COCO->Masques ? {e}")
        return

    if index >= len(train_dataset):
        print(f"Index {index} hors limites. Taille du dataset: {len(train_dataset)}")
        return
    
    H, W = TARGET_SIZE
    
    # 1. Charger l'image originale (pour la ligne de référence)
    image_path = train_dataset.images_list[index]
    mask_path = train_dataset.masks_dir / (image_path.stem + '.png')

    image_original = cv2.imread(str(image_path))
    image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)
    
    # Redimensionnement de l'original pour l'affichage à la taille cible
    image_original_resized = cv2.resize(image_original, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_original_resized = cv2.resize(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE), 
                                       (W, H), interpolation=cv2.INTER_NEAREST)

    
    # Dénormalisation (pour l'affichage)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 2. Configuration de la figure
    fig, axes = plt.subplots(n_versions + 1, 2, figsize=(10, 5 * (n_versions + 1)))
    
    # S'assurer que 'axes' est bien une grille 2D même pour n_versions=0
    if n_versions == 0:
        axes = np.array([[axes[0], axes[1]]])
    
    # Titres pour les colonnes
    axes[0, 0].set_title("Image Originale Redimensionnée", fontsize=12)
    axes[0, 1].set_title("Masque Original Redimensionné", fontsize=12)
    
    # LIGNE 0 : Original
    axes[0, 0].imshow(image_original_resized)
    axes[0, 0].axis('off')
    
    # Utilisation d'une cmap pour distinguer les classes 0 à 5
    axes[0, 1].imshow(mask_original_resized, cmap='tab10', vmin=0, vmax=5)
    axes[0, 1].axis('off')

    # 3. Afficher les N versions augmentées
    for i in range(n_versions):
        # Appel de __getitem__ : le point clé qui applique les transformations !
        image_tensor, mask_tensor = train_dataset[index] 

        # Dénormaliser l'image tensor (de [C, H, W] à [H, W, C])
        aug_image_vis = image_tensor.permute(1, 2, 0).numpy()
        aug_image_vis = aug_image_vis * std + mean
        aug_image_vis = np.clip(aug_image_vis, 0, 1)

        # Afficher l'image augmentée
        axes[i + 1, 0].imshow(aug_image_vis)
        axes[i + 1, 0].set_title(f'Augmentation #{i + 1} (Image)', fontsize=12)
        axes[i + 1, 0].axis('off')

        # Afficher le masque augmenté
        axes[i + 1, 1].imshow(mask_tensor.numpy(), cmap='tab10', vmin=0, vmax=5)
        axes[i + 1, 1].set_title(f'Augmentation #{i + 1} (Masque)', fontsize=12)
        axes[i + 1, 1].axis('off')

    plt.suptitle(f"Visualisation des Augmentations (Image: {image_path.name})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()

# --- Classe PyTorch Dataset ---

class LADOSDataset(Dataset):
    """
    Dataset personnalisé pour le jeu de données LADOS (Semantic Segmentation).
    """
    def __init__(self, dataset_path: str, split: str, transform: A.Compose):
        self.split = split
        self.transform = transform
        
        self.image_dir = Path(dataset_path) / split
        self.masks_dir = Path(dataset_path) / split / 'masks'
        
        # Liste des images (sans les annotations JSON)
        self.images_list = sorted([p for p in self.image_dir.glob('*.jpg') if not p.name.startswith('_')])
        
        if not self.images_list:
             raise FileNotFoundError(f"Aucune image trouvée dans {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images_list[idx]
        mask_path = self.masks_dir / (image_path.stem + '.png')
        
        # 1. Chargement des données
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        # S'assurer que les masques sont des tableaux 2D (H, W)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Si les dimensions diffèrent, redimensionner le masque pour correspondre à l'image
        h_img, w_img = image.shape[:2]
        h_mask, w_mask = mask.shape[:2]
        if (h_img, w_img) != (h_mask, w_mask):
            mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)

        # 2. Application de la transformation/augmentation
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].long() # Masque en Long Tensor pour CrossEntropyLoss
        
        # La normalisation dans l'augmentation place le tenseur dans l'intervalle [0, 1] et l'inverse l'ordre (C, H, W)
        
        return image, mask


def get_dataloaders(dataset_path: str = DATA_DIR, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS) -> Dict[str, DataLoader]:
    """
    Crée et retourne les DataLoaders pour l'entraînement et la validation.
    """
    
    train_transform = get_training_augmentation()
    val_test_transform = get_validation_augmentation()

    train_dataset = LADOSDataset(dataset_path, 'train', train_transform)
    val_dataset = LADOSDataset(dataset_path, 'valid', val_test_transform)
    test_dataset = LADOSDataset(dataset_path, 'test', val_test_transform)

    # 5. OPTIMISATION : Paramètres DataLoader
    common_params = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'prefetch_factor': PREFETCH_FACTOR,
        'persistent_workers': True if num_workers > 0 else False # AJOUT
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common_params)
    val_loader = DataLoader(val_dataset, shuffle= False, **common_params)
    test_loader = DataLoader(test_dataset,shuffle=False, **common_params)

    return {
        'train': train_loader,
        'valid': val_loader,
        'test': test_loader
    }