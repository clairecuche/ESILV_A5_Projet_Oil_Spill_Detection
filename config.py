from typing import Dict, Tuple
from pathlib import Path
import torch

# --- Constantes Générales du Projet ---
CLASS_TO_ID: Dict[str, int] = {
    'background': 0,
    'oil': 1,
    'emulsion': 2,
    'sheen': 3,
    'ship': 4,
    'oil-platform': 5
}

CLASS_NAMES = {
    0: 'Background',
    1: 'Oil',
    2: 'Emulsion',
    3: 'Sheen',
    4: 'Ship',
    5: 'Oil-platform'
}
NUM_CLASSES = len(CLASS_TO_ID)
TARGET_SIZE: Tuple[int, int] = (256, 256) #Reduit pour accélérer les tests
DATA_DIR = Path('./LADOS-2')
NUM_CLASSES = 6

# --- Paramètres d'Entraînement SegFormer (comme le paper pour l'instant)---
OUTPUT_DIR_SEG = Path('./outputs/segformer')
CHECKPOINT_DIR_SEG = OUTPUT_DIR_SEG / 'checkpoints'
BATCH_SIZE = 8 #Pas possible d'augmenter plus sur la machine actuelle
NUM_EPOCHS = 1
LEARNING_RATE = 6e-5 
WEIGHT_DECAY = 0.01
PATIENCE = 10  
MODEL_NAME_SEG = 'nvidia/segformer-b3-finetuned-cityscapes-1024-1024'  


# --- Paramètres d'Entraînement Yolo (comme le paper pour l'instant)---
OUTPUT_DIR_YOLO = Path('./outputs/yolo')
CHECKPOINT_DIR_YOLO = OUTPUT_DIR_YOLO / 'checkpoints'
BATCH_SIZE_YOLO = 32 #Pas possible d'augmenter plus sur la machine actuelle
NUM_EPOCHS_YOLO = 80
LEARNING_RATE_YOLO = 6e-5 
WEIGHT_DECAY = 0.01
PATIENCE = 10  
MODEL_NAME_SEG = 'nvidia/segformer-b3-finetuned-cityscapes-1024-1024'  

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 8
PIN_MEMORY = True 
PREFETCH_FACTOR = 2