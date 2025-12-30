from typing import Dict, Tuple
from pathlib import Path
import torch

# --- Constantes Générales du Projet ---
# Liste ordonnée des classes d'objets (SANS le background pour YOLO)
# Cet ordre doit être identique dans votre fichier data.yaml de YOLO
YOLO_CLASSES: list[str] = [
    'oil',          # ID 0 pour YOLO
    'emulsion',     # ID 1 pour YOLO
    'sheen',        # ID 2 pour YOLO
    'ship',         # ID 3 pour YOLO
    'oil-platform'  # ID 4 pour YOLO
]

# Mapping pour le convertisseur COCO -> YOLO
# Utilise les noms de ton JSON Roboflow comme clés
CLASS_TO_ID_YOLO: Dict[str, int] = {name: i for i, name in enumerate(YOLO_CLASSES)}
NUM_CLASSES =6

# Configuration pour SegFormer (Sémantique : nécessite le Background en ID 0)
CLASS_NAMES_SEG = ['background'] + YOLO_CLASSES
NUM_CLASSES_SEG = len(CLASS_NAMES_SEG)  # 6 classes

# Configuration pour YOLO (Instance : les classes commencent à 0)
NUM_CLASSES_YOLO = len(YOLO_CLASSES)    # 5 classes

TARGET_SIZE: Tuple[int, int] = (640, 640)
DATA_DIR = Path('./LADOS-2')

# --- Paramètres d'Entraînement SegFormer (comme le paper pour l'instant)---
OUTPUT_DIR_SEG = Path('./outputs/segformer')
CHECKPOINT_DIR_SEG = OUTPUT_DIR_SEG / 'checkpoints'
BATCH_SIZE = 8 #Pas possible d'augmenter plus sur la machine actuelle
NUM_EPOCHS = 10
LEARNING_RATE = 6e-5 
WEIGHT_DECAY = 0.01
PATIENCE = 10  
MODEL_NAME_SEG = 'nvidia/segformer-b3-finetuned-cityscapes-1024-1024'  


# --- Paramètres d'Entraînement Yolo (comme le paper pour l'instant)---
OUTPUT_DIR_YOLO = Path('./outputs/yolo')
CHECKPOINT_DIR_YOLO = OUTPUT_DIR_YOLO / 'checkpoints'
BATCH_SIZE_YOLO = 8
NUM_EPOCHS_YOLO = 80
LEARNING_RATE_YOLO = 6e-5 
WEIGHT_DECAY = 0.01
PATIENCE = 10  
MODEL_NAME_YOLO = 'yolo11m-seg.pt'  

# --- Hardware ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS =8
PIN_MEMORY = True 
PREFETCH_FACTOR = 2