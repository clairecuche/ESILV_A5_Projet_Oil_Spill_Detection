Setup du projet : 
Ouvre PowerShell en mode administrateur (important).

Tape la commande suivante pour installer Python 3.11 :
"""
py install 3.11-64
"""

Le -64 force la version 64-bit.

Si ça demande confirmation, tape y.

Une fois l’installation terminée, vérifie :
"""
py -0
py -3.11 --version
"""

Tu devrais voir Python 3.11 installé et listé.

Créer un venv pour ton projet

Dans ton dossier projet :
"""
py -3.11 -m venv .venv
"""

Active-le :

PowerShell :
"""
.\.venv\Scripts\Activate.ps1
"""

Git Bash :
"""
source .venv/Scripts/activate
"""

Dans l’environnement activé :
"""
pip install jupyter ipykernel
"""

Ajoute un kernel Jupyter nommé python311 :
"""
python -m ipykernel install --user --name python311 --display-name "Python 3.11"
"""
✅ 4. Ouvrir VSCode et choisir le bon kernel

Dans VSCode :

Ouvre ton .ipynb

En haut à droite → clique sur le nom du kernel actuel

Choisis Python 3.11 (ou “python311”)

Executer ProjetCV.ipynb pour charger LADOS
Puis executer la pipeline avec 
"""
python ./main.py
"""