
#  MMCTR 2025 — Solution Track 2  
## xDeepFM Enhanced Model (Multimodal CTR Prediction)

Ce dépôt contient une version améliorée du modèle **xDeepFM**, optimisée pour la compétition **MMCTR 2025 – Track 2 (Multimodal CTR Prediction)**.  
L'objectif est de prédire la probabilité de clic (CTR) en exploitant des données multimodales : IDs sparses, embeddings d’images, et historiques séquentiels.

Le modèle combine :

-  **CIN** pour les interactions explicites d'ordre élevé  
-  **DNN** pour les interactions implicites  
-  **Attention pooling** pour l’historique utilisateur  
-  **Projection multimodale** pour les embeddings d’images  

---

#  1. Présentation Générale

xDeepFM Enhanced intègre plusieurs améliorations essentielles pour exploiter pleinement les données multimodales :

- **Projection des images (128d)** dans l’espace latent des IDs  
- **Attention Target-Aware** (inspiré de DIN) pour pondérer l’historique utilisateur  
- **CIN** configuré pour capturer les interactions visuelles + contextuelles  
- **DNN profond** pour identifier des relations non-linéaires complexes  
- **Fusion de toutes les branches Wide + CIN + Deep**

Il s’agit d’une architecture hybride performante et adaptée aux données complexes du challenge.

---

#  2. Architecture du Modèle

## 🔹 2.1 Entrées
Le modèle traite plusieurs types de données :

- **User ID**  
- **Item ID**  
- **Context ID** (likes, vues, device, etc.)  
- **Image vector (128d)**  
- **Historique utilisateur (séquence d’items)**  

Toutes les features sont transformées en embeddings 128d.

## 🔹 2.2 Projection Multimodale (Images)

Les images ne sont pas directement utilisées comme embeddings ID.  
Elles passent par :
```
Linear(128 → 128)
LayerNorm
DICE Activation
```
 Objectif : faire correspondre l’espace visuel et l’espace des IDs.

##  2.3 Attention Pooling (Historique)
L’historique est traité par une attention dépendante de l'item cible :
- **Query** : (Embedding Item + Embedding Image projetée)  
- **Keys/Values** : embeddings des items historiques  
Résultat : un embedding pondéré qui capture les interactions séquentielles pertinentes.
##  2.4 Stacked Features
Tous les embeddings sont concaténés dans une matrice de taille :

```
(batch_size, num_fields, 128)
```
##  2.5 xDeepFM Core

###  Wide (Linear Component)  
Capture les effets de premier ordre.

###  CIN — Compressed Interaction Network  
Capture les interactions explicites d'ordre élevé.  
Configuration :
```
CIN Layers = [256, 128]
```
###  DNN — Deep Neural Network  
Capture les interactions implicites complexes.  
Architecture :
```
[512 → 256 → 1]

```
##  2.6 Sortie

Les trois branches Wide + CIN + Deep sont sommées, puis passent dans :

```
Sigmoid → CTR

```
---

#  3. Configuration & Hyperparamètres

Tous les paramètres sont définis dans **xdeepfm_config.yaml**.

### Paramètres principaux :

| Paramètre            | Valeur        | Rôle |
|----------------------|---------------|------|
| Embedding Dim        | 128           | Crucial pour CIN |
| CIN Layers           | [256, 128]    | Interactions d'ordre 2 et 3 |
| Batch Size           | 4096          | Stable en apprentissage |
| Optimizer            | AdamW         | Meilleure régularisation |
| Weight Decay         | 1e-5          | Anti-overfitting |
| Dropout (DNN)        | 0.25          | Régularisation du Deep |

---

#  4. Entraînement

## 4.1 Installation
```bash
pip install torch pandas numpy pyarrow pyyaml tqdm scikit-learn
````
---
## 4.2 Lancer l’entraînement
```bash
python src/train_xdeepfm.py
```
Le modèle sera sauvegardé automatiquement dans :
```
checkpoints/xDeepFM_best.pth
```
---
#  5. Inference & Génération de Soumission
Lancer le script d’inférence :
```bash
python src/inference_xdeepfm.py
```
Il génère automatiquement :
```
submission_xdeepfm.zip
```
Ce fichier est **prêt à être uploadé** sur le leaderboard MMCTR.

---

#  7. Points Forts de la Solution

*  Très bonne gestion des données multimodales
*  Fusion cohérente images + IDs
*  Attention dynamique pour l'historique
*  CIN puissant pour interactions complexes
*  Architecture modulaire et claire
*  Code propre et facile à étendre

---

#  8. Licence

Projet développé dans le cadre de la compétition **MMCTR 2025**.
Libre d’utilisation pour usage académique et expérimental.
---
#  9. Contributions
Les contributions, issues ou PR sont les bienvenues.
---
#  Contac 
Pour toute question :
**Younes — MMCTR 2025 Participant**
```




