Solution MMCTR Track 2 : Architecture xDeepFM Enhanced

Ce dépôt contient la solution basée sur le modèle xDeepFM (eXtreme Deep Factorization Machine), amélioré pour le challenge Multimodal.
Ce modèle combine la puissance des interactions explicites (CIN) avec un mécanisme d'Attention pour l'historique utilisateur.

📌 Architecture du Modèle

Le modèle prend en entrée des données hétérogènes (Sparse IDs, Dense Embeddings, Séquences) et produit une probabilité de clic (CTR).

Schéma des Entrées/Sorties

graph TD
    %% -- INPUTS --
    subgraph "1. Entrées (Inputs)"
        UserID[User Placeholder]
        Context[Contexte<br/>(Likes/Views)]
        ItemID[Item ID]
        ItemImg[Item Image<br/>(128d Float)]
        History[Historique Séquentiel<br/>(Liste d'IDs)]
    end

    %% -- EMBEDDING & PROJECTION --
    subgraph "2. Embedding & Projection Layer"
        Emb_User[User Emb]
        Emb_Ctx[Context Emb]
        Emb_Item[Item Emb]
        
        Proj_Img[<b>Projection Multimodale</b><br/>Linear + LayerNorm + DICE]
        
        Attn_Hist[<b>Attention Pooling</b><br/>(Target-Aware)]
    end

    %% CONNEXIONS COUCHE 2
    UserID --> Emb_User
    Context --> Emb_Ctx
    ItemID --> Emb_Item
    ItemImg --> Proj_Img
    
    %% Target pour Attention
    Emb_Item -.-> TargetComb
    Proj_Img -.-> TargetComb
    TargetComb[Target: ID + Image] -.-> Attn_Hist
    History --> Attn_Hist

    %% -- FEATURE STACK --
    Stack[<b>Stacked Features</b><br/>(Batch, 6 Champs, 128d)]
    Emb_User --> Stack
    Emb_Ctx --> Stack
    Emb_Item --> Stack
    Proj_Img --> Stack
    Attn_Hist --> Stack

    %% -- xDeepFM CORE --
    subgraph "3. xDeepFM Core (3 Branches)"
        direction TB
        
        %% Branche 1 : Linear
        Linear[<b>Linear Component</b><br/>(1st Order)<br/>Capture les biais globaux]
        
        %% Branche 2 : CIN
        CIN[<b>CIN (Compressed Interaction Network)</b><br/>(Explicit High-Order)<br/>Capture les interactions vectorielles]
        
        %% Branche 3 : DNN
        DNN[<b>DNN (Deep Neural Network)</b><br/>(Implicit High-Order)<br/>Capture les relations non-linéaires]
        
        Stack --> Linear
        Stack --> CIN
        Stack --> DNN
    end

    %% -- OUTPUT --
    subgraph "4. Sortie"
        Sum((Somme))
        Sigmoid{Sigmoid}
        Output[<b>Score CTR</b><br/>Probabilité [0-1]]
        
        Linear --> Sum
        CIN --> Sum
        DNN --> Sum
        Sum --> Sigmoid --> Output
    end

    style ItemImg fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    style Proj_Img fill:#fff3e0,stroke:#e65100
    style CIN fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style DNN fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    style Attn_Hist fill:#f3e5f5,stroke:#4a148c,stroke-width:2px


🧠 Composants Clés

1. Entrées Multimodales (Image)

Le vecteur image (128d) ne passe pas par une table d'embedding classique. Il traverse un module de projection :
Linear(128->128) ➔ LayerNorm ➔ DICE Activation.
Cela permet d'aligner l'espace sémantique de l'image avec celui des IDs.

2. Attention Pooling (Amélioration)

Contrairement au xDeepFM standard qui fait une moyenne de l'historique, nous utilisons un mécanisme d'attention (inspiré de DIN).

Query : L'item cible (ID + Image).

Key/Value : Les items de l'historique.

Résultat : L'historique est pondéré dynamiquement selon la pertinence avec la cible.

3. Les 3 Branches de Prédiction

Linear : Mémorisation simple des caractéristiques ("Wide").

CIN (Compressed Interaction Network) : Interactions explicites d'ordre élevé. Configuration : [256, 128].

DNN : Généralisation via un réseau profond. Configuration : [512 -> 256 -> 1].

⚙️ Configuration Optimisée

Le fichier xdeepfm_config.yaml utilise les hyperparamètres suivants pour la performance :

Paramètre

Valeur

Description

Embedding Dim

128

Haute résolution, crucial pour le CIN.

CIN Layers

[256, 128]

Capture des interactions d'ordre 2 et 3.

Batch Size

4096

Stabilise l'apprentissage.

Optimiseur

AdamW

Meilleure gestion du Weight Decay (1e-5).

Dropout

0.25

Prévient le sur-apprentissage dans le DNN.

🚀 Instructions d'Entraînement

1. Installation

pip install torch pandas numpy pyarrow pyyaml tqdm scikit-learn


2. Lancer l'entraînement

python src/train_xdeepfm.py


Le modèle sera sauvegardé dans checkpoints/xDeepFM_best.pth.

3. Générer la Soumission

python src/inference_xdeepfm.py


Cela générera le fichier submission_xdeepfm.zip prêt pour le leaderboard.

Développé pour la compétition MMCTR 2025.