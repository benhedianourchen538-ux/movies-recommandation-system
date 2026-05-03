import pandas as pd
import re
import pickle
import numpy as np
import nltk
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
import matplotlib.pyplot as plt

# Télécharger ressources NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
mots_a_garder = {'not', 'no', 'never', 'neither', 'nobody',
                 'nothing', 'nowhere', 'nor', "n't", 'without'}
stop_words = stop_words - mots_a_garder

# ---- Negation Handling ----
def handle_negation(texte):
    mots = texte.split()
    resultat = []
    negatif = False
    triggers = {'not', 'no', 'never', 'neither', 'without'}
    for mot in mots:
        if mot in triggers:
            negatif = True
            resultat.append(mot)
        elif mot in {'.', '!', '?', 'but'}:
            negatif = False
            resultat.append(mot)
        elif negatif:
            resultat.append('NOT_' + mot)
        else:
            resultat.append(mot)
    return ' '.join(resultat)

# ---- Nettoyage ----
def nettoyer(texte):
    texte = re.sub(r'<.*?>', ' ', texte)
    texte = texte.lower()
    texte = re.sub(r'http\S+|www\S+', '', texte)
    texte = texte.replace("n't", " not")
    texte = texte.replace("'s",  "")
    texte = texte.replace("'ve", " have")
    texte = texte.replace("'re", " are")
    texte = texte.replace("'ll", " will")
    texte = texte.replace("'d",  " would")
    texte = texte.replace("'m",  " am")
    texte = handle_negation(texte)
    texte = re.sub(r'[^a-z\s_]', '', texte)
    mots = texte.split()
    mots = [m for m in mots if m not in stop_words]
    mots = [lemmatizer.lemmatize(m) for m in mots]
    mots_negatifs = {'not', 'no', 'nor', 'never', 'nothing',
                     'nobody', 'nowhere', 'neither', 'without'}
    mots = [m for m in mots if len(m) > 2 or m in mots_negatifs]
    texte = ' '.join(mots)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

# ---- ETAPE 1 : Charger la data ----
print("Chargement de la data...")
df = pd.read_csv('IMDB Dataset.csv')

# Utiliser 5000 avis pour aller plus vite
# (BERT est lent — augmente ce nombre si tu as un GPU)
df = df.sample(5000, random_state=42)
df['review_propre'] = df['review'].apply(nettoyer)
print(f"Data chargée et nettoyée : {df.shape}")

# ---- ETAPE 2 : Charger BERT ----
print("\nChargement de BERT... (téléchargement ~400MB la première fois)")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()  # mode évaluation — pas d'entraînement de BERT

# Utiliser GPU si disponible sinon CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = bert_model.to(device)
print(f"BERT chargé ! Appareil utilisé : {device}")

# ---- ETAPE 3 : Encoder les avis avec BERT ----
def encoder_bert(textes, batch_size=32):
    tous_vecteurs = []
    total = len(textes)

    for i in range(0, total, batch_size):
        batch = textes[i:i+batch_size]

        # Tokeniser le batch
        encoded = tokenizer(
            batch,
            padding=True,        # même longueur pour tous
            truncation=True,     # max 512 tokens
            max_length=128,      # limiter pour aller plus vite
            return_tensors='pt'  # format PyTorch
        )

        # Envoyer sur GPU/CPU
        input_ids      = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Passer dans BERT sans calculer les gradients
        with torch.no_grad():
            outputs = bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # Prendre le vecteur [CLS] — représente l'avis entier
        cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        tous_vecteurs.append(cls_vectors)

        # Afficher la progression
        print(f"  Encodé {min(i+batch_size, total)}/{total} avis...", end='\r')

    print()
    return np.vstack(tous_vecteurs)

print("\nEncodage des avis avec BERT (patience...)")
X = encoder_bert(df['review_propre'].tolist())
y = df['sentiment']
print(f"Encodage terminé ! Shape : {X.shape}")

# ---- ETAPE 4 : Split train/test ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train : {X_train.shape} | Test : {X_test.shape}")

# ---- ETAPE 5 : Entraîner le classifieur ----
print("\nEntraînement du classifieur...")
modele = LogisticRegression(max_iter=1000, C=5)
modele.fit(X_train, y_train)
print("Classifieur entraîné !")

# ---- ETAPE 6 : Évaluer ----
y_pred  = modele.predict(X_test)
y_proba = modele.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='positive')
recall    = recall_score(y_test, y_pred, pos_label='positive')
f1        = f1_score(y_test, y_pred, pos_label='positive')
auc       = roc_auc_score((y_test == 'positive').astype(int), y_proba)

print("\n========== RÉSULTATS BERT ==========")
print(f"1. Accuracy  : {round(accuracy  * 100, 2)} %")
print(f"2. Precision : {round(precision * 100, 2)} %")
print(f"3. Recall    : {round(recall    * 100, 2)} %")
print(f"4. F1-Score  : {round(f1        * 100, 2)} %")
print(f"5. ROC-AUC   : {round(auc, 4)}")
print("=====================================\n")

# Graphiques
cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].imshow(cm, cmap='Blues')
axes[0].set_title('Matrice de confusion — BERT')
axes[0].set_xlabel('Prédit')
axes[0].set_ylabel('Réel')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Positif', 'Négatif'])
axes[0].set_yticklabels(['Positif', 'Négatif'])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i][j]),
                     ha='center', va='center', fontsize=16,
                     fontweight='bold',
                     color='white' if cm[i][j] > cm.max()/2 else 'black')

metriques = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
valeurs   = [accuracy, precision, recall, f1, auc]
couleurs  = ['#378ADD', '#1D9E75', '#EF9F27', '#7F77DD', '#D85A30']
bars = axes[1].bar(metriques, valeurs, color=couleurs, width=0.5)
axes[1].set_ylim(0, 1.1)
axes[1].set_title('Résumé des métriques — BERT')
axes[1].set_ylabel('Score')
for bar, val in zip(bars, valeurs):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f'{round(val*100,1)}%',
                 ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('resultats_bert.png')
plt.show()

# ---- ETAPE 7 : Tester avec tes phrases ----
def predire(texte):
    texte_propre = nettoyer(texte)
    vecteur = encoder_bert([texte_propre])
    return modele.predict(vecteur)[0]

print("--- TESTS ---")
print("Test 1 :", predire("This movie was absolutely amazing!"))
print("Test 2 :", predire("Terrible film, total waste of time."))
print("Test 3 :", predire("It was not bad at all!"))
print("Test 4 :", predire("I would not recommend this to anyone."))

# ---- ETAPE 8 : Sauvegarder ----
pickle.dump(modele, open('modele_bert.pkl', 'wb'))




