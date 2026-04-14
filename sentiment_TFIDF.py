import pandas as pd
import random 
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#from gensim.models import Word2Vec

# ---- ETAPE 1 : Charger la data ----
df = pd.read_csv('IMDB Dataset.csv')

def augment_data(df):

    negatives = [
        "I am not satisfied",
        "This is bad",
        "I would not recommend this",
        "Very disappointing experience",
        "Totally useless product",
        "I am very unhappy with this",
        "Worst purchase ever",
        "This does not work properly"
    ]

    new_texts = []
    new_labels = []

    for _ in range(500):
        new_texts.append(random.choice(negatives))
        new_labels.append("negative")  # 0 = negative

    df_aug = pd.DataFrame({
        "review": new_texts,
        "sentiment": new_labels
    })

    return pd.concat([df, df_aug], ignore_index=True)


df = augment_data(df)


print("Data chargée :", df.shape)



print(df['sentiment'].value_counts())


# ---- ETAPE 2 : Nettoyer le texte (version améliorée) ----
# Télécharger les ressources NLTK (une seule fois)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialiser le lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Garder ces mots négatifs importants pour le sentiment
mots_a_garder = {'not', 'no', 'never', 'neither', 'nobody',
                 'nothing', 'nowhere', 'nor', "n't", 'without'}

stop_words = stop_words - mots_a_garder

def nettoyer(texte):
    # 1. Supprimer les balises HTML
    texte = re.sub(r'<.*?>', ' ', texte)

    # 2. Mettre en minuscules
    texte = texte.lower()


    def handle_negation(texte):
        mots = texte.split()
        resultat = []
        negatif = False
    
        mots_negatifs_trigger = {'not', 'no', 'never', 'neither', 'without'}
    
        for mot in mots:
            if mot in mots_negatifs_trigger:
                negatif = True
                resultat.append(mot)
            elif mot in {'.', '!', '?', 'but'}:
                negatif = False
                resultat.append(mot)
            elif negatif:
            # Add "NOT_" prefix to the word after negation
                resultat.append('NOT_' + mot)
            else:
                resultat.append(mot)
    
        return ' '.join(resultat)

    texte = handle_negation(texte)

    # 3. Supprimer les URLs
    texte = re.sub(r'http\S+|www\S+', '', texte)

    # 4. Corriger les contractions anglaises
    texte = texte.replace("n't", " not")
    texte = texte.replace("'s",  "")
    texte = texte.replace("'ve", " have")
    texte = texte.replace("'re", " are")
    texte = texte.replace("'ll", " will")
    texte = texte.replace("'d",  " would")
    texte = texte.replace("'m",  " am")
    texte = texte.replace("don't",  " do not")

    

    
    # 5. Supprimer les caractères spéciaux et chiffres
    texte = re.sub(r'[^a-z\s]', '', texte)

    # 6. Tokeniser (découper en mots)
    mots = texte.split()

    # 7. Supprimer les stopwords SAUF les mots négatifs importants
    mots = [m for m in mots if m not in stop_words]

    # 8. Lemmatisation — ramener chaque mot à sa forme de base
    
    mots = [lemmatizer.lemmatize(m) for m in mots]

    # 9. Supprimer les mots trop courts SAUF les mots négatifs importants 
    mots_negatifs = {'not', 'no', 'nor', 'never', 'nothing', 
                 'nobody', 'nowhere', 'neither', 'without'}

    mots = [m for m in mots if len(m) > 2 or m in mots_negatifs]

    # 10. Rejoindre les mots
    texte = ' '.join(mots)

    # 11. Supprimer les espaces multiples
    texte = re.sub(r'\s+', ' ', texte).strip()

    return texte


# ---- ETAPE 3 : Préparer X et y ----

print(df.columns)
X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Division train/test OK !")
print("Taille train :", X_train.shape)
print("Taille test  :", X_test.shape)

# ---- ETAPE 4 : TF-IDF amélioré ----
vectorizer = TfidfVectorizer(
    max_features=10000,  # plus de mots = plus d'info
    stop_words='english', # ignorer mots inutiles
    ngram_range=(1, 2),   # paires de mots ex: "not good"
    min_df=2,             # ignorer mots très rares
    max_df=0.95,    # ignorer mots trop fréquents 
    
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print("TF-IDF OK ! Nombre de features :", X_train_vec.shape[1])

# ---- ETAPE 5 : Entraîner le modèle amélioré ----
print("\nEntraînement en cours... (2-3 minutes)")
modele = LogisticRegression(
    max_iter=1000,
    C=5,
    solver='lbfgs'
)
modele.fit(X_train_vec, y_train)
print("Modèle entraîné !")

# ---- ETAPE 6 : Évaluer ----
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import matplotlib.pyplot as plt
import numpy as np

y_pred  = modele.predict(X_test_vec)
y_proba = modele.predict_proba(X_test_vec)[:, 1]

print("\n========== RÉSULTATS ==========")

# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n1. Accuracy (Précision globale) : {round(accuracy * 100, 2)} %")
print("   → Sur 100 avis, le modèle en classe correctement combien ?")

# 2. Precision
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(y_test, y_pred, pos_label='positive')
print(f"\n2. Precision : {round(precision * 100, 2)} %")
print("   → Quand le modèle dit 'positif', il a raison combien de fois ?")

# 3. Recall
recall = recall_score(y_test, y_pred, pos_label='positive')
print(f"\n3. Recall : {round(recall * 100, 2)} %")
print("   → Sur tous les vrais avis positifs, combien le modèle trouve ?")

# 4. F1-Score
f1 = f1_score(y_test, y_pred, pos_label='positive')
print(f"\n4. F1-Score : {round(f1 * 100, 2)} %")
print("   → Équilibre entre Precision et Recall — la meilleure mesure globale")

# 5. ROC-AUC
auc = roc_auc_score((y_test == 'positive').astype(int), y_proba)
print(f"\n5. ROC-AUC Score : {round(auc, 4)}")
print("   → Entre 0.5 (nul) et 1.0 (parfait) — mesure la qualité globale du modèle")

print("\n================================")

# 6. Matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1 : Matrice de confusion
im = axes[0].imshow(cm, cmap='Blues')
axes[0].set_title('Matrice de confusion')
axes[0].set_xlabel('Prédit')
axes[0].set_ylabel('Réel')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Positif', 'Négatif'])
axes[0].set_yticklabels(['Positif', 'Négatif'])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i][j]),
                     ha='center', va='center',
                     fontsize=16, fontweight='bold',
                     color='white' if cm[i][j] > cm.max()/2 else 'black')

# Graphique 2 : Résumé des métriques
metriques = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
valeurs   = [accuracy, precision, recall, f1, auc]
couleurs  = ['#378ADD', '#1D9E75', '#EF9F27', '#7F77DD', '#D85A30']
bars = axes[1].bar(metriques, valeurs, color=couleurs, width=0.5)
axes[1].set_ylim(0, 1.1)
axes[1].set_title('Résumé des métriques')
axes[1].set_ylabel('Score')
for bar, val in zip(bars, valeurs):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f'{round(val * 100, 1)}%',
                 ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('resultats_modele.png')
plt.show()
print("\nGraphique sauvegardé : resultats_modele.png")


# ---- ETAPE 7 : Tester avec tes propres phrases ----
def predire(texte):
    texte_propre = nettoyer(texte)
    vec = vectorizer.transform([texte_propre])
    return modele.predict(vec)[0]

print("\n--- TESTS PERSONNELS ---")
print("Test 1 :", predire("This movie was absolutely amazing!"))
print("Test 2 :", predire("Terrible film, total waste of time."))
print("Test 3 :", predire("It was not bad at all, I loved it!"))
print("Test 4 :", predire("I would not recommend this to anyone."))
print("Test 5 :", predire("Worst experience ever"))


# ---- ETAPE 8 : Sauvegarder le modèle ----
pickle.dump(modele, open('modele_sentiment.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

