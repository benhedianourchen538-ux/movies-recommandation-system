import pandas as pd
import random 
import re
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             classification_report)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.phrases import Phrases, Phraser
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
    texte = texte.replace("don't",  "do not")
    
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


# ---- ETAPE 2 : Nettoyer ----
print("Nettoyage en cours...")
df['review_propre'] = df['review'].apply(nettoyer)
print("Nettoyage terminé !")

# ---- ETAPE 3 : Tokeniser pour Word2Vec ----
# Word2Vec a besoin d'une liste de mots par avis

sentences = [review.split() for review in df['review_propre']]
phrases = Phrases(sentences, min_count=2, threshold=5)
bigram = Phraser(phrases)

sentences_bigram = [bigram[s] for s in sentences]


df['review_propre'] = df['review_propre'].str.replace("not bad", "not_bad")


# ---- ETAPE 4 : Entraîner Word2Vec ----
print("Entraînement Word2Vec...")
w2v_model = Word2Vec (
    sentences,
    vector_size=200,  # taille du vecteur par mot
    window=7,         # nombre de mots autour à considérer
    min_count=1,      # ignorer les mots rares
    workers=4,        # utiliser 4 coeurs CPU
    epochs=10,       # nombre de passages sur la data
    sg=1
)
print("Word2Vec entraîné !")

#Calculer TF-IDF

tfidf = TfidfVectorizer()
tfidf.fit(df['review_propre'])

tfidf_dict = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
tfidf = TfidfVectorizer(ngram_range=(1,2))
# ---- ETAPE 5 : Transformer chaque avis en vecteur ----
# On fait la moyenne des vecteurs de tous les mots de l'avis
def review_to_vector(review):
    mots = review.split()
    vecteurs = []
    poids=[]

    for mot in mots:
        if mot in w2v_model.wv:
            vecteurs.append(w2v_model.wv[mot])
            poids.append(tfidf_dict.get(mot, 1))
    if len(vecteurs) == 0:
        return np.zeros(w2v_model.vector_size)

    vecteurs = np.array(vecteurs)
    poids = np.array(poids)

    return np.average(vecteurs, axis=0, weights=poids)

print("Transformation des avis en vecteurs...")
X = np.array([review_to_vector(r) for r in df['review_propre']])
y = df['sentiment']
print("Vecteurs créés ! Shape :", X.shape)

# ---- ETAPE 6 : Split train/test ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Split OK — Train:", X_train.shape, "Test:", X_test.shape)

# ---- ETAPE 7 : Entraîner le modèle ----
print("\nEntraînement du modèle de classification...")
modele = LogisticRegression(max_iter=1000, C=5)
modele.fit(X_train, y_train)
print("Modèle entraîné !")

# ---- ETAPE 8 : Évaluer ----
y_pred  = modele.predict(X_test)
y_proba = modele.predict_proba(X_test)[:, 1]

print("\n========== RÉSULTATS ==========")
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='positive')
recall    = recall_score(y_test, y_pred, pos_label='positive')
f1        = f1_score(y_test, y_pred, pos_label='positive')
auc       = roc_auc_score((y_test == 'positive').astype(int), y_proba)


print("================================\n")

# Matrice de confusion + graphique métriques
cm = confusion_matrix(y_test, y_pred, labels=['positive', 'negative'])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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
                     ha='center', va='center', fontsize=16,
                     fontweight='bold',
                     color='white' if cm[i][j] > cm.max()/2 else 'black')

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
                 f'{round(val*100,1)}%',
                 ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('resultats_word2vec.png')
plt.show()

# ---- ETAPE 9 : Tester avec tes phrases ----
def predire(texte):
    texte_propre = nettoyer(texte)
    vecteur = review_to_vector(texte_propre).reshape(1, -1)
    return modele.predict(vecteur)[0]

print("\n--- TESTS ---")
print("Test 1 :", predire("This movie was absolutely amazing!"))
print("Test 2 :", predire("Terrible film, total waste of time."))
print("Test 3 :", predire("It was not bad at all, I loved it!"))
print("Test 4 :", predire("I would not recommend this to anyone."))
print("Test 5 :", predire("Worst experience ever"))

# ---- ETAPE 10 : Sauvegarder ----
pickle.dump(modele, open('modele_sentiment.pkl', 'wb'))
w2v_model.save('word2vec_model.bin')

