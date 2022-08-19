# -*- coding: utf-8 -*-
"""
Deep Learning word2vec smaller dataset

"""

! pip install lime
! pip install transformers

## for bag-of-words
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, feature_selection
## for explainer
from lime import lime_text
## for word embedding
import gensim
import gensim.downloader as gensim_api
## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
## for bert language model
import transformers

import re
import nltk
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from string import punctuation
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')

!gdown "https://drive.google.com/uc?id=1x3B9h4zH-XCnqPXCt6BKbcfnaDOORJHA&confirm=t"

import pandas as pd
def read_dataframe(tsv_file: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_file, delimiter='\t', dtype=object)
    # replaces all "null" or "NaN" values with an empty string
    df.fillna("", inplace=True)
    # labels the columns in the dataset using the data dictionary described in the README
    df.columns = [
        'id',                # Column 1: the ID of the statement ([ID].json).
        'label',             # Column 2: the label.
        'statement',         # Column 3: the statement.
        'subjects',          # Column 4: the subject(s).
        'speaker',           # Column 5: the speaker.
        'speaker_job_title', # Column 6: the speaker's job title.
        'state_info',        # Column 7: the state info.
        'party_affiliation', # Column 8: the party affiliation.
        
        # Column 9-13: the total credit history count, including the current statement.
        'count_1', # barely true counts.
        'count_2', # false counts.
        'count_3', # half true counts.
        'count_4', # mostly true counts.
        'count_5', # pants on fire counts.
        'context' # Column 14: the context (venue / location of the speech or statement).
    ]
    
    return df
df = read_dataframe('/content/train.tsv').sample(frac=1.0).reset_index(drop=False)

df

labels = df.label.unique().tolist()
def getlabel(label):
  return 1 if label in [ 'false','barely-true','pants-fire'] else 0 
codes = {labels[i]:i for i in range(len(labels))}
df['bin_label'] = df['label'].apply(getlabel)
df['code_label'] = df[['label']].replace(codes).label
df = df[['label', 'bin_label', 'code_label', 'statement']].dropna()

df.columns =['label', 'bin_label', 'code_label', 'text']
df = df.head(3000)
print(df.shape)
df.head()

def process_text(text):
  text = re.sub(r'http\S+', '', str(text))  # removing urls
  for punc in punctuation+'—”–“’'  :
    text = text.replace(punc, '')
  text = text.lower()
  return text

def lemmatize_word(word):
  for p in 'varsn':
    lemmatized = lemmatizer.lemmatize(word, pos=p)
    if word is not lemmatized:
      return lemmatized
  return word

def preprocess(text):
  text = process_text(text)
  wordlist = text.split()
  text = ' '.join([lemmatize_word(word) for word in wordlist if word not in stop_words])
  return text
def scale_numerical(data):
  scaler = MinMaxScaler()
  data[data.columns] = scaler.fit_transform(data[data.columns])

df['unproc_text'] = df['text']
df['text'] = df['text'].apply(preprocess)
df.head()

#DataFlair - Split the dataset
df_train, df_test=train_test_split(df, test_size=0.7, random_state=7)

df_train

nlp = gensim_api.load("word2vec-google-news-300")

corpus = df_train["text"]

## create list of lists of unigrams
lst_corpus = []
for string in corpus:
   lst_words = string.split()
   lst_grams = [" ".join(lst_words[i:i+1]) 
               for i in range(0, len(lst_words), 1)]
   lst_corpus.append(lst_grams)

## detect bigrams and trigrams
bigrams_detector = gensim.models.phrases.Phrases(lst_corpus, 
                 delimiter=" ".encode(), min_count=5, threshold=10)
bigrams_detector = gensim.models.phrases.Phraser(bigrams_detector)
trigrams_detector = gensim.models.phrases.Phrases(bigrams_detector[lst_corpus], 
            delimiter=" ".encode(), min_count=5, threshold=10)
trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)

nlp = gensim.models.word2vec.Word2Vec(lst_corpus, size=300,   
            window=8, min_count=1, sg=1, iter=30)

word = "trump"
fig = plt.figure()
## word embedding
tot_words = [word] + [tupla[0] for tupla in 
                 nlp.most_similar(word, topn=20)]
X = nlp[tot_words]
## pca to reduce dimensionality from 300 to 3
pca = manifold.TSNE(perplexity=40, n_components=3, init='pca')
X = pca.fit_transform(X)
## create dtf
dtf_ = pd.DataFrame(X, index=tot_words, columns=["x","y","z"])
dtf_["input"] = 0
dtf_["input"].iloc[0:1] = 1
## plot 3d
from mpl_toolkits.mplot3d import Axes3D
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dtf_[dtf_["input"]==0]['x'], 
           dtf_[dtf_["input"]==0]['y'], 
           dtf_[dtf_["input"]==0]['z'], c="black")
ax.scatter(dtf_[dtf_["input"]==1]['x'], 
           dtf_[dtf_["input"]==1]['y'], 
           dtf_[dtf_["input"]==1]['z'], c="red")
ax.set(xlabel=None, ylabel=None, zlabel=None, xticklabels=[], 
       yticklabels=[], zticklabels=[])
for label, row in dtf_[["x","y","z"]].iterrows():
    x, y, z = row
    ax.text(x, y, z, s=label)

## tokenize text
tokenizer = kprocessing.text.Tokenizer(lower=True, split=' ', 
                     oov_token="NaN", 
                     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(lst_corpus)
dic_vocabulary = tokenizer.word_index
## create sequence
lst_text2seq= tokenizer.texts_to_sequences(lst_corpus)
## padding sequence
X_train = kprocessing.sequence.pad_sequences(lst_text2seq, 
                    maxlen=15, padding="post", truncating="post")

sns.heatmap(X_train==0, vmin=0, vmax=1, cbar=False)
plt.show()

i = 0

## list of text: ["I like this", ...]
len_txt = len(df_train["text"].iloc[i].split())
print("from: ", df_train["text"].iloc[i], "| len:", len_txt)

## sequence of token ids: [[1, 2, 3], ...]
len_tokens = len(X_train[i])
print("to: ", X_train[i], "| len:", len(X_train[i]))

## vocabulary: {"I":1, "like":2, "this":3, ...}
print("check: ", df_train["text"].iloc[i].split()[0], 
      " -- idx in vocabulary -->", 
      dic_vocabulary[df_train["text"].iloc[i].split()[0]])

print("vocabulary: ", dict(list(dic_vocabulary.items())[0:5]), "... (padding element, 0)")

corpus = df_test["text"]

## create list of n-grams
lst_corpus = []
for string in corpus:
    lst_words = string.split()
    lst_grams = [" ".join(lst_words[i:i+1]) for i in range(0, 
                 len(lst_words), 1)]
    lst_corpus.append(lst_grams)
    
## detect common bigrams and trigrams using the fitted detectors
lst_corpus = list(bigrams_detector[lst_corpus])
lst_corpus = list(trigrams_detector[lst_corpus])
## text to sequence with the fitted tokenizer
lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

## padding sequence
X_test = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=15,
             padding="post", truncating="post")

## start the matrix (length of vocabulary x vector size) with all 0s
embeddings = np.zeros((len(dic_vocabulary)+1, 300))
for word,idx in dic_vocabulary.items():
    ## update the row with vector
    try:
        embeddings[idx] =  nlp[word]
    ## if word not in model then skip and the row stays all 0s
    except:
        pass

word = "trump"
print("dic[word]:", dic_vocabulary[word], "|idx")
print("embeddings[idx]:", embeddings[dic_vocabulary[word]].shape, 
      "|vector")

## code attention layer

## input
x_in = layers.Input(shape=(15,))
## embedding
x = layers.Embedding(input_dim=embeddings.shape[0],  
                     output_dim=embeddings.shape[1], 
                     weights=[embeddings],
                     input_length=15, trainable=False)(x_in)
## 2 layers of bidirectional lstm
x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2, 
                         return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2))(x)
## final dense layers
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
y_out = layers.Dense(2, activation='softmax')(x)
## compile
model = models.Model(x_in, y_out)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

y_train = df_train['bin_label']
## encode y
dic_y_mapping = {n:label for n,label in 
                 enumerate(np.unique(y_train))}
inverse_dic = {v:k for k,v in dic_y_mapping.items()}
y_train = np.array([inverse_dic[y] for y in y_train])
## train
training = model.fit(x=X_train, y=y_train, batch_size=256, 
                     epochs=10, shuffle=True, verbose=1, 
                     validation_split=0.3)

## plot loss and accuracy
metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
ax[0].set(title="Training")
ax11 = ax[0].twinx()
ax[0].plot(training.history['loss'], color='black')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss', color='black')
for metric in metrics:
    ax11.plot(training.history[metric], label=metric)
ax11.set_ylabel("Score", color='steelblue')
ax11.legend()
ax[1].set(title="Validation")
ax22 = ax[1].twinx()
ax[1].plot(training.history['val_loss'], color='black')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss', color='black')
for metric in metrics:
     ax22.plot(training.history['val_'+metric], label=metric)
ax22.set_ylabel("Score", color="steelblue")
plt.show()

predicted_prob = model.predict(X_test)
predicted = [dic_y_mapping[np.argmax(pred)] for pred in 
             predicted_prob]

from sklearn import metrics as skmetrics
y_test = df_test['bin_label']
code_1 = {0:'True', 1:'Fake'}
classes = [0,1]
classes_names = ['True','Fake']
def evaluate(y_train, classes, classes_names):
  # EVALUATIOM
  y_test_array = pd.get_dummies(y_test, drop_first=False).values
    

  ## Accuracy, Precision, Recall
  accuracy = skmetrics.accuracy_score(y_test, predicted)
  try:
    auc = skmetrics.roc_auc_score(y_test, predicted_prob, 
                              multi_class="ovr")
  except:
    auc = skmetrics.roc_auc_score(y_test, predicted_prob[:,1], 
                            multi_class="ovr")
  print("Accuracy:",  round(accuracy,2))
  print("Auc:", round(auc,2))
  print("Detail:")
  print(skmetrics.classification_report(y_test, predicted))
      
  ## Plot confusion matrix
  cm = skmetrics.confusion_matrix(y_test, predicted)
  print(cm)
  fig, ax = plt.subplots()
  sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
              cbar=False)
  ax.set(xlabel="Pred", ylabel="True", xticklabels=classes_names, 
        yticklabels=classes_names, title="Confusion matrix")
  plt.yticks(rotation=0)

  fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,8))
  ## Plot roc
  for i in range(len(classes)):
      fpr, tpr, thresholds = skmetrics.roc_curve(y_test_array[:,i],  
                            predicted_prob[:,i])
      ax[0].plot(fpr, tpr, lw=3, 
                label='{0} (area={1:0.2f})'.format(classes_names[i], 
                                skmetrics.auc(fpr, tpr))
                )
  ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
  ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], 
            xlabel='False Positive Rate', 
            ylabel="True Positive Rate (Recall)", 
            title="Receiver operating characteristic")
  ax[0].legend(loc="lower right")
  ax[0].grid(True)
      
  ## Plot precision-recall curve
  for i in range(len(classes)):
      precision, recall, thresholds = skmetrics.precision_recall_curve(
                  y_test_array[:,i], predicted_prob[:,i])
      ax[1].plot(recall, precision, lw=3, 
                label='{0} (area={1:0.2f})'.format(classes_names[i], 
                                    skmetrics.auc(recall, precision))
                )
  ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', 
            ylabel="Precision", title="Precision-Recall curve")
  ax[1].legend(loc="best")
  ax[1].grid(True)
  plt.show()
  return predicted, predicted_prob

def explain(i, predicted, predicted_prob, model, predbool):
  txt_instance = df_test["text"].iloc[i]
  ## check true value and predicted value
  print("True:", code_1[y_test[i]], "--> Pred:", code_1[predicted[i]], "| Prob:", round(np.max(predicted_prob[i]),2))
  ## show explanation
  explainer = lime_text.LimeTextExplainer(class_names=classes_names)
  explained = explainer.explain_instance(txt_instance, 
            model.predict_proba, num_features=3)
  explained.show_in_notebook(text=txt_instance, predict_proba=predbool)

predicted, predicted_prob = evaluate(y_train, classes, classes_names)

## input
x_in = layers.Input(shape=(15,))
## embedding
x = layers.Embedding(input_dim=embeddings.shape[0],  
                     output_dim=embeddings.shape[1], 
                     weights=[embeddings],
                     input_length=15, trainable=False)(x_in)
## 2 layers of bidirectional lstm
x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2, 
                         return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2))(x)
## final dense layers
x = layers.Dense(64, activation='relu')(x)
y_out = layers.Dense(6, activation='softmax')(x)
## compile
model2 = models.Model(x_in, y_out)
model2.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model2.summary()

y_train = np.array(df_train['code_label'].tolist())
training = model2.fit(x=X_train, y=y_train, batch_size=256, 
                     epochs=11, shuffle=True, verbose=1, validation_split=0.3)

## plot loss and accuracy
metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
ax[0].set(title="Training")
ax11 = ax[0].twinx()
ax[0].plot(training.history['loss'], color='black')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss', color='black')
for metric in metrics:
    ax11.plot(training.history[metric], label=metric)
ax11.set_ylabel("Score", color='steelblue')
ax11.legend()
ax[1].set(title="Validation")
ax22 = ax[1].twinx()
ax[1].plot(training.history['val_loss'], color='black')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss', color='black')
for metric in metrics:
     ax22.plot(training.history['val_'+metric], label=metric)
ax22.set_ylabel("Score", color="steelblue")
plt.show()

predicted_prob = model2.predict(X_test)
dic_y_mapping = {0:0,1:1,2:2,3:3,4:4,5:5,}
y_test = df_test['code_label']
code_1 = {0: 'half-true',
 1: 'mostly-true',
 2: 'false',
 3: 'true',
 4: 'barely-true',
 5: 'pants-fire'}
classes = list(code_1.keys())
classes_names = list(code_1.values())
predicted = [dic_y_mapping[np.argmax(pred)] for pred in predicted_prob]
y_test_array = pd.get_dummies(y_test, drop_first=False).values
  

## Accuracy, Precision, Recall
accuracy = skmetrics.accuracy_score(y_test, predicted)
try:
  auc = skmetrics.roc_auc_score(y_test, predicted_prob, 
                            multi_class="ovr")
except:
  auc = skmetrics.roc_auc_score(y_test, predicted_prob[:,1], 
                          multi_class="ovr")
print("Accuracy:",  round(accuracy,2))
print("Auc:", round(auc,2))
print("Detail:")
print(skmetrics.classification_report(y_test, predicted))
    
## Plot confusion matrix
cm = skmetrics.confusion_matrix(y_test, predicted)
print(cm)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes_names, 
      yticklabels=classes_names, title="Confusion matrix")
plt.yticks(rotation=0)

fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,8))
## Plot roc
for i in range(len(classes)):
    fpr, tpr, thresholds = skmetrics.roc_curve(y_test_array[:,i],  
                          predicted_prob[:,i])
    ax[0].plot(fpr, tpr, lw=3, 
              label='{0} (area={1:0.2f})'.format(classes_names[i], 
                              skmetrics.auc(fpr, tpr))
              )
ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], 
          xlabel='False Positive Rate', 
          ylabel="True Positive Rate (Recall)", 
          title="Receiver operating characteristic")
ax[0].legend(loc="lower right")
ax[0].grid(True)
    
## Plot precision-recall curve
for i in range(len(classes)):
    precision, recall, thresholds = skmetrics.precision_recall_curve(
                y_test_array[:,i], predicted_prob[:,i])
    ax[1].plot(recall, precision, lw=3, 
              label='{0} (area={1:0.2f})'.format(classes_names[i], 
                                  skmetrics.auc(recall, precision))
              )
ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', 
          ylabel="Precision", title="Precision-Recall curve")
ax[1].legend(loc="best")
ax[1].grid(True)
plt.show()



