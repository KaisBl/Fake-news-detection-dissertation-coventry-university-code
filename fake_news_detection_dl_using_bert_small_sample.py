# -*- coding: utf-8 -*-
"""
Fake News Detection USING BERT on small sample.ipynb

"""

! pip install lime
! pip install transformers
!pip install tensorflow-text

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
df = read_dataframe('/content/train.tsv').sample(frac = 1.0).head(500)

df

labels = df.label.unique().tolist()
def getlabel(label):
  return 1 if label in [ 'false','barely-true','pants-fire'] else 0 
codes = {labels[i]:i for i in range(len(labels))}
df['bin_label'] = df['label'].apply(getlabel)
df['code_label'] = df[['label']].replace(codes).label
df = df[['label', 'bin_label', 'code_label', 'statement']].dropna()

df.columns =['label', 'bin_label', 'code_label', 'text']
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

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

bert_preprocess = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')

def get_sentence_embeding(sentences):
    preprocessed_text = bert_preprocess(sentences)
    return bert_encoder(preprocessed_text)['pooled_output']

# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.2, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(400, activation='relu')(l)
l = tf.keras.layers.Dropout(0.3)(l)
l = tf.keras.layers.Dense(100, activation='relu')(l)
l = tf.keras.layers.Dense(1, activation='sigmoid')(l)

# Use inputs and outputs to construct a final model
model1 = tf.keras.Model(inputs=[text_input], outputs = [l])

model1.summary()

y_train = np.asarray(df_train['bin_label']).astype('float32').reshape((-1,1))
y_test = np.asarray(df_test['bin_label']).astype('float32').reshape((-1,1))

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
training = model1.fit(df_train['text'],y_train , epochs=4, batch_size = 64)

y_predicted = model1.predict(df_test['text'])

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_predicted)

cm



from matplotlib import pyplot as plt
import seaborn as sn
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')



from sklearn import metrics as skmetrics
dic_y_mapping = {0:0,1:1}
predicted = [dic_y_mapping[np.argmax(pred)] for pred in 
             predicted_prob]

y_test = df_test['bin_label']
code_1 = {0: 'True',
 1: 'Fake'}
classes = list(code_1.keys())
classes_names = list(code_1.values())
y_test_array = pd.get_dummies(y_test, drop_first=False).values
  

## Accuracy, Precision, Recall
# accuracy = skmetrics.accuracy_score(y_test, predicted)
# auc = skmetrics.roc_auc_score(y_test, predicted_prob, 
#                             multi_class="ovr")
# print("Accuracy:",  round(accuracy,2))
# # print("Auc:", round(auc,2))
# print("Detail:")
# print(skmetrics.classification_report(y_test, predicted))
    
# ## Plot confusion matrix
# cm = skmetrics.confusion_matrix(y_test, predicted)
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

