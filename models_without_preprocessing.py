# -*- coding: utf-8 -*-
"""
Machine Learning Models without pre-processing

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
from sklearn.svm import SVC


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
df = read_dataframe('/content/train.tsv')

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

#DataFlair - Split the dataset
df_train, df_test=train_test_split(df, test_size=0.7, random_state=7)

df_train

"""## Using Count Vectorizer 
https://github.com/cassieview/intro-nlp-wine-reviews

![count vect](https://raw.githubusercontent.com/cassieview/intro-nlp-wine-reviews/master/imgs/vectorchart.PNG)
![count vect](https://raw.githubusercontent.com/cassieview/intro-nlp-wine-reviews/master/imgs/ngram.PNG)
"""

# initiate count vectorizer object 
count_vectorizer = feature_extraction.text.CountVectorizer(max_features=10000, ngram_range=(1,2))

corpus = df_train["text"]
count_vectorizer.fit(corpus)
X_train = count_vectorizer.transform(corpus)
dic_vocabulary = count_vectorizer.vocabulary_

sns.heatmap(X_train.todense()[:,np.random.randint(0,X_train.shape[1],100)]==0, vmin=0, vmax=1, cbar=False).set_title('Sparse Matrix Sample')

y = df_train["label"].astype(str)
X_names = count_vectorizer.get_feature_names()
p_value_limit = 0.95
dtf_features = pd.DataFrame()
for cat in np.unique(y):
    chi2, p = feature_selection.chi2(X_train, y==cat)
    dtf_features = dtf_features.append(pd.DataFrame(
                   {"feature":X_names, "score":1-p, "y":cat}))
    dtf_features = dtf_features.sort_values(["y","score"], 
                    ascending=[True,False])
    dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
X_names = dtf_features["feature"].unique().tolist()

for cat in np.unique(y):
   print("# {}:".format(cat))
   print("  . selected features:",
         len(dtf_features[dtf_features["y"]==cat]))
   print("  . top features:", ",".join(
dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))
   print(" ")

"""We can refit the vectorizer on the corpus by giving this new set of words as input. That will produce a smaller feature matrix and a shorter vocabulary."""

vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_

"""## machine learning models"""

def train_and_predict(classifier, y_train, classes, classes_names):
  ## pipeline
  model = pipeline.Pipeline([("vectorizer", vectorizer),  
                            ("classifier", classifier)])
  ## train classifier
  model["classifier"].fit(X_train, y_train)
  ## test
  X_test = df_test["text"].values
  predicted = model.predict(X_test)
  predicted_prob = model.predict_proba(X_test)

  code_1 = dict([ item[::-1] for item in codes.items()])
  # EVALUATIOM
  y_test_array = pd.get_dummies(y_test, drop_first=False).values
    

  ## Accuracy, Precision, Recall
  accuracy = metrics.accuracy_score(y_test, predicted)
  try:
    auc = metrics.roc_auc_score(y_test, predicted_prob, 
                              multi_class="ovr")
  except:
    auc = metrics.roc_auc_score(y_test, predicted_prob[:,1], 
                            multi_class="ovr")
  print("Accuracy:",  round(accuracy,2))
  print("Auc:", round(auc,2))
  print("Detail:")
  print(metrics.classification_report(y_test, predicted))
      
  ## Plot confusion matrix
  cm = metrics.confusion_matrix(y_test, predicted)
  fig, ax = plt.subplots()
  sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, 
              cbar=False)
  ax.set(xlabel="Pred", ylabel="True", xticklabels=classes_names, 
        yticklabels=classes_names, title="Confusion matrix")
  plt.yticks(rotation=0)

  fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,8))
  ## Plot roc
  for i in range(len(classes)):
      fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i],  
                            predicted_prob[:,i])
      ax[0].plot(fpr, tpr, lw=3, 
                label='{0} (area={1:0.2f})'.format(classes_names[i], 
                                metrics.auc(fpr, tpr))
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
      precision, recall, thresholds = metrics.precision_recall_curve(
                  y_test_array[:,i], predicted_prob[:,i])
      ax[1].plot(recall, precision, lw=3, 
                label='{0} (area={1:0.2f})'.format(classes_names[i], 
                                    metrics.auc(recall, precision))
                )
  ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', 
            ylabel="Precision", title="Precision-Recall curve")
  ax[1].legend(loc="best")
  ax[1].grid(True)
  plt.show()
  return model, predicted, predicted_prob
def explain(i, predicted, predicted_prob, model, predbool):
  txt_instance = df_test["text"].iloc[i]
  ## check true value and predicted value
  print("True:", code_1[y_test[i]], "--> Pred:", code_1[predicted[i]], "| Prob:", round(np.max(predicted_prob[i]),2))
  ## show explanation
  explainer = lime_text.LimeTextExplainer(class_names=classes_names)
  explained = explainer.explain_instance(txt_instance, 
            model.predict_proba, num_features=3)
  explained.show_in_notebook(text=txt_instance, predict_proba=predbool)

"""MultinomialNB with 6 targets"""

code_1 = {0: 'half-true',
 1: 'mostly-true',
 2: 'false',
 3: 'true',
 4: 'barely-true',
 5: 'pants-fire'}

"""MultinomialNB"""

y_train = df_train['code_label']
y_test = df_test['code_label']
classes = np.unique(y_test)
classes_names = [code_1[i] for i in classes]
classifier = naive_bayes.MultinomialNB()
model, predicted, predicted_prob = train_and_predict(classifier, y_train, classes, classes_names)
explain(6, predicted, predicted_prob, model, False)
explain(9, predicted, predicted_prob, model, False)

"""SVM"""

y_train = df_train['code_label']
y_test = df_test['code_label']
classes = np.unique(y_test)
classes_names = [code_1[i] for i in classes]
classifier = SVC(gamma='auto', probability=True, kernel = 'sigmoid')
model, predicted, predicted_prob = train_and_predict(classifier, y_train, classes, classes_names)
explain(6, predicted, predicted_prob, model, False)
explain(11, predicted, predicted_prob, model, False)

"""xgboost"""

import xgboost as xgb

y_train = df_train['code_label']
y_test = df_test['code_label']
classes = np.unique(y_test)
classes_names = [code_1[i] for i in classes]
classifier = xgb.XGBClassifier()
model, predicted, predicted_prob = train_and_predict(classifier, y_train, classes, classes_names)
explain(6, predicted, predicted_prob, model, False)
explain(11, predicted, predicted_prob, model, False)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=3)

y_train = df_train['code_label']
y_test = df_test['code_label']
classes = np.unique(y_test)
classes_names = [code_1[i] for i in classes]
model, predicted, predicted_prob = train_and_predict(classifier, y_train, classes, classes_names)
explain(6, predicted, predicted_prob, model, False)
explain(11, predicted, predicted_prob, model, False)

y_train = df_train['bin_label']
y_test = df_test['bin_label']
classes = np.unique(y_test)
classes_names = [{0:'True', 1:'Fake'}[i] for i in classes]
classifier = naive_bayes.MultinomialNB()
model, predicted, predicted_prob = train_and_predict(classifier, y_train, classes, classes_names)
explain(6, predicted, predicted_prob, model, False)
explain(9, predicted, predicted_prob, model, False)

y_train = df_train['bin_label']
y_test = df_test['bin_label']
classes = np.unique(y_test)
classes_names = [{0:'True', 1:'Fake'}[i] for i in classes]
classifier = SVC(gamma='auto', probability=True, kernel = 'sigmoid')
model, predicted, predicted_prob = train_and_predict(classifier, y_train, classes, classes_names)
explain(6, predicted, predicted_prob, model, False)
explain(11, predicted, predicted_prob, model, False)

import xgboost as xgb

y_train = df_train['bin_label']
y_test = df_test['bin_label']
classes = np.unique(y_test)
classes_names = [{0:'True', 1:'Fake'}[i] for i in classes]
classifier = xgb.XGBClassifier()
model, predicted, predicted_prob = train_and_predict(classifier, y_train, classes, classes_names)
explain(6, predicted, predicted_prob, model, False)
explain(11, predicted, predicted_prob, model, False)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=3)

y_train = df_train['bin_label']
y_test = df_test['bin_label']
classes = np.unique(y_test)
classes_names = [{0:'True', 1:'Fake'}[i] for i in classes]
model, predicted, predicted_prob = train_and_predict(classifier, y_train, classes, classes_names)
explain(6, predicted, predicted_prob, model, False)
explain(11, predicted, predicted_prob, model, False)

from sklearn.ensemble import RandomForestClassifier
y_train = df_train['code_label']
y_test = df_test['code_label']
classes = np.unique(y_test)
classes_names = [code_1[i] for i in classes]
classifier = RandomForestClassifier(n_estimators = 500, random_state = 42)
model, predicted, predicted_prob = train_and_predict(classifier, y_train, classes, classes_names)
explain(6, predicted, predicted_prob, model, False)
explain(9, predicted, predicted_prob, model, False)

y_train = df_train['bin_label']
y_test = df_test['bin_label']

classes = np.unique(y_test)
classes_names = [code_1[i] for i in classes]
classifier = RandomForestClassifier(n_estimators = 500, random_state = 42)
model, predicted, predicted_prob = train_and_predict(classifier, y_train, classes, classes_names)
explain(6, predicted, predicted_prob, model, False)
explain(9, predicted, predicted_prob, model, False)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=3)

y_train = df_train['bin_label']
y_test = df_test['bin_label']
classes = np.unique(y_test)
classes_names = [{0:'True', 1:'Fake'}[i] for i in classes]
model, predicted, predicted_prob = train_and_predict(classifier, y_train, classes, classes_names)
explain(9, predicted, predicted_prob, model, False)
explain(12, predicted, predicted_prob, model, False)

y_train = df_train['code_label']
y_test = df_test['code_label']

classes = np.unique(y_test)
classes_names = [code_1[i] for i in classes]
classifier = RandomForestClassifier(n_estimators = 500, random_state = 42)
model, predicted, predicted_prob = train_and_predict(classifier, y_train, classes, classes_names)
explain(6, predicted, predicted_prob, model, False)
explain(9, predicted, predicted_prob, model, False)