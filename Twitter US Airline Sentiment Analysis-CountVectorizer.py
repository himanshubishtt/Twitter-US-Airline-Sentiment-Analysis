#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
lemmatizer=WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


data=pd.read_csv(r"C:\Users\Home\Downloads\0000000000002747_training_twitter_x_y_train (1).csv",delimiter=",",usecols=[1,7])


# In[3]:


data["class_label"]=data["airline_sentiment"]
del data["airline_sentiment"]


# In[5]:


data.sort_values(by="class_label",inplace=True)


# In[6]:


stop=stopwords.words('english')+list(punctuation)


# In[7]:


def clean(words):
    output=[]
    for w in words:
        if w.lower() not in stop:
            output.append(w)
    return output


# In[8]:


def simplepos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
        
    else:
        return wordnet.NOUN


# In[9]:


documents=[]
for a,b in data.itertuples(index=False):
    documents.append((clean(word_tokenize(a)),b)) 


# In[10]:


def wordlemmatizer(words):
    output=[]
    for w in words:
        pos=pos_tag([w])
        cleanword=lemmatizer.lemmatize(w,pos=simplepos(pos[0][1]))
        output.append(cleanword.lower())
    return output


# In[58]:


new=[(wordlemmatizer(document),category) for document,category in documents]


# In[59]:


categories=[category for document,category in new]
textdocs=[" ".join(document) for document,category in documents]


# In[60]:


countvec=CountVectorizer(max_features=3000)
xtrainfeatures=countvec.fit_transform(textdocs)


# In[61]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()


# In[63]:


rfc.fit(xtrainfeatures,categories)


# In[64]:


testing_data=pd.read_csv(r"C:\Users\Home\Downloads\0000000000002747_test_twitter_x_test.csv",delimiter=",",usecols=[6])


# In[65]:


testdocs=[]
for i in testing_data.text:
    testdocs.append(clean(word_tokenize(i)))


# In[66]:


testdocs=[wordlemmatizer(doc) for doc in testdocs]


# In[67]:


testdocs=[" ".join(document) for document in testdocs]


# In[68]:


xtestfeatures=countvec.transform(testdocs)


# In[72]:


predictions=rfc.predict(xtestfeatures)


# In[73]:


predictions=pd.DataFrame(predictions)
predictions.to_csv(r"C:\Users\Home\Desktop\predictionstwitter.csv")

