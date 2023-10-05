# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 23:54:21 2023

@author: Gujar
"""

import pickle
import re
import string
import xgboost as xgb
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt






loaded_model = pickle.load(open('News_Detection_Trained_Model.sav', 'rb'))
loaded_vectorizer = pickle.load(open('News_Detection_Tfidf_Vectorizer.sav', 'rb'))

class Preprocessing:
    
    def __init__(self, data):
        self.data = data
    
    def text_preprocessing(self):
        pred_text = [self.data]
        preprocess_text = []
        lm = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        for text in pred_text:
            text = text.lower()
            text = re.sub('[^a-zA-Z0-9\s]', '', text)
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('https?://\S+/www\.\S+', '', text)
            text = re.sub("<.*?>+", " ", text)
            text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
            text = re.sub("\n", " ", text)
            text = re.sub("\w*\d\w*", " ", text)
            text = word_tokenize(text)
            text = [lm.lemmatize(x) for x in text if x not in stop_words]
            text = " ".join(text)
            preprocess_text.append(text)
        return preprocess_text      
    
    
class Prediction:
    
    def __init__(self, pred_data, model, vectorizer):
        self.pred_data = pred_data
        self.model = model
        self.vectorizer = vectorizer
    
    def prediction_model(self):
        preprocess_data = Preprocessing(self.pred_data).text_preprocessing()
        data = self.vectorizer.transform(preprocess_data)
        prediction = self.model.predict(data)
        probability = self.model.predict_proba(data)

        
        return {
            "prediction": "News is Fake." if prediction[0] == 0 else "News is Real.",
            "probability_fake": probability[0][0],
            "probability_real": probability[0][1]
        }

def main():
    
    st.title("FAKE NEWS DETECTION")
    news_title = st.text_input("News Title :")
    news_text = st.text_input("Text:")
    user_data = news_title + " " + news_text
    
    if st.button('Test Result:'):
        result = Prediction(user_data, loaded_model, loaded_vectorizer).prediction_model()
        st.success(f"Prediction: {result['prediction']}")
        st.write(f"Probability of being Fake: {result['probability_fake']:.2f}")
        st.write(f"Probability of being Real: {result['probability_real']:.2f}")
        words = Preprocessing(user_data).text_preprocessing()[0]  # Processed text
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)


              
if __name__ == '__main__':
   main()
