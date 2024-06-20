#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import string

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')

app = Flask(__name__)

@app.route('/')
def home():
    return "Web Scraping and Text Analysis Service"

@app.route('/scrape_reviews', methods=['GET'])
def scrape_reviews():
    wm_title = []
    wm_date = []
    wm_content = []
    wm_rating = []

    for i in range(1, 3):  # Adjust range for testing; use range(1, 150) for full scraping
        link = f"https://www.amazon.in/Apple-MacBook-Air-13-3-inch-MQD32HN/product-reviews/B073Q5R6VR/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&amp;amp;reviewerType=all_reviews&amp;amp;pageNumber={i}"
        response = requests.get(link)
        soup = bs(response.content, "html.parser")

        title = soup.find_all('a', class_='review-title-content')
        review_title = [titles.get_text().strip() for titles in title]
        wm_title += review_title

        rating = soup.find_all('i', class_='review-rating')
        review_rating = [ratings.get_text().rstrip(' out of 5 stars') for ratings in rating[2:]]
        wm_rating += review_rating

        review = soup.find_all("span", {"data-hook": "review-body"})
        review_content = [reviews.get_text().strip() for reviews in review]
        wm_content += review_content

        dates = soup.find_all('span', class_='review-date')
        review_dates = [dates[i].get_text().lstrip('Reviewed in India on').strip() for i in range(2, len(rating))]
        wm_date += review_dates

    df = pd.DataFrame({'Title': wm_title, 'Ratings': wm_rating, 'Comments': wm_content, 'Date': wm_date})

    df['Date'] = pd.to_datetime(df['Date'])
    df['Ratings'] = df['Ratings'].astype(float)

    df["Comments"] = df["Comments"].apply(lambda x: clean_text(x))
    df["Title"] = df["Title"].apply(lambda x: clean_text(x))

    sid = SentimentIntensityAnalyzer()
    df["sentiments"] = df["Comments"].apply(lambda x: sid.polarity_scores(x))
    df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)

    df["nb_chars"] = df["Comments"].apply(lambda x: len(x))
    df["nb_words"] = df["Comments"].apply(lambda x: len(x.split(" ")))

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df["Comments"].apply(lambda x: x.split(" ")))]
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

    doc2vec_df = df["Comments"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    df = pd.concat([df, doc2vec_df], axis=1)

    tfidf = TfidfVectorizer(min_df=10)
    tfidf_result = tfidf.fit_transform(df["Comments"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns=["word_" + str(x) for x in tfidf.get_feature_names_out()])
    tfidf_df.index = df.index
    df = pd.concat([df, tfidf_df], axis=1)

    return df.to_json()

def clean_text(text):
    text = text.lower()
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    text = [word for word in text if not any(c.isdigit() for c in word)]
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    text = [t for t in text if len(t) > 0]
    pos_tags = pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    text = [t for t in text if len(t) > 1]
    return " ".join(text)

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

if __name__ == '__main__':
    app.run(debug=True)

