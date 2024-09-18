# preprocess.py

import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import os
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

def preprocess_and_save(data_dir='./data', n_topics=20, n_top_words=10, n_top_topics=3):
    """
    Preprocesses the 20 Newsgroups dataset by applying LDA and saves the processed data.

    Args:
        data_dir (str): Directory to save the preprocessed data.
        n_topics (int): Number of topics for LDA.
        n_top_words (int): Number of top words per topic.
        n_top_topics (int): Number of top topics to consider per document.
    """
    # defining the categories
    categories = [
        'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
        'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med',
        'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast',
        'talk.politics.misc', 'talk.religion.misc'
    ]

    # loading the training and testing data
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

    # encoding the labels
    label_encoder = LabelEncoder()
    label_encoder.fit(newsgroups_train.target)

    train_labels = label_encoder.transform(newsgroups_train.target)
    test_labels = label_encoder.transform(newsgroups_test.target)

    # vectorizing the text data for LDA 
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    train_texts_vectorized = vectorizer.fit_transform(newsgroups_train.data)
    test_texts_vectorized = vectorizer.transform(newsgroups_test.data)

    # fitting the LDA algorithm
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    train_lda_td = lda_model.fit_transform(train_texts_vectorized)  # Topic distribution for training data
    test_lda_td = lda_model.transform(test_texts_vectorized)  # Topic distribution for test data

    # storing the vocabulary (word to index mapping) in vocab
    vocab = vectorizer.get_feature_names_out()

    # local functions used in the script
    def get_top_words_per_topic(lda_model, vectorizer, n_top_words=10):
        """
            Function to get the top words for each topic
        """
        topics_words = []
        for topic_idx, topic in enumerate(lda_model.components_):
            # Get the top N words for this topic
            top_words = [vocab[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics_words.append(top_words)
        return topics_words

    # getting the top words for each topic
    top_words_per_topic = get_top_words_per_topic(lda_model, vectorizer, n_top_words)

    def map_topic_distributions_to_words(topic_distributions, top_words_per_topic, n_top_topics=3):
        """
            Function to map topic distribution to words for each document
        """
        topic_words_docs = []
        for dist in topic_distributions:
            top_topic_indices = dist.argsort()[:-n_top_topics -1:-1]  # get the indices of the top N topics from the model
            top_words = [top_words_per_topic[i] for i in top_topic_indices]
            topic_words_docs.append(top_words)
        return topic_words_docs

    # mapping the topic distributions to words for both training and testing sets
    train_topics_mapped_to_words = map_topic_distributions_to_words(train_lda_td, top_words_per_topic, n_top_topics)
    test_topics_mapped_to_words = map_topic_distributions_to_words(test_lda_td, top_words_per_topic, n_top_topics)

    # prepare the data dictionaries , final training and testing data
    train_data = {
        'train_texts': newsgroups_train.data,
        'train_lda_words': train_topics_mapped_to_words,  # topic distribution mapped to words
        'train_labels': train_labels,
        'target_names': newsgroups_train.target_names
    }

    test_data = {
        'test_texts': newsgroups_test.data,
        'test_lda_words': test_topics_mapped_to_words,  # topic distribution mapped to words
        'test_labels': test_labels,
        'target_names': newsgroups_train.target_names  
    }

    # saving the data to a pickle files
    os.makedirs(data_dir, exist_ok=True)
    joblib.dump(train_data, os.path.join(data_dir, '20newsgroups_with_lda_words.pkl'))
    joblib.dump(test_data, os.path.join(data_dir, '20newsgroups_test_with_lda_words.pkl'))
    # saving the LDA model and vectorizer for future use in the prediction script
    joblib.dump(lda_model, os.path.join(data_dir, 'lda_model.pkl'))
    joblib.dump(vectorizer, os.path.join(data_dir, 'vectorizer.pkl'))
    print(f"Data saved to {data_dir}/20newsgroups_with_lda_words.pkl and {data_dir}/20newsgroups_test_with_lda_words.pkl")
    print(f"LDA model saved to {data_dir}/lda_model.pkl")
    print(f"Vectorizer saved to {data_dir}/vectorizer.pkl")

if __name__ == '__main__':
    preprocess_and_save(data_dir='./data')
