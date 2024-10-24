# Base class for dataset
import json
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import nltk
from nltk.corpus import stopwords
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DetectorFactory.seed = 42
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

with open('../models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('../models/lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)

with open('../data/topic_dict.json', 'r') as f:
    topic_dict = json.load(f)


###### Useful functions
def is_english(text):
    """
    Check if text is in English

    Args:
    text (str): text to check
    """
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing numbers, punctuation, and stopwords

    Args:
    text (str): text to preprocess

    Returns:
    text (str): preprocessed text
    """
    # convert to lowercase
    text = text.lower()
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # remove whitespaces
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

class Dataset(json_object=json_obj, vectorizer=vectorizer, lda_model=lda_model):
    def __init__(self) -> None:
        self.data = self.parse(json_object)
        self.vectorizer = vectorizer
        self.lda_model = lda_model
        self.aspects = self.extract_aspect()
        self.vader_model = SentimentIntensityAnalyzer()
        self.topic_dict = topic_dict
        self.get_sentiment()
        self.get_absa_pair()


    def parse(self, json_object: object) -> object:
        """
        Method to load json object and preprocess text

        Args:
        json_object (object): json object to parse

        Returns:
        object: parsed json object as a dataframe
        """
        print("Parsing json objects")
        data = json.loads(json_object)
        if "title" in data[0]:
            # dealing with posts
            # check for english-only text
            df = pd.DataFrame(data, columns=["id", "date", "title", "content", "username", "commentCount", "score", "subreddit"])
            df = df[df["content"].apply(is_english)]
            df = df.reset_index(drop=True)

            # combine title with content and preprocess
            df["content"] = df["title"] + " " +  df["content"]
            df = df.drop(columns=['title', 'username', 'commentCount', 'score', 'subreddit', 'Code']) 

        else:
            # dealing with comments
            df = pd.DataFrame(data, columns=["id", "date", "content", "username", "score", "post_id", "parent_id"])
            df = df[df["content"].apply(is_english)]
            df = df.reset_index(drop=True)

            df = df.drop(columns=['username', 'score', 'post_id', 'parent_id']) 

        # resultant dataframe should only have id, date and content columns
        df["content"] = df["content"].apply(preprocess_text)
        print("Parsed json objects of size {df.shape}")
        return df

    def extract_aspect(self):
        """
        Exctract aspects from self.data using LDA model

        Returns:
        list: list of dominant aspects in self.data
        """
        print("Extracting aspects")
        # vectorize text
        tfidf_vector = self.vectorizer.transform(self.data["content"])
        # extract aspects using LDA model
        aspects = self.lda_model.transform(tfidf_vector)
        dominant_aspect = aspects.argmax(axis=1)
        return dominant_aspect
    
    def get_sentiment(self):
        """
        Get sentiment of text using VADER

        Returns:
        float: sentiment score
        """
        self.data['sentiment'] = self.data['content'].apply(lambda x: self.vader_model.polarity_scores(x)['compound'])
        return

    def get_absa_pair(self):
        """
        Add aspect-sentiment pair of text using VADER to self.data
        """
        print("Getting sentiment")
        self.data["topic"] = self.data["topic"].apply(lambda x: list(self.topic_dict.keys())[x])
