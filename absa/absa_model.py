# Base class for dataset
from abc import abstractmethod
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
from nltk.stem import PorterStemmer

DetectorFactory.seed = 42
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

with open('../models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('../models/lda_model.pkl', 'rb') as f:
    lda_model = pickle.load(f)

with open('../data/topic_dict.pkl', 'rb') as f:
    topic_dict = pickle.load(f)

stemmer = PorterStemmer()

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
    # remove spaces and stopwords
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])

    return text

def get_aspect(df, vectorizer=vectorizer, lda_model=lda_model, topic_dict=topic_dict):
    """
    Get aspect of text using LDA model.

    Args:
    text (str): text to extract aspect from
    vectorizer (object): vectorizer object
    lda_model (object): lda model object

    Returns:
    str: dominant aspect of text
    """
    tfidf_vector = vectorizer.transform(df['content'])
    # extract aspects using LDA model
    aspects = lda_model.transform(tfidf_vector)

    #get the most likely topic
    dominant_aspect = aspects.argmax(axis=1)

    # x is the key so we get the value of the key 0,1,2 and then get the value of it eg. technology
    df['topic'] = pd.Series(dominant_aspect).apply(lambda x: list(topic_dict.keys())[x])

    return aspects


class Dataset:
    def __init__(self, json_object, vectorizer=vectorizer, lda_model=lda_model, topic_dict=topic_dict) -> None:
        # main template method
        self.data = self.parse(json_object)
        self.vectorizer = vectorizer
        self.lda_model = lda_model
        self.topic_dict = topic_dict
        self.aspects = self.extract_aspect()
        self.vader_model = SentimentIntensityAnalyzer()
        self.get_sentiment()

    @abstractmethod
    def parse(self, json_object: object) -> object:
        """
        Abstract method to parse JSON object to be implemented by child class.

        Return:
        dataframe containing "content" column
        """

    def extract_aspect(self):  #check what does extract aspect do 
        """
        Extract aspects from self.data using LDA model

        Returns:
        list: list of dominant aspects in self.data
        """
        print("Extracting aspects")
        # vectorize text
        return get_aspect(self.data, self.vectorizer, self.lda_model, self.topic_dict)
    
    def get_sentiment(self):
        """
        Get sentiment of text using VADER

        Returns:
        float: sentiment score
        """
        self.data['sentiment'] = self.data['content'].apply(lambda x: self.vader_model.polarity_scores(x)['compound'])
        return


class JSON_Dataset(Dataset):
    def parse(self, json_object: object) -> object:
        """
        Polymorphosized method to load json object and preprocess text

        Args:
        json_object (object): json object to parse

        Returns:
        object: parsed json object as a dataframe
        """
        print("Parsing json objects")
        # data = json.loads(json_object)
        data = json_object
        if "title" in data[0]:
            # dealing with posts
            # check for english-only text
            df = pd.DataFrame(data, columns=["id", "date", "title", "content", "username", "commentCount", "score", "subreddit"])
            df = df[df["content"].apply(is_english)]
            # to remove the old index
            df = df.reset_index(drop=True)

            # combine title with content and preprocess
            df["content"] = df["title"] + " " +  df["content"]
            df = df.drop(columns=['title', 'username', 'commentCount', 'score', 'subreddit']) 

        else:
            # dealing with comments
            df = pd.DataFrame(data, columns=["id", "date", "content", "username", "score", "post_id", "parent_id"])
            df = df[df["content"].apply(is_english)]
            df = df.reset_index(drop=True)

            df = df.drop(columns=['username', 'score', 'post_id', 'parent_id']) 

        # resultant dataframe should only have id, date and content columns
        df["content"] = df["content"].apply(preprocess_text)
        print(f"Parsed json objects of size {df.shape}")
        return df