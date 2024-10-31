# Base and children class for Dataset objects
# This script contains the base class Dataset and how to perform ABSA.
# Example of the implementation can be found in test.ipynb
# Explanation of ABSA steps can be found in absa.ipynb

from abc import abstractmethod
import collections
from dotenv import load_dotenv
import json
from langchain_core.prompts import PromptTemplate
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import nltk
from nltk.corpus import stopwords
import openai
import os
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer

load_dotenv()
DetectorFactory.seed = 42
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
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

def get_aspect(df, vectorizer=None, lda_model=None, topic_dict=None):
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
    def __init__(self, object, dataset_name=None, vectorizer=None, lda_model=None, vader_model=None, topic_dict=None) -> None:
        self.name = dataset_name
        # main template method
        self.data = self.parse(object)
        # insert hook: if vectorizer, lda_model and topic_dict are not provided, prepare them
        if not vectorizer or not lda_model or not topic_dict or not vader_model:
            self.prepare_ABSA()
        else:
            self.vectorizer = vectorizer
            self.lda_model = lda_model
            self.topic_dict = topic_dict
            self.vader_model = vader_model
        self.perform_ABSA()
    
    # MAIN TEMPLATE METHODS
    def prepare_ABSA(self):
        """
        Prepare ABSA by setting up vectorizer and LDA model

        Returns:
        Modifies self.vectorizer, self.lda_model, self.vader_model and self.topic_dict
        """
        print("Preparing dataset for ABSA...")
        self.prepare_vectorizer()
        self.prepare_lda_model()
        self.prepare_vader_model()
        print("Dataset prepared for ABSA")

    def perform_ABSA(self):
        """
        Perform ABSA on text data

        Returns:
        Modifies self.data containing "content", "sentiment" and "aspect" columns
        """
        print("Performing ABSA...")
        print("Extracting aspects...")
        self.aspects = self.extract_aspect()
        print("Getting sentiment...")
        self.get_sentiment()
        print("ABSA completed")
        

    # FUNCTIONAL METHODS
    def prepare_vectorizer(self):
        """
        Prepare vectorizer for text data

        Returns:
        Modifies self.X, self.vectorizer and self.feature_names
        """
        print(f"Preparing vectorizer...")
        # initialize and train vectorizer
        vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
        self.X = vectorizer.fit_transform(self.data['content'])
        self.vectorizer = vectorizer
        # retrieve feature names
        self.feature_names = vectorizer.get_feature_names_out()

        # save vectorizer
        with open(f'../models/{self.name}_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"Vectorizer saved as {self.name}_vectorizer.pkl")
        return

    def prepare_lda_model(self):
        """
        Prepare LDA model, extract topics and generate titles using chatgpt

        Returns:
        Modifies self.lda_model and self.topic_dict
        """
        print(f"Preparing LDA model...")
        # initialize all dependencies for lda model
        topic_dict = collections.defaultdict(list)
        openai_model = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
        lda_model.fit(self.X)
        self.lda_model = lda_model

        # document_topics = lda_model.transform(self.X)
        # dominant_topic = document_topics.argmax(axis=1)
        
        # get the top 50 features for each topic
        topics = self.lda_model.components_

        for idx, topic in enumerate(topics):
            top_features = [self.feature_names[j] for j in topic.argsort()[:-20]]
            # feed chatgpt the top 20 features and generate a title
            prompt = f"""Generate a unique noun phrase or one-word topic for posts that contain the following features. 
            This topic will be used for Aspect-Based Sentiment Analysis on social media data. 
            Ensure the topic is different from previously generated topics. 
            Feature names:\n{", ".join(top_features)}\nTopic:"""
            prompt = PromptTemplate.from_template(prompt)
            response = openai_model.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt.template}],
                max_tokens=10,
                temperature=1,
            )

            title = response.choices[0].message.content.strip()
            # deal with duplicate titles
            if title in topic_dict:
                title = title + "_" + str(idx)
            # add title to topic dictionary
            topic_dict[title] = [self.feature_names[i] for i in topic.argsort()]

        self.topic_dict = topic_dict

        # save lda model and topic dictionary
        with open(f'../models/{self.name}_lda_model.pkl', 'wb') as f:
            pickle.dump(lda_model, f)
        with open(f'../data/{self.name}_topic_dict.pkl', 'wb') as f:
            pickle.dump(topic_dict, f)

        print(f"LDA model saved as {self.name}_lda_model.pkl")
        print(f"Topic dictionary saved as {self.name}_topic_dict.pkl")
        return

    def prepare_vader_model(self):
        """
        Prepare VADER model for sentiment analysis

        Returns:
        Modifies self.vader_model
        """
        print(f"Preparing VADER model...")
        self.vader_model = SentimentIntensityAnalyzer()
        with open(f'../models/{self.name}_vader_model.pkl', 'wb') as f:
            pickle.dump(self.vader_model, f)

        print(f"VADER model saved as {self.name}_vader_model.pkl")
        return

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
    
class CSV_Dataset(Dataset):
    def parse(self, csv_object: object) -> object:
        """
        Polymorphosized method to load csv object and preprocess text

        Args:
        object (object): csv object to parse

        Returns:
        object: parsed csv object as a dataframe
        """
        print("Parsing csv objects")
        df = csv_object
        df['content'] = df['content'].astype(str)
        # check for english-only text
        df = df[df["content"].apply(is_english)]
        # to remove the old index
        df = df.reset_index(drop=True)
        # resultant dataframe should only have id, date and content columns
        df["content"] = df["content"].apply(preprocess_text)
        print(f"Parsed csv objects of size {df.shape}")
        return df