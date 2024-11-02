import boto3
from abc import abstractmethod
import collections
from datetime import datetime
from deep_translator import GoogleTranslator
import io 
import joblib
from langchain_core.prompts import PromptTemplate
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import matplotlib.pyplot as plt
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize 
from nltk.stem import PorterStemmer
import numpy as np
np.bool = bool
nltk.download('punkt')
nltk.download('punkt_tab')
import openai
import os
import pandas as pd
import pickle
import re
import requests
import s3fs
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import sys
import uuid
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from datetime import datetime
from deep_translator import GoogleTranslator
from nltk.tokenize import sent_tokenize 

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'OPENAI_API_KEY'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

glue_db = "is459-project-reddit-database"
glue_post_tbl = "new_posts"
glue_comment_tbl = "new_comments"
glue_skytrax_tbl = "reviews"

dynamic_posts_frame = glueContext.create_dynamic_frame.from_catalog(
    database=glue_db,
    table_name=glue_post_tbl,        
)
dynamic_comments_frame = glueContext.create_dynamic_frame.from_catalog(
    database=glue_db,
    table_name=glue_comment_tbl,        
)
dynamic_skytrax_frame = glueContext.create_dynamic_frame.from_catalog(
    database=glue_db,
    table_name=glue_skytrax_tbl,        
)

s3 = boto3.client('s3')

posts_df = dynamic_posts_frame.toDF().toPandas()
comments_df = dynamic_comments_frame.toDF().toPandas()
skytrax_df = dynamic_skytrax_frame.toDF().toPandas()

posts_df = posts_df.replace("", np.nan)
posts_df.dropna(inplace=True)

comments_df = comments_df.replace("", np.nan)
comments_df.dropna(inplace=True)

airlines = {
    'SouthwestAirlines': 'WN', 
    'Southwest_Airlines': 'WN', 
    'AmericanAir': 'AA',
    'DeltaAirlines': 'DL',
    'HawaiianAirlines': 'HA',
    'frontierairlines': 'F9',
    'delta': 'DL'
}
posts_df['Code'] = posts_df['subreddit'].map(airlines)

posts_df = posts_df.drop_duplicates(subset="id", keep="first")
comments_df = comments_df.drop_duplicates(subset="id", keep="first")

code_post_dict = posts_df.set_index('id')['Code'].to_dict()
comments_df['Code'] = comments_df['post_id'].map(code_post_dict)

skytrax_df = skytrax_df.drop_duplicates(subset=["airline", "username", "title", "publishedDate"], keep="first")
skytrax_airlines = {
    'southwest-airlines': 'WN', 
    'american-airlines': 'AA',
    'delta-air-lines': 'DL',
    'hawaiian-airlines': 'HA',
    'frontier-airlines': 'F9'
}
skytrax_df['Code'] = skytrax_df['airline'].map(skytrax_airlines)

posts_df = posts_df.dropna()
comments_df = comments_df.dropna()
skytrax_df = skytrax_df.dropna()

DetectorFactory.seed = 42
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

translator = GoogleTranslator(source='auto', target='english')

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

def chunk_text(text, max_length=5000):
    chunks = []
    while len(text) > max_length:
        split_index = text[:max_length].rfind(' ')
        if split_index == -1:
            split_index = max_length
        chunks.append(text[:split_index])
        text = text[split_index:].strip()
    chunks.append(text)
    return chunks
    
def translate_text(text):
    try:
        if not is_english(text):
            if len(text) > 5000:
                chunks = chunk_text(text)
                translated_chunks = [translator.translate(chunk) for chunk in chunks]
                return ' '.join(translated_chunks)
            else:
                return translator.translate(text)
        else:
            return text
    except Exception as e:
        print(f"Error translating text: {e}")
        return text
        
        
def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing numbers, punctuation, and stopwords

    Args:
    text (str): text to preprocess

    Returns:
    text (str): preprocessed text
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
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
    aspects = lda_model.transform(tfidf_vector)
    dominant_aspect = aspects.argmax(axis=1)
    df['topic'] = pd.Series(dominant_aspect).apply(lambda x: list(topic_dict.keys())[x])
    df['topic'] = df['topic'].str.replace(f"[{string.punctuation}\d]", "", regex=True)
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
        openai_model = openai.OpenAI(api_key=args['OPENAI_API_KEY'])
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


class DF_Dataset(Dataset):
    def parse(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method to preprocess text data from a DataFrame.

        Args:
        df (pd.DataFrame): DataFrame to parse and preprocess.

        Returns:
        pd.DataFrame: Processed DataFrame.
        """
        print("Parsing DataFrame")
        
        if "publishedDate" in df.columns:
            df["review"] = df["review"].apply(translate_text)
            df["review"] = df["review"].replace("", np.nan)
            df["review"] = df["review"].replace("[deleted]", np.nan)

            df["title"] = df["title"].apply(translate_text)
            df["title"] = df["title"].replace("", np.nan)
            df["title"] = df["title"].replace("[deleted]", np.nan)

            df = df.dropna(subset=["review", "title"]).reset_index(drop=True)
        
            df['publishedDate'] = df['publishedDate'].apply(lambda x: re.sub(r'(\d+)(st|nd|rd|th)', r'\1', x))
            df['publishedDate'] = pd.to_datetime(df['publishedDate'], errors='coerce')
            df = df.dropna(subset=['publishedDate'])
            df['publishedDate'] = df['publishedDate'].dt.strftime('%Y-%m-%d %H:%M:%S')

            df['content'] = df["title"] + " " + df["review"]
            df['id'] = [uuid.uuid4() for _ in range(len(df))]
            df = df.rename(columns={'publishedDate': 'date'})
            df = df.drop(columns=['airline', 'username', 'rating', 'title', 'verified', 'review', 'recommend'])

        elif "title" in df.columns:
            df["content"] = df["content"].apply(translate_text)
            df["content"] = df["content"].replace("", np.nan)
            df["content"] = df["content"].replace("[deleted]", np.nan)

            df["title"] = df["title"].apply(translate_text)
            df["title"] = df["title"].replace("", np.nan)
            df["title"] = df["title"].replace("[deleted]", np.nan)
            
            df.dropna(inplace=True)
            df = df.reset_index(drop=True)

            df["content"] = df["title"] + " " + df["content"]
            df = df.drop(columns=['title', 'username', 'commentCount', 'score', 'subreddit'])

        else:
            df["content"] = df["content"].apply(translate_text)
            df["content"] = df["content"].replace("", np.nan)
            df["content"] = df["content"].replace("[deleted]", np.nan)
            
            df.dropna(inplace=True)
            df = df.reset_index(drop=True)

            df = df.drop(columns=['username', 'score', 'post_id', 'parent_id'])

        df["content"] = df["content"].apply(preprocess_text)
        
        print(f"Parsed DataFrame with shape: {df.shape}")
        return df


csv_buffer = io.StringIO()

codes = comments_df['Code'].unique()

for code in codes:
    try:
        vader = SentimentIntensityAnalyzer()
        
        # Comments
        comments_lda_file = s3.get_object(Bucket="is459-project-data", Key=f"reddit/models/{code}_comments_lda_model.pkl")['Body'].read()
        comments_lda_model = pickle.loads(comments_lda_file)

        comments_vectorizer_file = s3.get_object(Bucket="is459-project-data", Key=f"reddit/models/{code}_comments_vectorizer.pkl")['Body'].read()
        comments_vectorizer_model = pickle.loads(comments_vectorizer_file)
        
        comments_topic_dict_file = s3.get_object(Bucket="is459-project-data", Key=f"reddit/models/{code}_comments_topic_dict.pkl")['Body'].read()
        comments_topic_dict = pickle.loads(comments_topic_dict_file)

        comments_data = DF_Dataset(comments_df[comments_df['Code'] == code].copy(), vectorizer=comments_vectorizer_model, lda_model=comments_lda_model, topic_dict=comments_topic_dict, vader_model=vader)
        comments_data.data.to_csv(csv_buffer, index=False)
        
        s3.put_object(Bucket="is459-project-output-data", Key=f"reddit/{code}_comments_{datetime.utcnow().strftime('%Y-%m-%d')}.csv", Body=csv_buffer.getvalue())
        
        
        # Posts
        posts_lda_file = s3.get_object(Bucket="is459-project-data", Key=f"reddit/models/{code}_posts_lda_model.pkl")['Body'].read()
        posts_lda_model = pickle.loads(posts_lda_file)

        posts_vectorizer_file = s3.get_object(Bucket="is459-project-data", Key=f"reddit/models/{code}_posts_vectorizer.pkl")['Body'].read()
        posts_vectorizer_model = pickle.loads(posts_vectorizer_file)
        
        posts_topic_dict_file = s3.get_object(Bucket="is459-project-data", Key=f"reddit/models/{code}_posts_topic_dict.pkl")['Body'].read()
        posts_topic_dict = pickle.loads(posts_topic_dict_file)
        
        posts_data = DF_Dataset(posts_df[posts_df['Code'] == code].copy(), vectorizer=posts_vectorizer_model, lda_model=posts_lda_model, topic_dict=posts_topic_dict, vader_model=vader)
        
        s3.put_object(Bucket="is459-project-output-data", Key=f"reddit/{code}_posts_{datetime.utcnow().strftime('%Y-%m-%d')}.csv", Body=csv_buffer.getvalue())

        
        # Skytrax
        skytrax_data = DF_Dataset(skytrax_df[skytrax_df['Code'] == code].copy(), vectorizer=posts_vectorizer_model, lda_model=posts_lda_model, topic_dict=posts_topic_dict, vader_model=vader)
        
        s3.put_object(Bucket="is459-project-output-data", Key=f"reddit/{code}_skytrax_{datetime.utcnow().strftime('%Y-%m-%d')}.csv", Body=csv_buffer.getvalue())
        
    except Exception as e:
        print(f"Error loading file from S3: {e}")


job.commit()