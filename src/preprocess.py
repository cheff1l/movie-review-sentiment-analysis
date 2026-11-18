"""
Module: Text Preprocessing Pipeline
Project: Movie Review Sentiment Analysis (IMDb 50k)
Author: Denis (Data & Deployment Lead)

Description:
Provides modular functions and a pipeline class for cleaning movie reviews:
- Stripping HTML artifacts (BeautifulSoup)
- Removing URLs, emails, and non-alphabetic characters
- Converting to lowercase
- Intelligent stopword filtering preserving negation terms ('not', 'no', 'never')
- Lemmatization via NLTK WordNetLemmatizer
"""

import os
import re
import argparse
from typing import List, Union
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """Production-ready text cleaner for sentiment analysis inference and training."""

    def __init__(self):
        self._ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        
        # Load English stopwords while preserving critical negation words
        base_stops = set(stopwords.words("english"))
        negation_words = {"not", "no", "nor", "neither", "never", "none", "without", "barely", "hardly"}
        self.stop_words = base_stops - negation_words
        
        # Regex patterns precompiled for maximum execution speed
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.non_alpha_pattern = re.compile(r"[^a-zA-Z\s]")

    @staticmethod
    def _ensure_nltk_resources():
        """Ensure necessary NLTK corpora are downloaded silently."""
        for res in ["stopwords", "wordnet", "omw-1.4"]:
            try:
                nltk.data.find(f"corpora/{res}")
            except LookupError:
                nltk.download(res, quiet=True)

    def clean_text(self, text: str) -> str:
        """
        Cleans a single input review string.
        
        Args:
            text (str): Raw movie review text.
            
        Returns:
            str: Preprocessed, normalized and lemmatized string.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # 2. Remove URLs
        text = self.url_pattern.sub(" ", text)

        # 3. Remove non-letter characters
        text = self.non_alpha_pattern.sub(" ", text)

        # 4. Lowercase
        text = text.lower()

        # 5. Lemmatization and stop-word filtering
        tokens = text.split()
        cleaned = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 1
        ]

        return " ".join(cleaned)

    def process_batch(self, texts: Union[List[str], pd.Series]) -> List[str]:
        """Processes a list or pandas Series of text inputs."""
        return [self.clean_text(t) for t in texts]


def preprocess_csv(input_path: str, output_path: str):
    """CLI utility function to clean a CSV dataset and save the result."""
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    if "review" not in df.columns:
        raise ValueError("Input CSV must contain a 'review' column.")

    print(f"Total rows: {len(df)}. Removing duplicates...")
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    preprocessor = TextPreprocessor()
    print("Preprocessing text reviews...")
    df["cleaned_review"] = preprocessor.process_batch(df["review"])

    if "sentiment" in df.columns:
        df["target"] = df["sentiment"].map({"positive": 1, "negative": 0})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset successfully saved to: {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw IMDb dataset for model training.")
    parser.add_argument("--input", type=str, default="data/IMDB Dataset.csv", help="Path to raw CSV dataset")
    parser.add_argument("--output", type=str, default="data/cleaned_dataset.csv", help="Path to save cleaned CSV")
    args = parser.parse_args()

    # Quick test on sample
    cleaner = TextPreprocessor()
    sample = "<br />This movie was NOT good! Terrible acting, worst 0/10: https://imdb.com"
    print("Original:", sample)
    print("Cleaned: ", cleaner.clean_text(sample))
