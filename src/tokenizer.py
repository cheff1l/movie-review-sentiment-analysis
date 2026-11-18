"""
Module: Tokenizer Pipeline and Sequence Vectorizer
Project: Movie Review Sentiment Analysis (IMDb 50k)
Author: Denis (Data & Deployment Lead)

Description:
Handles Keras Tokenizer fitting, sequence conversion, padding,
and persistence (saving/loading pickle artifacts) for model training and deployment.
"""

import os
import pickle
import argparse
from typing import List, Tuple, Union
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ReviewTokenizer:
    """Manages text tokenization, vocabulary mapping and fixed-length padding."""

    def __init__(self, vocab_size: int = 15000, max_len: int = 200, oov_token: str = "<OOV>"):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.oov_token = oov_token
        self.tokenizer = None

    def fit(self, texts: Union[List[str], np.ndarray]):
        """Fits the tokenizer on training text corpus."""
        print(f"Fitting tokenizer (vocab_size={self.vocab_size}, oov_token='{self.oov_token}')...")
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_token)
        self.tokenizer.fit_on_texts(texts)
        print(f"Total unique words indexed: {len(self.tokenizer.word_index)}")

    def transform(self, texts: Union[List[str], np.ndarray, str]) -> np.ndarray:
        """
        Converts text or list of texts into padded integer sequences.
        
        Args:
            texts: A single string or collection of strings.
            
        Returns:
            np.ndarray: Padded matrix of shape (n_samples, max_len).
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been fitted or loaded yet. Call fit() or load() first.")

        if isinstance(texts, str):
            texts = [texts]

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding="post",
            truncating="post"
        )
        return padded

    def save(self, filepath: str = "models/tokenizer.pkl"):
        """Saves fitted tokenizer to a pickle file."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tokenizer saved to: {filepath}")

    def load(self, filepath: str = "models/tokenizer.pkl"):
        """Loads a pre-fitted tokenizer from a pickle file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tokenizer file not found at: {filepath}")
        with open(filepath, "rb") as f:
            self.tokenizer = pickle.load(f)
        print(f"Tokenizer loaded successfully from: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test tokenizer sequence padding.")
    parser.add_argument("--path", type=str, default="models/tokenizer.pkl", help="Path to tokenizer file")
    args = parser.parse_args()

    # Self-test on dummy data
    t = ReviewTokenizer(vocab_size=1000, max_len=10)
    sample_corpus = [
        "this movie was absolutely fantastic and brilliant",
        "worst film ever made total waste of time"
    ]
    t.fit(sample_corpus)
    transformed = t.transform(["this movie was fantastic"])
    print("Transformed padded sequence shape:", transformed.shape)
    print("Vector:", transformed[0])
