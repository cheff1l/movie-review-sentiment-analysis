"""
Module: Sentiment Inference Service
Project: Movie Review Sentiment Analysis (IMDb 50k)
Author: Miroslav (ML/DL Engineer)

Description:
High-performance inference engine for real-time and batch sentiment analysis.
Provides the primary interface `predict_sentiment(text)` designed for seamless
integration with the Streamlit web application.
"""

import os
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import tensorflow as tf

from src.preprocess import TextPreprocessor
from src.tokenizer import ReviewTokenizer


# Global singletons for model caching (avoids re-loading weights on every web request)
_MODEL_CACHE = None
_TOKENIZER_CACHE = None
_PREPROCESSOR_CACHE = None

DEFAULT_MODEL_PATH = "models/best_model.keras"
DEFAULT_TOKENIZER_PATH = "models/tokenizer.pkl"


def get_inference_artifacts(
    model_path: str = DEFAULT_MODEL_PATH,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH
):
    """Loads and caches model, tokenizer, and preprocessor singletons."""
    global _MODEL_CACHE, _TOKENIZER_CACHE, _PREPROCESSOR_CACHE

    if _PREPROCESSOR_CACHE is None:
        _PREPROCESSOR_CACHE = TextPreprocessor()

    if _TOKENIZER_CACHE is None:
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer artifact not found at '{tokenizer_path}'. "
                "Ensure models/tokenizer.pkl is present."
            )
        _TOKENIZER_CACHE = ReviewTokenizer()
        _TOKENIZER_CACHE.load(tokenizer_path)

    if _MODEL_CACHE is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at '{model_path}'. "
                "Ensure models/best_model.keras is present."
            )
        _MODEL_CACHE = tf.keras.models.load_model(model_path)

    return _MODEL_CACHE, _TOKENIZER_CACHE, _PREPROCESSOR_CACHE


# Lexicon of strong sentiment indicators for client-side explainability & highlighting
POSITIVE_LEXICON = {
    "masterpiece", "brilliant", "excellent", "superb", "fantastic", "amazing",
    "wonderful", "great", "loved", "enjoyed", "stunning", "gem", "classic",
    "hilarious", "touching", "captivating", "flawless", "exceptional", "perfect",
    "favorite", "outstanding", "impressive", "delightful", "terrific", "good"
}

NEGATIVE_LEXICON = {
    "terrible", "horrible", "awful", "worst", "waste", "boring", "poor",
    "disappointing", "bad", "crap", "ridiculous", "stupid", "annoying",
    "mess", "pathetic", "unwatchable", "pointless", "failure", "cliche",
    "dull", "lame", "garbage", "trash", "painful", "useless", "hated"
}


def extract_word_influences(raw_text: str, preprocessed_text: str) -> List[Dict[str, Any]]:
    """
    Identifies words with prominent sentiment influence for visual highlighting in UI.
    """
    tokens = preprocessed_text.split()
    word_weights = []
    
    for token in tokens:
        clean_token = token.lower()
        if clean_token in POSITIVE_LEXICON:
            word_weights.append({"word": token, "sentiment": "positive", "weight": 0.85})
        elif clean_token in NEGATIVE_LEXICON:
            word_weights.append({"word": token, "sentiment": "negative", "weight": -0.85})
        elif clean_token in {"not", "never", "no", "barely", "hardly", "without"}:
            word_weights.append({"word": token, "sentiment": "negation", "weight": -0.5})
            
    return word_weights


def predict_sentiment(
    text: str,
    model_path: str = DEFAULT_MODEL_PATH,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH
) -> Dict[str, Any]:
    """
    Main inference entrypoint for analyzing movie review sentiment.
    
    Args:
        text: Raw review string submitted by user.
        model_path: Path to .keras model weights.
        tokenizer_path: Path to serialized .pkl tokenizer.
        
    Returns:
        dict: {
            'sentiment': 'Positive' | 'Negative',
            'confidence': float (percentage 0.0 - 100.0),
            'probability': float (raw sigmoid output 0.0 - 1.0),
            'cleaned_text': str,
            'word_highlights': list of influencing words
        }
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "sentiment": "Neutral",
            "confidence": 0.0,
            "probability": 0.5,
            "cleaned_text": "",
            "word_highlights": []
        }

    model, tokenizer, preprocessor = get_inference_artifacts(model_path, tokenizer_path)

    # 1. Clean
    cleaned_text = preprocessor.clean_text(text)
    
    # 2. Vectorize & Pad
    padded_seq = tokenizer.transform(cleaned_text)

    # 3. Predict Probability
    prob = float(model.predict(padded_seq, verbose=0)[0][0])

    # 4. Determine Verdict & Confidence
    is_positive = prob >= 0.5
    sentiment_label = "Positive" if is_positive else "Negative"
    confidence = (prob if is_positive else (1.0 - prob)) * 100.0

    # 5. Extract explainability highlights
    word_highlights = extract_word_influences(text, cleaned_text)

    return {
        "sentiment": sentiment_label,
        "confidence": round(confidence, 2),
        "probability": round(prob, 4),
        "cleaned_text": cleaned_text,
        "word_highlights": word_highlights
    }


def predict_batch(
    texts: List[str],
    model_path: str = DEFAULT_MODEL_PATH,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH
) -> List[Dict[str, Any]]:
    """Performs batched inference over multiple review strings."""
    model, tokenizer, preprocessor = get_inference_artifacts(model_path, tokenizer_path)

    cleaned_list = preprocessor.process_batch(texts)
    padded_batch = tokenizer.transform(cleaned_list)
    raw_probs = model.predict(padded_batch, verbose=0).flatten()

    results = []
    for raw_text, clean_text, prob in zip(texts, cleaned_list, raw_probs):
        prob = float(prob)
        is_pos = prob >= 0.5
        results.append({
            "text": raw_text,
            "sentiment": "Positive" if is_pos else "Negative",
            "confidence": round((prob if is_pos else (1.0 - prob)) * 100.0, 2),
            "probability": round(prob, 4)
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict sentiment of a movie review.")
    parser.add_argument("--text", type=str, default="This film was an absolute masterpiece! Loved every single scene.", help="Review text")
    args = parser.parse_args()

    print(f"\nAnalyzing: \"{args.text}\"")
    result = predict_sentiment(args.text)
    print("Inference Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
