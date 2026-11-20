"""
Module: Standalone Neural Network Training Pipeline
Project: Movie Review Sentiment Analysis (IMDb 50k)
Author: Miroslav (ML/DL Engineer)

Description:
Autonomous CLI training script that ties together preprocessing, tokenization,
model instantiation, callbacks, and metric evaluation.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.preprocess import TextPreprocessor
from src.tokenizer import ReviewTokenizer
from src.model import build_bilstm_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train BiLSTM sentiment analysis model on IMDb dataset.")
    parser.add_argument("--data-path", type=str, default="data/IMDB Dataset.csv", help="Path to input dataset CSV")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--vocab-size", type=int, default=15000, help="Maximum vocabulary size")
    parser.add_argument("--max-len", type=int, default=200, help="Maximum sequence padding length")
    return parser.parse_args()


def train_pipeline(args):
    print("=" * 70)
    print("🎬 STARTING SENTIMENT MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Dataset
    if not os.path.exists(args.data_path):
        print(f"Dataset not found at {args.data_path}. Downloading sample fallback...")
        url = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        df = pd.read_csv(url)
        df.to_csv(args.data_path, index=False)
    else:
        df = pd.read_csv(args.data_path)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"Dataset loaded: {len(df):,} unique reviews.")

    # 2. Text Preprocessing
    print("Applying modular TextPreprocessor cleaning...")
    preprocessor = TextPreprocessor()
    df["cleaned_review"] = df["review"].apply(preprocessor.clean_text)
    df["target"] = df["sentiment"].map({"positive": 1, "negative": 0})

    # 3. Train / Val / Test Split (70% / 15% / 15%)
    X = df["cleaned_review"].values
    y = df["target"].values

    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Split sizes: Train={len(X_train_raw)}, Val={len(X_val_raw)}, Test={len(X_test_raw)}")

    # 4. Tokenization & Sequence Padding
    tokenizer = ReviewTokenizer(vocab_size=args.vocab_size, max_len=args.max_len)
    tokenizer.fit(X_train_raw)

    X_train = tokenizer.transform(X_train_raw)
    X_val = tokenizer.transform(X_val_raw)
    X_test = tokenizer.transform(X_test_raw)

    os.makedirs(args.models_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.models_dir, "tokenizer.pkl")
    tokenizer.save(tokenizer_path)

    # 5. Build Model
    model = build_bilstm_model(
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        learning_rate=args.lr
    )
    model.summary()

    # 6. Callbacks
    best_model_path = os.path.join(args.models_dir, "best_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5, verbose=1),
        ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    # 7. Training Execution
    start_train = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start_train
    print(f"Training completed in {train_time:.1f} seconds.")

    # 8. Evaluation on Unseen Test Set
    print("\nEvaluating best saved model on Test Set...")
    best_model = tf.keras.models.load_model(best_model_path)
    y_pred_probs = best_model.predict(X_test, batch_size=args.batch_size).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)

    print("=" * 70)
    print(f"🏆 FINAL TEST SET BENCHMARKS:")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"ROC-AUC:   {auc:.4f}")
    print("=" * 70)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))


if __name__ == "__main__":
    cli_args = parse_args()
    train_pipeline(cli_args)
