"""
Module: Neural Network Model Architecture
Project: Movie Review Sentiment Analysis (IMDb 50k)
Author: Miroslav (ML/DL Engineer)

Description:
Defines the winning Deep Learning architecture (Embedding + Bidirectional LSTM)
selected after empirical benchmarking against Vanilla LSTM and CNN-BiLSTM.
"""

import os
from typing import Optional
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    SpatialDropout1D,
    GlobalMaxPooling1D,
    BatchNormalization
)


def build_bilstm_model(
    vocab_size: int = 15000,
    max_len: int = 200,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    dropout_rate: float = 0.4,
    spatial_dropout: float = 0.25,
    learning_rate: float = 1e-3
) -> Model:
    """
    Builds and compiles the production Bidirectional LSTM model for binary sentiment classification.
    
    Architecture:
        1. Input Layer (sequence of token indices)
        2. Embedding Layer (maps discrete tokens to continuous vector space)
        3. SpatialDropout1D (drops full 1D feature maps across the embedding dimension)
        4. Bidirectional LSTM (processes sequence forward and backward)
        5. GlobalMaxPooling1D (extracts the most salient features across time steps)
        6. Dense + ReLU + BatchNormalization (non-linear transformation & stability)
        7. Dropout (prevents overfitting)
        8. Dense (1 unit) + Sigmoid activation (output probability)
    """
    inputs = Input(shape=(max_len,), name="input_token_ids")
    
    # 1. Word Embeddings
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_len,
        name="word_embedding"
    )(inputs)
    
    # 2. Spatial Regularization
    x = SpatialDropout1D(spatial_dropout, name="spatial_dropout")(x)
    
    # 3. Bidirectional Recurrent Layer
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True, dropout=0.2),
        name="bidirectional_lstm"
    )(x)
    
    # 4. Feature Extraction via Global Pooling
    x = GlobalMaxPooling1D(name="global_max_pool")(x)
    
    # 5. Fully Connected Dense Projection
    x = Dense(64, activation="relu", name="dense_projection")(x)
    x = BatchNormalization(name="batch_norm")(x)
    x = Dropout(dropout_rate, name="dense_dropout")(x)
    
    # 6. Classification Output
    outputs = Dense(1, activation="sigmoid", name="sentiment_output")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="BiLSTM_Sentiment_Classifier")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


if __name__ == "__main__":
    # Smoke test of model instantiation and summary
    sample_model = build_bilstm_model()
    sample_model.summary()
