import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path

import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers import models

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l1_l2

from config import CFG


def load_sbert(model_path: str, 
               sbert_path: str, 
               pretrained_model: str = CFG.PRETRAINED_MODEL):
    model_path  = Path(model_path)
    s_bert_path = model_path / f"{sbert_path}"

    # If trained model saved in local directory, load model. 
    # Otherwise load pretrained model
    if s_bert_path.is_dir():
        s_bert = SentenceTransformer(s_bert_path)

    else:
        # load Pre-trained Transformer model
        word_embedding_model = models.Transformer(pretrained_model)
        
        # Add Pooling layer
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension()
        )
        # Add Dense layer 
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=512, 
            activation_function=torch.nn.Tanh()
        )
        
        s_bert = SentenceTransformer(
            modules=[word_embedding_model, pooling_model, dense_model]
        )

    return s_bert

    
def get_classifier():
    inputs = Input(shape=(512,))
    x = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.003, 0.003))(inputs)
    x = Dropout(0.2)(x)
    x = Dense(180, activation='relu', kernel_regularizer=l1_l2(0.003, 0.003))(x)
    x = Dropout(0.2)(x)
    outputs = Dense(15, activation='sigmoid')(x)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model