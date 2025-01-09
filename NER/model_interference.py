import dataset
import model
import tensorflow as tf
import numpy as np
from collections import Counter
import keras
from keras import ops
from model_train import lowercase_and_convert_to_ids

def tokenize_and_convert_to_ids(text):
    tokens = text.split()
    return lowercase_and_convert_to_ids(tokens)

with open('./NER/trainedModels/data', "w", encoding="utf-8") as f:
    tag_length=f.read()
    vocab_size=f.read()


ner_model = model.NERModel(tag_length, vocab_size, embed_size=128, num_heads=4, filter_size=64)


ner_model.load_weights('./NER/trainedModels/nerModel')

