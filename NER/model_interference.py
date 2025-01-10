import model
import tensorflow as tf
import numpy as np
from collections import Counter
import keras
from keras import ops
# from model_train import lowercase_and_convert_to_ids

with open('./NER/trainedModels/data', "r", encoding="utf-8") as f:
    tag_length=f.readline()
    vocab_size=f.readline()
    mapping=f.readline()
    vocab=f.readlines()

print(vocab)
StringLookup = keras.layers.StringLookup(vocabulary=vocab)

# close the file
f.close()    
def lowercase_and_convert_to_ids(tokens):
    tokens = tf.strings.lower(tokens)
    return StringLookup(tokens)

def tokenize_and_convert_to_ids(text):
    tokens = text.split()
    return lowercase_and_convert_to_ids(tokens)

ner_model = model.NERModel(int(tag_length), int(vocab_size), embed_size=128, num_heads=4, filter_size=64)

ner_model.build(1024)

ner_model.load_weights('./NER/trainedModels/nerModel.weights.h5')

sample_input = tokenize_and_convert_to_ids(
    "The highest mountain in the world is Everest"
)
sample_input = ops.reshape(sample_input, [1, -1])
print(sample_input)

output = ner_model.predict(sample_input)
prediction = np.argmax(output, axis=-1)[0]
prediction = [mapping[i] for i in prediction]

print(prediction)