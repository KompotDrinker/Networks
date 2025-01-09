import dataset
import model
import tensorflow as tf
import numpy as np
from collections import Counter
import keras
from keras import ops

# Make dict to categorize needed tags
def tag_dict():
    all_labels = ["[PAD]", "O", "1"]
    return dict(zip(range(0, len(all_labels) + 1), all_labels))

# Function that splits data from files and returns  
def map_record_to_training_data(record):
    record = tf.strings.split(record, sep="\t")
    length = tf.strings.to_number(record[0], out_type=tf.int32)
    tokens = record[1 : length + 1]
    tags = record[length + 1 :]
    tags = tf.strings.to_number(tags, out_type=tf.int64)
    tags += 1
    return tokens, tags

# Function lowers letters case and matches them to vocabulary ids
def lowercase_and_convert_to_ids(tokens):
    tokens = tf.strings.lower(tokens)
    return StringLookup(tokens)

mapping = tag_dict()
ds=dataset.getData()

tokens = sum(ds[0]["tokens"],[])
tokens_array = np.array(list(map(str.lower, tokens)))
counter = Counter(tokens_array)
tag_length = len(mapping)

# Determine vocabulary of sentences
vocab_size = 20000
vocabulary = [token for token, count in counter.most_common(vocab_size - 2)]

# Convert tokens to token ids
StringLookup = keras.layers.StringLookup(vocabulary=vocabulary)

# Get data from files
train_data = tf.data.TextLineDataset("./NER/data/train.txt")
test_data = tf.data.TextLineDataset("./NER/data/test.txt")

batch_size = 32

# Transform data into more suitable form for training 
train_dataset = (
    train_data.map(map_record_to_training_data) #slicing strings and returning tags and tokens in different structures
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y)) #assign ids to words (tokens)
    .padded_batch(batch_size) # organize data into batches
)
# Repeat process for test dataset
test_dataset = (
    test_data.map(map_record_to_training_data)
    .map(lambda x, y: (lowercase_and_convert_to_ids(x), y))
    .padded_batch(batch_size)
)

# Building model
ner_model = model.NERModel(tag_length, vocab_size, embed_size=128, num_heads=4, filter_size=64)

tf.config.run_functions_eagerly(True)

# Compiling model, training and saving weights
ner_model.compile(optimizer="Adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=None), metrics=['accuracy'])
ner_model.fit(train_dataset, epochs=5,validation_data=test_dataset)
ner_model.save_weights('./NER/trainedModels/nerModel')

with open('./NER/trainedModels/data', "w", encoding="utf-8") as f:
    f.write(tag_length)
    f.write(vocab_size)