import tensorflow as tf
import numpy as np
import pandas as pd
import gensim
from sklearn.manifold import TSNE
!pip install Phyme
import Phyme

vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=2000,
    output_mode='int',
    output_sequence_length=30)
dataset = tf.data.TextLineDataset("/content/sample_data/sentences.txt")
vectorizer.adapt(dataset)
vectorizer.get_vocabulary()[:5]
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

num_tokens = len(voc) + 2
long_embedding_dim = 300
short_embedding_dim = 3
short_embedding_matrix = np.zeros((num_tokens, short_embedding_dim))

model = gensim.models.KeyedVectors.load_word2vec_format('/content/sample_data/GoogleNews-vectors-negative300.bin.gz', binary=True)  
words = list(word for word in model.vocab.keys())
long_embedding = [model[word] for word in words]
tsne = TSNE(n_components = short_embedding_dim, init = 'random', random_state = 10, perplexity = 100)
tsne_vectors = tsne.fit_transform(long_embedding)
short_embedding = dict(zip(words,tsne.fit_transform(tsne_vectors)))

for word, i in word_index.items():
  short_embedding_vector = short_embedding.get(word)
  if short_embedding_vector is not None:
    embedding_matrix[i] = short_embedding_vector
    hits += 1
  else:
    misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

pickle.dump(embedding_matrix,open("embedding_matrix.pkl",'wb'))
pickle.dump(embedding_words,open("embedding_words.pkl",'wb'))

embedding_layer = tf.keras.layers.Embedding(num_tokens, embedding_dim, embedding_initializer=tf.keras.initializers.Constant(embedding_matrix)),trainable=False)

ph = Phyme()
ph.get_perfect_rhymes(word).values()

input = tf.keras.Input((None,), dtype="str")
embedded_sequences = short_embedding_layer(input)
x = tf.keras.layers.Dense(10, activation="relu")(embedded_sequences)
x = tf.keras.layers.Dense(5, activation="relu")(x)
x = layers.Dropout(.5)(x)
x = custom_loss(x,input)
loss = tf.keras.losses.MeanAbsoluteError()
opt = tf.optimizers.Adam(learning_rate=0.001)
