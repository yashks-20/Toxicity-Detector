import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding

df = pd.read_csv("D:\\COLLAGE PROJECT\\Toxicity Detector\\train.csv")

X = df['comment_text']
y = df[df.columns[2:]].values

MAX_FEATURES = 30000

vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=400,
    output_mode='int'
)

vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)

train = dataset.take(int(len(dataset) * .7))
val = dataset.skip(int(len(dataset) * .7)).take(int(len(dataset) * .2))
test = dataset.skip(int(len(dataset) * .9)).take(int(len(dataset) * .1))

model = Sequential()

# Create the embedding layer
model.add(Embedding(MAX_FEATURES + 1, 32))

# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(32, activation='tanh')))

# Feature extractor Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))

# Final layer
model.add(Dense(6, activation='sigmoid'))

model.save("toxicity_detector.h5")

