import json
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.regularizers import l2
import gc
import os
from datetime import datetime
import utils
import sys
import logging
from constants import MAX_LENGTH
from threading import Thread
from matplotlib import pyplot as plt

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)


logging.info("Loading files")
files = []

# get all files in raw_data directory
d = os.fsencode(f"{os.getcwd()}/raw_data")
for f in os.listdir(d):
    filename = os.fsdecode(f)
    if filename.endswith(".ndjson"):
        files.append(filename)

objects = []

for i in files:
    objects.extend(utils.load_from_file(i))


logging.info("Shuffling")

# shuffle the data
random.shuffle(objects)

logging.info(len(objects))

logging.info("Normalizing data:")

# multithread data normalization
def normalize_batch(idx, objects, draw, lab):
    logging.info(f"Running thread: {idx}")
    tmp = []
    tmp2 = []
    for index, i in enumerate(objects):
        transform = utils.transformdata(i["drawing"])
        tmp2.append(utils.get_label(i["word"]))
        transform = utils.pad_data(transform)
        tmp.append(transform)
    draw[idx] = tmp
    lab[idx] = tmp2


# create thread pool
threads = [None] * 10
drawings_tmp = [None] * 10
labels_tmp = [None] * 10

div10 = int(len(objects) / 10)

for i in range(len(threads)):
    threads[i] = Thread(
        target=normalize_batch,
        args=[i, objects[i * div10 : i * div10 + div10], drawings_tmp, labels_tmp],
    )
    threads[i].start()

for i in range(len(threads)):
    logging.info(f"Collecting thread: {i}")
    threads[i].join()

drawings = []
labels = []

for i in range(10):
    drawings.extend(drawings_tmp[i])
    labels.extend(labels_tmp[i])

drawings = np.array(drawings)
labels = np.array(labels)

logging.info("Creating one hot vectors")

# convert labels to one-hot vector
# [0,2,3,1,2]
# [[1,0,0,0,0]
# [0,0,1,0,0]
# [0,0,0,1,0]]
one_hot = np.zeros((labels.size, labels.max() + 1))
one_hot[np.arange(labels.size), labels] = 1

logging.info("Splitting into test and train data")

# split into training and testing
train_X = drawings[: int(0.8 * len(drawings))]
test_X = drawings[int(0.8 * len(drawings)) :]
train_y = one_hot[: int(0.8 * len(one_hot))]
test_y = one_hot[int(0.8 * len(one_hot)) :]

logging.info(f"Train_X Shape: {train_X.shape}")
logging.info(f"Train_y Shape: {train_y.shape}")

# free memory
del drawings
del objects
del labels
gc.collect()

logging.info("Creating and compiling model")

# create model
model = Sequential()
model.add(Flatten())
model.add(Dense(units=200, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dense(units=1000, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dense(units=500, activation="relu", kernel_regularizer=l2(0.01)))
model.add(
    Dense(units=len(train_y[0]), activation="softmax", kernel_regularizer=l2(0.01))
)

# compile with category
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

logging.info("Training model")

# train
model.fit(x=train_X, y=train_y, epochs=10, batch_size=32)

logging.info("Testing model")

# get test loss and accuracy
loss = model.evaluate(x=test_X, y=test_y)
print("Test Loss:", loss)

# save the model
model.save(f"{os.getcwd()}/models/model-{datetime.now().strftime('%s')}.h5")

logging.info("Model saved")
