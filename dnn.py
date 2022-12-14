import json
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.regularizers import l2
import gc
import os
from datetime import datetime
import utils
from constants import IMAGES_USED

files = [
    "full-simplified-airplane.ndjson",
    "full-simplified-cactus.ndjson",
    "full-simplified-lighthouse.ndjson",
]

objects = []

for i in files:
    objects.extend(utils.load_from_file(i))

# shuffle the data
random.shuffle(objects)

# split into labels and normalized data
drawings = []
labels = []

for i in objects[:IMAGES_USED]:
    transform = utils.transformdata(i["drawing"])
    labels.append(utils.get_label(i["word"]))
    transform = utils.pad_data(transform)
    drawings.append(transform)


drawings = np.array(drawings)
labels = np.array(labels)

# convert labels to one-hot vector
# [0,2,3,1,2]
# [[1,0,0,0,0]
# [0,0,1,0,0]
# [0,0,0,1,0]]
one_hot = np.zeros((labels.size, labels.max() + 1))
one_hot[np.arange(labels.size), labels] = 1

# split into training and testing
train_X = drawings[: int(0.8 * len(drawings))]
test_X = drawings[int(0.8 * len(drawings)) :]
train_y = one_hot[: int(0.8 * len(one_hot))]
test_y = one_hot[int(0.8 * len(one_hot)) :]

# free memory
del drawings
del objects
del labels
gc.collect()

# create model
model = Sequential()
model.add(Flatten())
model.add(Dense(units=64, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dense(units=500, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dense(units=700, activation="relu", kernel_regularizer=l2(0.01)))
model.add(Dense(units=3, activation="softmax", kernel_regularizer=l2(0.01)))

# compile with category
model.compile(loss="categorical_crossentropy", optimizer="adam")

# train
model.fit(x=train_X, y=train_y, epochs=10, batch_size=32)

# test
loss = model.evaluate(x=test_X, y=test_y)
print("Test Loss:", loss)

# test single samples to get an idea of accuracy
correct = 0
predictions = model.predict(
    test_X[:1000],
    verbose=0,
)
for i in range(1000):
    # print(f"Predicted: {np.argmax(predictions[i])} Actual: {np.argmax(test_y[i])}")

    if np.argmax(predictions[i]) == np.argmax(test_y[i]):
        correct += 1

print(f"Test Accuracy: {correct / 1000}")

# save the model

model.save(f"{os.getcwd()}/models/model-{datetime.now().strftime('%s')}.h5")
