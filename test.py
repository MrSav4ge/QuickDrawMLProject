from show_img import transformdata
import json
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten

# number of points to use from input. If there are not enough,
# then the array will be padded with PADDING_VALUE
MAX_LENGTH = 10000
PADDING_VALUE = 0
# number of images to use
IMAGES_USED = 10000

# load file
def load_from_file(filename):
    with open(filename) as f:
        obj = []
        # loop through lines
        for i in f.readlines():
            # convert to dict
            d = json.loads(i)
            # add if quality
            if d["recognized"]:
                obj.append(d)
    return obj


def get_label(label: str) -> int:
    if label == "airplane":
        return 0
    elif label == "cactus":
        return 1
    elif label == "lighthouse":
        return 2
    else:
        print(f"Invalid label found: {label}")
        exit(1)


files = [
    "/Users/jakelanders/code/QuickDrawMLProject/raw_data/full-simplified-airplane.ndjson",
    "/Users/jakelanders/code/QuickDrawMLProject/raw_data/full-simplified-cactus.ndjson",
    "/Users/jakelanders/code/QuickDrawMLProject/raw_data/full-simplified-lighthouse.ndjson",
]

objects = []

for i in files:
    objects.extend(load_from_file(i))

# shuffle the data
random.shuffle(objects)

# split into labels and normalized data
drawings = []
labels = []

for i in objects[:IMAGES_USED]:
    transform = transformdata(i["drawing"])
    labels.append(get_label(i["word"]))
    if len(transform) < MAX_LENGTH:
        for _ in range(len(transform), MAX_LENGTH):
            transform.append([PADDING_VALUE, PADDING_VALUE])
    elif len(transform) > MAX_LENGTH:
        transform = transform[:MAX_LENGTH]
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

# create model
model = Sequential()
model.add(Flatten())
model.add(Dense(units=64, activation="relu"))
model.add(Dense(units=500, activation="relu"))
model.add(Dense(units=700, activation="relu"))
model.add(Dense(units=3, activation="softmax"))

# compile with category
model.compile(loss="categorical_crossentropy", optimizer="adam")

# train
model.fit(x=train_X, y=train_y, epochs=10, batch_size=32)

# test
loss = model.evaluate(x=test_X, y=test_y)
print("Loss:", loss)
