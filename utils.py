import os
import json
from constants import PADDING_VALUE, MAX_LENGTH

__labels = {
    "airplane": 0,
    "cactus": 1,
    "lighthouse": 2,
}


def transformdata(raw):
    # transfroms 3d array into a list of tuples containing x and y values
    polylines = [list(zip(polyline[0], polyline[1])) for polyline in raw]
    coords = []

    for i in polylines:
        for (x, y) in i:
            coords.append([x, y])

    return coords


# load file
def load_from_file(filename):
    with open(f"{os.getcwd()}/raw_data/{filename}") as f:
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
    return __labels[label]


def get_label_name(label: int) -> str:
    flipped = dict((v, k) for k, v in __labels.items())
    return flipped[label]


def pad_data(row):
    if len(row) < MAX_LENGTH:
        for _ in range(len(row), MAX_LENGTH):
            row.append([PADDING_VALUE, PADDING_VALUE])
    elif len(row) > MAX_LENGTH:
        row = row[:MAX_LENGTH]
    return row
