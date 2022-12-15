import os
import json
from constants import PADDING_VALUE, MAX_LENGTH
from PIL import Image, ImageDraw

__labels = {
    "airplane": 0,
    "basket": 1,
    "butterfly": 2,
    "campfire": 3,
    "coffee cup": 4,
    "door": 5,
    "fork": 6,
    "pants": 7,
    "sailboat": 8,
    "tornado": 9,
}


# def transformdata(raw):
#     # transfroms 3d array into a list of tuples containing x and y values
#     polylines = [list(zip(polyline[0], polyline[1])) for polyline in raw]
#     coords = []

#     for i in polylines:
#         for (x, y) in i:
#             coords.append([x, y])

#     return coords


def transformdata(raw):
    # transfroms 3d array into a list of tuples containing x and y values
    polylines = (list(zip(polyline[0], polyline[1])) for polyline in raw)

    # make a new image
    pil_img = Image.new("RGB", (256, 256), (255, 255, 255))
    # get a drawing context
    d = ImageDraw.Draw(pil_img)
    # render each polyline
    for polyline in polylines:
        d.line(polyline, fill=(0, 0, 0), width=1)

    coords = []
    # convert to coordinates
    for i, item in enumerate(pil_img.getdata()):
        if item != (255, 255, 255):
            coords.append([int(i % 256), int(i / 256)])

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
                if len(obj) == 1000:
                    break
    return obj[:1000]


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
