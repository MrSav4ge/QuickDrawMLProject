import json
from PIL import Image, ImageDraw


def show():
    # read ndjson lines
    lines = open(
        "/Users/jakelanders/code/QuickDrawMLProject/full-simplified-airplane.ndjson",
        "r",
    ).readlines()
    # grab the first line, JSON parse it and fetch the 'drawing' array
    raw_drawing = json.loads(lines[100])["drawing"]

    print("before", raw_drawing)
    # zip x,y coordinates for each point in every polyline
    polylines = (list(zip(polyline[0], polyline[1])) for polyline in raw_drawing)
    # notice how the data is shuffled to (x1,y1),(x2,y2) order
    print("after", polylines)

    # make a new image
    pil_img = Image.new("RGB", (240, 270), (255, 255, 255))
    # get a drawing context
    d = ImageDraw.Draw(pil_img)
    # render each polyline
    for polyline in polylines:
        d.line(polyline, fill=(0, 0, 0), width=3)
    # display image
    pil_img.show()


def transformdata(raw):
    # transfroms 3d array into a list of tuples containing x and y values
    polylines = [list(zip(polyline[0], polyline[1])) for polyline in raw]
    coords = []

    for i in polylines:
        for (x, y) in i:
            coords.append([x, y])

    return coords


if __name__ == "__main__":
    show()
