import numpy as np
import json


def parse_line(ndjson_line):
    sample = json.loads(ndjson_line)
    className = sample["word"]
    pixelArray = sample["drawing"]
    strokeLengths = [len(stroke[0]) for stroke in pixelArray]
    totalPixels = sum(strokeLengths)
    npInk = np.zeros((totalPixels, 3), dtype=np.float32)
    currentT = 0
    for stroke in pixelArray:
        for i in [0,1]:
            npInk[currentT:(currentT + len(stroke[0])), i] = stroke[i]
        currentT += len(stroke[0])
        npInk[currentT - 1, 2] = 1
    
    # Normalizing Size
    lower = np.min(npInk[:,0:2], axis=0)
    upper = np.max(npInk[:,0:2], axis=0)
    scale = upper-lower
    scale[scale == 0] = 1
    npInk[:, 0:2] = (npInk[:,0:2] - lower) / scale

    # Compute Deltas
    npInk[1:, 0:2] -= npInk[0:-1, 0:2]
    npInk = npInk[1:,:]
    return npInk, className



def loadfile(datafile):
  output = []
  with open(datafile, 'r') as f:        
    for line in f: 
      output.append(parse_line(line))
  return output



if __name__ == "__main__":
    clean_data = []
    data = loadfile(r"C:\Users\skinn\OneDrive\Desktop\437_Project\Data_Files\full_simplified_basketball.ndjson")
    #print(data)