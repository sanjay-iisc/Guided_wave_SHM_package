import numpy as np
import os

def CivaData(path, FileName):
    Data = []
    min = []
    FileName = os.path.join(path, FileName)
    with open(FileName, "r") as f:
        for count, line in enumerate(f):
            if count == 0:
                l = line.split(" ")
                l = l[3].split("\n")
                Minfreq = float(l[0])
                # print(line.split(';'))
                # print(count)
            elif count == 1:
                l = line.split(" ")

                l = l[3].split("\n")
                Maxfreq = float(l[0])
                
            elif count == 2:
                l = line.split(" ")

                l = l[4].split("\n")
                NumberOfPoints = int(l[0])

            if count >= 5:
                k = line.split(";")
                k = [w.replace("?", "0") for w in k]
                Data.append(k)
                # print("Line {}: {}".format(count, k))
                a = np.array(Data, dtype=float)
        return a, np.linspace(Minfreq, Maxfreq, NumberOfPoints, endpoint=False)