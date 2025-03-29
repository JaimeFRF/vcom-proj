import os
import json

Direc = os.getcwd()+ "/images"
files = os.listdir(Direc)

files = [f for f in files if os.path.isfile(Direc+'/'+f)]

solutions = []

for file in files:
    entry = {
        "name" : file,
        "pieces" : [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]
    }
    solutions.append(entry)

with open("solutions.json", "w") as outfile:
    json.dump(solutions, outfile, indent=4, separators=(",", ": "))