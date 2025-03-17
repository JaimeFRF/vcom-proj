import json, cv2

#Stores an array of tuples with the image name and the object image
def read_images():
    data = json.load(open("input.json"))
    images = [(i, cv2.imread("images/{name}".format(name=i))) for i in data["images"]]
    return images

#TODO
def write_results():
    return 0