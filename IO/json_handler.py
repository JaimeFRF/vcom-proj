import json, cv2

#Stores an array of tuples with the image name and the object image
def read_images():
    data = json.load(open("input.json"))
    images_dict = [{
        "name": i, 
        "orig_img": cv2.imread("images/{name}".format(name=i)), # this one is to keep the original image
        "image": cv2.imread("images/{name}".format(name=i)), # this one is to apply the operations 
        "metadata": {}
        } 
    for i in data["images"]]
    return images_dict

#TODO
def write_results():
    return 0