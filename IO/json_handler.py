import json, cv2

#Stores an array of tuples with the image name and the object image
def read_images():
    data = json.load(open("input.json"))
    images_dict = [{
        "name": i, 
        "image": cv2.imread("images/{name}".format(name=i)), # this one is to apply the operations 
        "orig_img": cv2.imread("images/{name}".format(name=i)), # this one is to keep the original image
        "debug": [], # store debug images here, to print at the end in a single window
        "metadata": {}
        } 
    for i in data["image_files"]]
    return images_dict

def read_single_image(path):
    orig_img = cv2.imread(path)
    img_dict = [{
        "name": path,
        "orig_img": orig_img,
        "image": orig_img,
        "debug": [],
        "metadata": {}
    }]
    
    return img_dict

def read_results():
    data = json.load(open("tests/solutions.json"))
    solutions_dict = [{
        "name" : i["name"],
        "pieces": i["pieces"],
        "black_pieces" : i["black_pieces"],
        "white_pieces": i["white_pieces"]
    } for i in data]
    return solutions_dict

#TODO
def write_results():
    return 0