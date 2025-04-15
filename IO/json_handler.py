import json, cv2
import re

#Stores an array of tuples with the image name and the object image
def read_images():
    data = json.load(open("input.json"))
    images_dict = [{
        "name": i, 
        "image": cv2.imread("{name}".format(name=i)), # this one is to apply the operations 
        "orig_img": cv2.imread("{name}".format(name=i)), # this one is to keep the original image
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

def write_results(data):

    write_data = []
    for i in data:
        matrix = i["metadata"]["chessboard_matrix"]
        matrix[matrix == 2] = 1  
    
        write_data.append({
            "image" : "{name}".format(name=i["name"]),
            "num_pieces" : i["metadata"]["total_black"] + i["metadata"]["total_white"],
            "board": matrix.tolist(),
            "detected_pieces": format_bboxes(i["metadata"]["refined_bounding_boxes"]),
        })

    json_string = json.dumps(write_data, indent=4)

    def collapse_board_arrays(match):
        # extract content inside the brackets
        content = match.group(1)
        # remove newlines and extra spaces
        content = re.sub(r'\s*\n\s*', '', content)
        # spacing after commas
        content = re.sub(r',\s*', ', ', content.strip())
        return f"[{content}]"
    
    # collpase 2d board rows into one line, like in the example output file
    json_string = re.sub(r'\[\s*(\d+(?:,\s*\d+)*)\s*\]', collapse_board_arrays, json_string)
    
    with open("output.json", "w") as f:
        f.write(json_string)

    return 0

#format array of bounding boxes to a list of dictionaries
def format_bboxes(array):
    result_list = []
    for bbox in array:
        xmin, ymin, xmax, ymax = bbox
        bbox_dict = {
            "xmin": int(xmin),
            "ymin": int(ymin),
            "xmax": int(xmax),
            "ymax": int(ymax)
        }

        result_list.append(bbox_dict)
    return result_list