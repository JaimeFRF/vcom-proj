import cv2
import utils.utils as Utils
import math
import numpy as np

## Module for functions related to rotating the board to the correct position, based on the little horse in the corner

# apply sift, to detect feature keypoints and compute their descriptors
def sift(data, keypointsFieldTitle="keypoints", descriptorsFieldTitle="descriptors"):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(data["image"], None)
    data["metadata"][keypointsFieldTitle] = keypoints
    data["metadata"][descriptorsFieldTitle] = descriptors
    return data

# aplly flann_matcher to match feature descriptors between two images
def flann_matcher(data, descriptors1, descriptors2="descriptors", matchesTitle="matches"):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(data["metadata"][descriptors1], data["metadata"][descriptors2], k=2)

    # Store all matches (taking the best match from each pair)
    good = [m for m, _ in matches]

    data["metadata"][matchesTitle] = good
    return data

#apply RANSAC to estimate homography from matches
def find_homography_from_matches(data, keypoints1, keypoints2="keypoints", matchesTitle="matches", homographyTitle_result="homography"):
    
    matches = data["metadata"].get(matchesTitle,None)
    kp1 = data["metadata"].get(keypoints1, None)
    kp2 = data["metadata"].get(keypoints2, None)

    if (matches is None or kp1 is None or kp2 is None):
        raise ValueError("Matches and keypoints must be defined previously in pipeline, to find homography")
    
    if len(data["metadata"][matchesTitle]) < 4:
        raise ValueError("Must have at least 4 matches to find homography")

    # Prepare points for homography
    query_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)  # img1
    train_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)  # img2

    # Find homography from query (img1) to train (img2)
    M, _ = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 2.0)

    data["metadata"][homographyTitle_result] = M
    return data


# given a homography matrix, extract the rotation angle, fixed to 0, 90, 180 or 270 degrees
def extract_rotation_from_homography(data, homographyTitle="homography"):
    M = data["metadata"].get(homographyTitle, None)
    if (M is None):
        raise ValueError("Homography matrix must be defined previously in pipeline, to extract rotation angle")
    
    # Extract rotation angle from homography matrix
    a = M[0, 0]
    b = M[0, 1]
    c = M[1, 0]
    d = M[1, 1]
    
    # Calculate rotation angle
    angle_rad = np.arctan2(c, a)
    raw_angle_deg = np.degrees(angle_rad)
    
    # Normalize angle to 0-360 range
    raw_angle_deg = raw_angle_deg % 360
    if raw_angle_deg < 0:
        raw_angle_deg += 360
        
    # Quantize to 0, 90, 180, or 270 degrees
    quantized_angle = round(raw_angle_deg / 90) * 90
    if quantized_angle == 360:
        quantized_angle = 0
        
    print(f"Raw rotation angle: {raw_angle_deg:.2f} degrees")
    print(f"Quantized rotation angle: {quantized_angle:.0f} degrees")
    return quantized_angle

def rotate_img_from_homography(data, homographyTitle="homography"):
    M = data["metadata"].get(homographyTitle, None)
    if (M is None):
        raise ValueError("Homography matrix must be defined previously in pipeline, to extract rotation angle")
    
    # extract rotation angle from homography matrix M
    rotation_angle = extract_rotation_from_homography(data, homographyTitle)

    # create a pure rotation matrix centered on the image
    h, w = data["image"].shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # apply the rotation to the train image
    data["image"] = cv2.warpAffine(data["image"], rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

    return data
