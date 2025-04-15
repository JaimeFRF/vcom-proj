import cv2
import numpy as np
from scipy.ndimage import maximum_filter, label, find_objects


def draw_hough_lines_on_warped_image(data):
    lines = data["metadata"]["lines"]

    height, width = data["image"].shape[:2]
    line_map = np.zeros((height, width), dtype=np.uint8)
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_map, (x1, y1), (x2, y2), (255, 255, 0), 1)
    
    data["metadata"]["line_map"] = line_map 
    return data

def calculate_corners(data, boardContourFieldName="board_contour_warped", boardCornersFieldName="board_corners"):
    density_map = cv2.GaussianBlur(data["metadata"]["line_map"], (45, 45), 0)
    
    height, width = data["metadata"]["line_map"].shape
    center = np.array([width // 2, height // 2])

    neighborhood_size = 200
    filtered = maximum_filter(density_map, neighborhood_size)
    peaks = (density_map == filtered)

    labeled, _ = label(peaks)
    slices = find_objects(labeled)

    dense_points = []
    for dyx in slices:
        y_center = (dyx[0].start + dyx[0].stop) // 2
        x_center = (dyx[1].start + dyx[1].stop) // 2
        dense_points.append((x_center, y_center))


    first_quadrant = [point for point in dense_points if point[0] <= width // 2 and point[1] <= height // 2]
    second_quadrant = [point for point in dense_points if point[0] >= width // 2 and point[1] <= height // 2]
    third_quadrant = [point for point in dense_points if point[0] <= width // 2 and point[1] >= height // 2]
    fourth_quadrant = [point for point in dense_points if point[0] >= width // 2 and point[1] >= height // 2]
    
    dense_points = sorted(dense_points, key=lambda pt: density_map[pt[1], pt[0]], reverse=True)
    points = [first_quadrant, second_quadrant, fourth_quadrant, third_quadrant]
    scored_quadrants = []
    
    for quadrant in points:
        scored_quadrant = []
        for point in quadrant:
            pt_arr = np.array(point)
            dist_from_center = np.linalg.norm(pt_arr - center)
            density = density_map[point[1], point[0]]
            score = dist_from_center * 0.4 + density * 0.6
            scored_quadrant.append((point[0], point[1], score))

        tlist = sorted(scored_quadrant, key=lambda t: t[2], reverse=True)
        scored_quadrants.append(tlist)

    top_4 = [(quadrant[0][0], quadrant[0][1]) for quadrant in scored_quadrants]
    transformed_top_4 = np.array(top_4, dtype=np.float32)
    transformed_2 = [transformed_top_4.astype(np.int32).reshape(-1, 1, 2)] 

    output = data["metadata"]["line_map"].copy()    
    for point in top_4:
        cv2.circle(output, point, 10, 255, -1) 

    [cv2.circle(output, pt, 10, 200, -1) for pt in dense_points if pt not in top_4]
    
    data["metadata"]["line_map"] = output
    data["metadata"][boardCornersFieldName] = np.array(top_4, dtype=np.float32)
    data["metadata"][boardContourFieldName] = transformed_2

    return data