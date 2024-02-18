
import numpy as np
import cv2
import math


def margin_points(point, r, n=20):
    return [(point[0] + math.cos(2*math.pi/n*x)*r, point[1] + math.sin(2*math.pi/n*x)*r) for x in range(0,n+1)]

def margin_hull(in_points, margin, n):
    new_points = []
    for each_point in in_points:
        new_points += margin_points(each_point, margin, n)
    new_points = np.array(new_points).astype(np.int32)
    new_points = cv2.convexHull(new_points)
    return new_points

def mask(image, landmarks):
    landmarks = landmarks.astype(np.int32)
    mask = np.zeros_like(image)
    size = image.shape[0]

    # mouth_indices = list(range(48, 68))
    # landmarks = landmarks[mouth_indices]

    left_eye_idx = list(range(36, 41))
    left_eye_lm = landmarks[left_eye_idx]
    left_eye_lm = margin_hull(left_eye_lm, 10, 20)
    cv2.fillPoly(mask, [left_eye_lm], (255, 255, 255))

    right_eye_idx = list(range(42, 47))
    right_eye_lm = landmarks[right_eye_idx]
    right_eye_lm = margin_hull(right_eye_lm, 10, 20)
    cv2.fillPoly(mask, [right_eye_lm], (255, 255, 255))

    mask = mask / 255
    mask = cv2.resize(mask, (512, 512))    
    mask = cv2.GaussianBlur(mask, (27, 27), 51)
    mask = cv2.resize(mask, (size, size))

    result = image * (1 - mask)
    result = result.astype(int)
    return result, mask

# visualize
def draw_landmarks(image, landmarks):
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

def draw_lines(image, points, color):
    for (start, end) in points:
        cv2.line(image, tuple(start), tuple(end), color, 1)

def draw_face_schematic(image, landmarks):

    # Draw landmarks on the image
    draw_landmarks(image, landmarks)

    # Connect specific landmarks to represent features like eyes and lips
    eye_lines = [(landmarks[i], landmarks[i + 1]) for i in range(36, 41)] + [(landmarks[i], landmarks[i + 1]) for i in range(42, 47)]
    # lip_lines = [(landmarks[i], landmarks[i + 1]) for i in range(48, 59)] + [(landmarks[48], landmarks[59]), (landmarks[60], landmarks[67])]

    # Draw lines for eyes and lips
    draw_lines(image, eye_lines, (255, 0, 0))  # Blue color for eyes
    # draw_lines(image, lip_lines, (0, 0, 255))  # Red color for lips
