import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import math

def detect_edges(frame):
    # convert from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # cv2.imshow("hsv", hsv)

    '''# lift blue from frame
    lower_blue = np.array([30, 40, 40]) # 60, 150 is range of blue, the second and third parameters are not much important.
    upper_blue = np.array([90, 255, 255])'''
    
    # lift white from frame
    lower_white = np.array([0, 0, 255 - 30])
    upper_white = np.array([255, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # cv2.imshow("blue mask", mask)

    # detect edges
    edges = cv2.Canny(mask, 200, 400)  # the second and third are recommented: 200vs400 or 100vs200

    return edges


def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 0.8),
        (width, height * 0.8),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges


def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold,
                                    np.array([]), minLineLength=8, maxLineGap=4)

    return line_segments


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logging.info('No line_segment segments detected')
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary  # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logging.debug('lane lines: %s' % lane_lines)  # [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]

    return lane_lines


def detect_lane(frame):
    edges = detect_edges(frame)
    cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segments)

    return lane_lines

#Vẽ 2 lane left-right
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=5):
    line_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                x_temp = int((6 * x1 + 4 * x2) / 10)
                y_temp = int((6 * y1 + 4 * y2) / 10)

                # vẽ lề
                cv2.line(line_image, (x1, y1), (x_temp, y_temp), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image




def display_steering_angle(frame, lines, line_color=(0, 255, 0), line_width=5):
    line_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    if len(lines) == 2:
        # trường hợp có 2 lề
        x1_left, y1_left, x2_left, y2_left = lines[0][0]
        x1_right, y1_right, x2_right, y2_right = lines[1][0]
    else:
        #trường hợp 1 lề
        if lines[0][0][0] < width / 2:
            x1_left, y1_left, x2_left, y2_left = lines[0][0]
            x1_right, y1_right, x2_right, y2_right = [width + 13, height, width + 15, y2_left]
        if lines[0][0][0] > width / 2:
            x1_right, y1_right, x2_right, y2_right = lines[0][0]
            x1_left, y1_left, x2_left, y2_left = [0 - 15, height, 0 - 13, y2_right]

    # rút ngắn doan thẳng
    x2_left = int((6 * x1_left + 4 * x2_left) / 10)
    y2_left = int((6 * y1_left + 4 * y2_left) / 10)
    x2_right = int((6 * x1_right + 4 * x2_right) / 10)
    y2_right = int((6 * y1_right + 4 * y2_right) / 10)

    # tọa độ angle steering
    x1_mid = int((x1_left + x1_right) / 2)
    #y1_mid = int((x1_left + y1_right) / 2)
    x2_mid = int((x2_left + x2_right) / 2)
    #y2_mid = int((x2_left + y2_right) / 2)

    # vẽ màu
    contours = np.array([[x1_left + 20, y1_left], [x2_left + 13, y2_left], [x2_right - 13, y2_right], [x1_right - 15, y1_right]])
    cv2.fillPoly(frame, pts=[contours], color= line_color)

    # vẽ angle steering
    cv2.line(line_image, (x1_mid, y1_left), (x2_mid, y2_left), (255,0,0), line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


cap = cv2.VideoCapture('C:\\Users\\Admin\\Downloads\\in.mp4')
while(cap.isOpened()):
    #Đọc video
    ret, frame = cap.read()
    lane_lines = detect_lane(frame)
    #frame = display_lines(frame, lane_lines)        #Vẽ 2 lane left-right
    stear = display_steering_angle(frame, lane_lines)
    cv2.imshow('Video', stear)
    cv2.waitKey(10)
    #if cv2.waitKey(1) == ord('q'):
     #   break
cap.release()
cv2.destroyAllWindows()

