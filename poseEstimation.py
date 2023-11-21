import cv2 as cv
import numpy as np
import pickle

# Load the camera calibration parameters from the .pkl files
calibration_data = pickle.load(open("calibration.pkl", "rb"))
cameraMatrix, dist = calibration_data

# Load an image for pose estimation
image_path = 'images/img17.png'  # Replace with the path to your image
img = cv.imread(image_path)

# Perform pose estimation
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Define the chessboard size used for calibration
chessboard_size = (7, 7)

# Find chessboard corners
ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

if ret:
    # Refine corner locations
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, 0.001)
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


    # SolvePnP for pose estimation
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= 20  # Assuming square size in mm

    _, rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners, cameraMatrix, dist)

    # Draw axis on the image
    axis_size = 50
    axis_points = np.float32([[0, 0, 0], [axis_size, 0, 0], [0, axis_size, 0], [0, 0, -axis_size]]).reshape(-1, 3)
    imgpts, _ = cv.projectPoints(axis_points, rvecs, tvecs, cameraMatrix, dist)

    img = cv.drawFrameAxes(img, cameraMatrix, dist, rvecs, tvecs, axis_size)
    # Display the image with pose information
    cv.imshow('Pose Estimation', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Chessboard corners not found in the image.")
