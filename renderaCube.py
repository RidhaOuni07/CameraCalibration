import cv2 as cv
import numpy as np
import pickle

# Load the camera calibration parameters from the .pkl files
calibration_data = pickle.load(open("calibration.pkl", "rb"))
cameraMatrix, dist = calibration_data

# Load an image for pose estimation
image_path = 'images/img21.png'  # Replace with the path to your image
img = cv.imread(image_path)

# Perform pose estimation
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Define the chessboard size used for calibration
chessboard_size = (7, 7)

# Find chessboard corners
ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

if ret:
    # Refine corner locations
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # SolvePnP for pose estimation
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= 20  # Assuming square size in mm

    _, rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners, cameraMatrix, dist)

    # Define cube vertices in the object coordinate system
    cube_size = 50
    cube_points = np.float32([
        [0, 0, 0],
        [cube_size, 0, 0],
        [cube_size, cube_size, 0],
        [0, cube_size, 0],
        [0, 0, -cube_size],
        [cube_size, 0, -cube_size],
        [cube_size, cube_size, -cube_size],
        [0, cube_size, -cube_size]
    ])

    # Project cube points into the image plane
    cube_img_points, _ = cv.projectPoints(cube_points, rvecs, tvecs, cameraMatrix, dist)

    # Draw cube edges on the image
    img = cv.drawFrameAxes(img, cameraMatrix, dist, rvecs, tvecs, cube_size)
    img = cv.drawContours(img, [cube_img_points.astype(int).reshape(-1, 2)], -1, (0, 255, 0), 3)

    # Display the image with the rendered cube
    cv.imshow('Rendered Cube', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Chessboard corners not found in the image.")
