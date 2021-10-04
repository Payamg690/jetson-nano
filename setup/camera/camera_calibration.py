import cv2
import numpy as np
import glob
import pickle

show_mode = False
store_camera_mtx = True

img_size = (1280, 720)

# Number of object points
num_intersections_in_x = 9
num_intersections_in_y = 7

# Size of square in meters
square_size = 0.02

# Arrays to store 3D points and 2D image points
obj_points = []
img_points = []

# Prepare expected object 3D object points (0,0,0), (1,0,0) ...
object_points = np.zeros((9*7,3), np.float32)
object_points[:,:2] = np.mgrid[0:9, 0:7].T.reshape(-1,2)
object_points = object_points*square_size

fnames = glob.glob('camera/pics/'+'*.'+'png')

for f in fnames:
    img = cv2.imread(f)

    # Find chess board corners
    gray_scale = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    ret, corners = cv2.findChessboardCorners(gray_scale, (num_intersections_in_x, num_intersections_in_y), None)
    if ret:
        obj_points.append(object_points)
        img_points.append(corners)
        if show_mode:
            # Draw the corners
            drawn_img = cv2.drawChessboardCorners(img, (9,7), corners, ret)
            cv2.imshow("main", drawn_img)
            cv2.waitKey(0)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)

# re-projection error
mean_error = 0
for i in range(len(obj_points)):
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(obj_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(obj_points)) )

if store_camera_mtx:
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("camera/camera_mtx_dist.p", "+wb"))