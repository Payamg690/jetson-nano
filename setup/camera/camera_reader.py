# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import pickle

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline(
    capture_width=3280,
    capture_height=2464,
    display_width=1280,
    display_height=720,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera(cam_mtx=None, cam_dist=None, h=720, w=1280, remap=True):
    if cam_mtx is not None and cam_dist is not None:
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_mtx, cam_dist, (w,h), 0, (w,h))
                if remap:
                    mapx, mapy = cv2.initUndistortRectifyMap(cam_mtx, cam_dist, None, newcameramtx, (w,h), 5)
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        counter = 0
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()

            # undistort
            if cam_mtx is not None and cam_dist is not None:
                if remap:
                    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
                else:            
                    dst = cv2.undistort(img, cam_mtx, cam_dist, None, newcameramtx)
                # crop the image
                x, y, w, h = roi
                img = dst[y:y+h, x:x+w]
        
            cv2.imshow("CSI Camera", img)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
            if keyCode == 99:
                cv2.imwrite("camera/pics/img_{}.png".format(counter), img)
                counter += 1 
                print("camera/pics/img_{}.png camptured.".format(counter))
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    undistort = True
    
    if undistort:
        with open("camera/camera_mtx_dist.p", "rb") as cam_file:
            cam_mtx_dist = pickle.load(cam_file)
        show_camera(cam_mtx_dist['mtx'], cam_mtx_dist['dist'])
    else:
        show_camera()