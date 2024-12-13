import cv2
import numpy as np
import glob
import os.path as osp


# root directory
_ROOT_DIR = r'./calibrate'

# size
_H = 5
_W = 7


def cal(h, w, image_path_list):
    # h, w: Checkerboard corner specifications
    # Set subpixel corner search parameters: max iterations 30, max error tolerance 0.001.
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # Obtain the world coordinate positions of the calibration board corners. 
    # The world coordinate system is set on the calibration board, with all Z coordinates set to 0, 
    # so only the X and Y coordinates need to be assigned.
    '''
    objp:
    [[0. 0. 0.]
    [1. 0. 0.]
    [2. 0. 0.]
    ...
    [4. 4. 0.]
    [5. 4. 0.]
    [6. 4. 0.]]
    '''
    objp = np.zeros((h * w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:h, 0:w].T.reshape(-1, 2)

    obj_points = []
    img_points = []


    for image_path in image_path_list:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (h, w), None)
        print(ret)

        if ret:
            obj_points.append(objp)

            # Specifically used to obtain the precise positions of the inner corner points on the chessboard image, 
            # i.e., finding subpixel corner points based on the original corners.
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)

            # cv2.drawChessboardCorners(img, (h, w), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(1000)

    print(len(img_points))
    # cv2.destroyAllWindows()

    # calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    np.set_printoptions(suppress=True)
    print("ret:", ret)
    print("mtx:\n", mtx.reshape(1, -1).tolist())  # Intrinsic parameter matrix
    print("dist:\n", dist.tolist())  # Distortion coefficients,   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)


if __name__ == '__main__':
    cal(
        _H,
        _W,
        glob.glob(
            osp.join(_ROOT_DIR, '*.jpg')
        )
    )

