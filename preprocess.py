import cv2
import os
import glob
import numpy as np
import os.path as osp

# configurations
_INPUT_DIR = r'./preprocess/gopro_preprocess/videos/train/'
_OUTPUT_DIR = r'./preprocess/gopro_preprocess/videos/train/processed/'
_IMAGE_SIZE = (1920, 1080)
_CROP_SIZE = (1600, 512)


def _process_intrinsic():
    K = np.array([
        [1009.06701296, 0., 964.43331087],
        [0., 1007.57532723, 544.86182277],
        [0., 0., 1.]
    ], dtype=np.float32)
    dist = np.array([0.03629680, -0.09102069, 0.00239321, 0.00108665, 0.08793252], dtype=np.float32)

    return K, dist


def _extract_frames(fn):
    video = cv2.VideoCapture(fn)
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    save_rate = 0.1

    idx = 0
    frames = []
    while True:
        _, frame = video.read()
        if frame is None:
            break
        if idx % (frame_rate * save_rate) == 0:
            frames.append(frame)
        idx = idx + 1
    video.release()
    return frames


def _crop(image):
    h, w, _ = image.shape
    assert (w, h) == _IMAGE_SIZE

    m_w = round((_IMAGE_SIZE[0] - _CROP_SIZE[0]) / 2.0)
    m_h = round((_IMAGE_SIZE[1] - _CROP_SIZE[1]) / 2.0)

    return image[m_h: -m_h, m_w: -m_w, :]


def _recompute_intrinsic(K):
    fx, cx = K[0, 0], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]

    m_w = (_IMAGE_SIZE[0] - _CROP_SIZE[0]) / 2.0
    m_h = (_IMAGE_SIZE[1] - _CROP_SIZE[1]) / 2.0

    cx -= m_w
    cy -= m_h

    return np.array(
        [
            [fx, 0., cx],
            [0., fy, cy],
            [0., 0., 1.],
        ], dtype=np.float32
    )


def main():
    for item in ('hazy_video', 'clear_video'):  
        K, dist = _process_intrinsic()

        # create directory
        output_dir = osp.join(_OUTPUT_DIR, item)
        os.makedirs(output_dir, exist_ok=True)

        # find videos
        for file in glob.glob(osp.join(_INPUT_DIR, item, '*.mp4')):
            print(f'Now processing {file}...')
            # create directory
            video_dir = osp.join(output_dir, osp.basename(file)[:-4])
            os.makedirs(video_dir)

            frames = _extract_frames(file)
            # print(len(frames))
            for idx, frame in enumerate(frames):
                # undistort
                frame = cv2.undistort(frame, K, dist)
                frame = _crop(frame)
                cv2.imwrite(osp.join(video_dir, f'{idx:05}.jpg'), frame)
    # write intrinsic
    np.save(osp.join(_OUTPUT_DIR, 'intrinsic.npy'), _recompute_intrinsic(K))




if __name__ == '__main__':
    main()
