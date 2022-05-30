from matplotlib import pyplot as plt
import numpy as np
import os
import cv2  # pip install opencv-python


def make_dir(dir_path, reset_old=False):
    try:
        if reset_old or not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except OSError:
        print('Error: Failed to create dir ' + dir_path)


def to_dir_name(name):
    if not name.endswith('/'):
        name += '/'

    return name


def check_if_exists(path):
    if not os.path.exists(path):
        print(f"Error: {path} does not exist.")


def get_file_paths_in_dir(dir_path):
    dir_path = to_dir_name(dir_path)

    files = []
    for name in os.listdir(dir_path):
        files.append(dir_path + name)

    return files


def save_video_frames(video_capture, frame_output_dir):
    _current_frame = 0
    while True:
        _frame_was_found, _frame = video_capture.read()
        if _frame_was_found:
            name = f"{frame_output_dir}/{str(_current_frame)}.jpg"
            cv2.imwrite(name, _frame)
            _current_frame += 1
        else:
            break

    print(f'    Captured {_current_frame} frames for ' + frame_output_dir)

    video_capture.release()
    cv2.destroyAllWindows()


def show_image(_image):
    if type(_image) is str:
        _image = cv2.imread(_image)

    # _image = cv2.cvtColor(_image, cv2.COLOR_BGRA2RGB)
    plt.imshow(_image)
    plt.title('my picture')
    plt.show()


def show_image_in_window(_img):
    cv2.imshow('image window', _img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def split_image_in_fragments_strides(_img, fragment_height, fragment_width):

    image_height, image_width, channel = _img.shape
    byte_size = _img.dtype.alignment

    shape = (
        channel,
        int(image_height / fragment_height),
        int(image_width / fragment_width),
        fragment_height,
        fragment_width
    )

    strides = (
        image_width * fragment_height * fragment_width * byte_size,
        image_width * fragment_height * byte_size,
        fragment_width * byte_size,
        image_width * byte_size,
        byte_size
    )

    return np.lib.stride_tricks.as_strided(_img, shape=shape, strides=strides)


def split_image_in_fragments(img, height, width):
    return [img[x:x + height, y:y + width] for x in range(0, img.shape[0], height) for y in range(0, img.shape[1], width)]


def get_simple_file_name(file_path):
    return file_path.split('/')[-1].split('.')[0]
