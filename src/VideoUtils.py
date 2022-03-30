from matplotlib import pyplot as plt
import numpy as np
import os
import cv2  # pip install opencv-python


def make_dir(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except OSError:
        print('Error: Failed to create dir ' + dir_path)


def check_if_exists(path):
    if not os.path.exists(path):
        print(f"Error: {path} does not exist.")


def get_file_paths_in_dir(dir_path, silent=False):
    if not dir_path.endswith('/'):
        dir_path += '/'

    files = []
    for name in os.listdir(dir_path):
        files.append(dir_path + name)

    if not silent:
        print(f"{len(files)} objects found in {dir_path}.")

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


def extract_frames_from_video(video_paths, output_dir):
    print("Start capturing frames for the trainings videos")
    for video_number in range(len(video_paths)):
        video_path = video_paths[video_number]
        video = cv2.VideoCapture(video_path)

        frame_output_dir = output_dir + video_path.split('.')[0].split('/')[-1]
        make_dir(frame_output_dir)

        save_video_frames(video, frame_output_dir)

    print("Successfully captured frames for the trainings videos")


def get_image_sizes(frames_dir):
    frame_dirs = os.listdir(frames_dir)
    frame_size_dict = {}
    for frame_dir in frame_dirs:
        current_frame_dir = frames_dir + frame_dir
        frame_names = os.listdir(current_frame_dir)
        frame = cv2.imread(current_frame_dir + '/' + frame_names[0])
        h, w, c = frame.shape
        # frame_size_set

        if not frame_size_dict.__contains__((h, w)):
            frame_size_dict[(h, w)] = 1
        else:
            frame_size_dict[(h, w)] += 1

    for frame_size, count in frame_size_dict.items():
        print(f'{count} times {frame_size}')


def show_image(_image):
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


def read_video_annotation(annotation_file, variables):
    with open(annotation_file, 'r') as file:

        lines = file.readlines()
        frame_annotation = np.full((len(lines), variables), fill_value=-1, dtype=np.int16)
        for line_nr in range(len(lines)):

            line = lines[line_nr].replace('\n', '')
            line_sections = line.split(' ')
            for section in range(len(line_sections[1:variables])):
                frame_annotation[line_nr][section] = line_sections[section + 1]

        return frame_annotation


def get_simple_file_name(file_path):
    return file_path.split('/')[-1].split('.')[0]


def generate_dataset(dir_of_frames, annotations_map, frag_h, frag_w, limit_used_frames=None, ):

    dataset = []
    for frame_dir in get_file_paths_in_dir(dir_of_frames, silent=True)[:limit_used_frames]:

        video_name = get_simple_file_name(frame_dir)
        print(f"Extract data for {video_name}")
        for frame_path in get_file_paths_in_dir(frame_dir, silent=True):

            frame_nr = int(get_simple_file_name(frame_path))
            frame = cv2.imread(frame_path)
            fragments = split_image_in_fragments(frame, frag_h, frag_w)

            frame_annotation = annotations_map[video_name][frame_nr]
            frame_contains_drone = frame_annotation[0]
            for fragment_nr in range(len(fragments)):

                if frame_contains_drone == 0:
                    frag_contains_drone = 0

                else:
                    frame_h, frame_w = frame.shape[:2]
                    pos_w = frame_annotation[1]
                    pos_h = frame_annotation[2]

                    drone_in_frame_nr = pos_w // frag_w + (pos_h // frag_h) * (frag_w // frame_w)
                    frag_contains_drone = fragment_nr.__eq__(drone_in_frame_nr)

                dataset.append([video_name, frame_nr, fragment_nr, frag_contains_drone, fragments[fragment_nr]])

    print(f"return dataset containing {len(dataset)} entries")
    return np.array(dataset, dtype=object)
