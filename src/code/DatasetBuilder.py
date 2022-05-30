import src.code.FileUtils as fu
import numpy as np
import cv2

from src.code.Dataset import Dataset


class DatasetBuilder:
    def __init__(self, dataset_dir):
        self.dataset_dir = fu.to_dir_name(dataset_dir)
        self.videos_dir = None
        self.dir_of_frames = None
        self.annotations_map = None
        self.frag_height = None
        self.frag_width = None

    def set_annotations_map(self, annotations_dir):
        """
        Sets the annotations_map in format:

        {video_name: [frame_nr [contains_drone, pos_w, pos_h, obj_w, obj_h]}

        :param annotations_dir: the directory containing the annotation files
        """

        self.annotations_map = {}
        for annotation in fu.get_file_paths_in_dir(annotations_dir):
            annotation_name = annotation.split('/')[-1].split('.', maxsplit=1)[0]
            self.annotations_map[annotation_name] = _read_video_annotation(annotation, 6)

    def set_frag_height(self, height):
        self.frag_height = height

    def set_frag_width(self, width):
        self.frag_width = width

    def set_videos_dir(self, videos_dir=None):
        if videos_dir is None:
            video_dirs = self.dataset_dir + 'frames/'
        else:
            video_dirs = fu.to_dir_name(videos_dir)

        self.videos_dir = video_dirs

    def set_dir_of_frames(self, dir_of_frames=None, reset_old=False):
        if dir_of_frames is None:
            dir_of_frames = self.dataset_dir + 'frames/'
        else:
            dir_of_frames = fu.to_dir_name(dir_of_frames)

        self.dir_of_frames = dir_of_frames
        fu.make_dir(dir_of_frames, reset_old=reset_old)

    def show_test_frame(self, frame_name=None, frame_nr=0):
        if frame_name is None:
            frame_name = fu.get_file_paths_in_dir(self.dir_of_frames)[0]
        fu.show_image(f"{frame_name}/{frame_nr}.jpg")

    def show_test_frags(self, frame_name=None, frag_nr=0):
        """
        Shows fragments of a frame

        Split of the image using vu.split_image_in_fragments results in split from left to right and top to bottom like:


        0 | 1 | 2

        3 | 4 | 5

        6 | 7 | 8

        :param frame_name:
        :param frag_nr:
        """
        if frame_name is None:
            frame_name = fu.get_file_paths_in_dir(self.dir_of_frames)[0]

        frags = fu.split_image_in_fragments(
            cv2.imread(f"{frame_name}/{frag_nr}.jpg"),
            self.frag_height,
            self.frag_width)

        for frag in frags:
            fu.show_image(frag)

    def extract_frames_from_video(self, limit=None):

        video_paths = fu.get_file_paths_in_dir(self.videos_dir)[:limit]
        video_count = len(video_paths)
        print(f"Start capturing frames for the {video_count} trainings videos")
        for video_number in range(video_count):
            video_path = video_paths[video_number]
            video = cv2.VideoCapture(video_path)

            frame_output_dir = self.dir_of_frames + video_path.split('.')[0].split('/')[-1]
            fu.save_video_frames(video, frame_output_dir)

        print("Successfully captured frames for the trainings videos")

    def analyze_frame_sizes(self) -> None:
        frame_size_dict = {}
        for frame_dir in fu.get_file_paths_in_dir(self.dir_of_frames):
            frame = cv2.imread(fu.get_file_paths_in_dir(frame_dir)[0])
            h, w = frame.shape[:2]

            if not frame_size_dict.__contains__((h, w)):
                frame_size_dict[(h, w)] = 1
            else:
                frame_size_dict[(h, w)] += 1

        for frame_size, count in frame_size_dict.items():
            print(f'{count} times {frame_size}')

    def build(self, limit_used_videos=None) -> Dataset:
        dataset_content = []
        for frame_dir in fu.get_file_paths_in_dir(self.dir_of_frames)[:limit_used_videos]:

            video_name = fu.get_simple_file_name(frame_dir)
            print(f"Extract data for {video_name}")
            for frame_path in fu.get_file_paths_in_dir(frame_dir):

                frame_nr = int(fu.get_simple_file_name(frame_path))
                frame = cv2.imread(frame_path)
                if not self._frame_clean_dividable(frame):
                    continue

                fragments = fu.split_image_in_fragments(frame, self.frag_height, self.frag_width)

                frame_annotation = self.annotations_map[video_name][frame_nr]
                frag_containing_drone = self._fragment_containing_drone(frame, frame_annotation)
                for fragment_nr in range(len(fragments)):
                    dataset_content.append(np.array([
                        video_name,
                        frame_nr,
                        fragment_nr,
                        np.array(1 if frag_containing_drone.__eq__(fragment_nr) else 0),
                        np.array(fragments[fragment_nr], dtype=float)], dtype=object))

        print(f"return dataset containing {len(dataset_content)} entries")
        return Dataset(np.array(dataset_content, dtype=object))

    def _fragment_containing_drone(self, frame, frame_annotation):

        frame_contains_drone = frame_annotation[0]
        if frame_contains_drone == 0:
            return -1

        frame_h, frame_w = frame.shape[:2]
        pos_w = frame_annotation[1]
        pos_h = frame_annotation[2]

        return pos_w // self.frag_width + (pos_h // self.frag_height) * (self.frag_width // frame_w)

    def _frame_clean_dividable(self, frame):
        h, w = frame.shape[:2]
        return h % self.frag_height == 0 and w % self.frag_width == 0


def _read_video_annotation(annotation_file, variables):
    with open(annotation_file, 'r') as file:

        lines = file.readlines()
        frame_annotation = np.full((len(lines), variables), fill_value=-1, dtype=np.int16)
        for line_nr in range(len(lines)):

            line = lines[line_nr].replace('\n', '')
            line_sections = line.split(' ')
            for section in range(len(line_sections[1:variables])):
                frame_annotation[line_nr][section] = line_sections[section + 1]

        return frame_annotation
