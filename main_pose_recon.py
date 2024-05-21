import os
import os.path as osp
import sys

import natsort
import numpy as np
from PyQt5 import QtWidgets, QtCore
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from filter import BatchOneEuroFilter
from recon import PoseReconstructor
from renderer import EgocentricPlotter

# Function to read a CSV file and parse it into a dictionary
def read_csv(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    arcuo_dict = {}
    for line in lines:
        marker_id, coords = line.strip().split(':')  # Remove newline characters
        coords = [int(coord) for coord in coords.split(',')]
        arcuo_dict[int(marker_id)] = coords
    return arcuo_dict

# Function to get batch file paths from a sequence directory
def get_batch_paths(seq_dir):
    file_names = natsort.natsorted(os.listdir(seq_dir))
    paths_dict = {}
    for file_name in file_names:
        cam_idx = int(file_name.split('_')[0])
        if cam_idx not in paths_dict:
            paths_dict[cam_idx] = []
        file_path = os.path.join(seq_dir, file_name)
        paths_dict[cam_idx].append(file_path)
    
    batch_file_paths = []
    for frame_idx in range(len(paths_dict[0])):
        batch_file_path = [paths_dict[cam_idx][frame_idx] for cam_idx in range(len(paths_dict))]
        batch_file_paths.append(batch_file_path)
    return batch_file_paths

# Function to load batch marker data
def load_batch_marker(marker_dir):
    detectedTagsAll = np.load(marker_dir, allow_pickle=True)  # Load detected tags
    num_cams = len(detectedTagsAll)
    seq_len = len(detectedTagsAll[0])
    tag_num = 5
    
    # Initialize the array for cube face centers
    cube_face_centers = np.zeros((seq_len, num_cams, tag_num * 4, 3))  # Shape: (frame, cam, points, xyz)

    for cam_idx, detectedTags_cam in enumerate(detectedTagsAll):
        for seq_idx, tags in enumerate(detectedTags_cam):
            for tag_idx, tag_data in tags.items():
                if tag_idx not in [0, 1, 2, 3, 4]:
                    continue
                # Reshape tag data and store in cube_face_centers
                cube_face_center = tag_data[:8].reshape(-1, 2)
                cube_face_centers[seq_idx, cam_idx, tag_idx * 4:tag_idx * 4 + 4, :-1] = cube_face_center
                cube_face_centers[seq_idx, cam_idx, tag_idx * 4:tag_idx * 4 + 4, -1] = 1.0

    return cube_face_centers  # Return the processed data

# Main visualizer class
class Visualizer(QtWidgets.QWidget):
    def __init__(self, cfg_path, calib_path, data_dir, parent=None):
        super().__init__(parent)
        self.timer = QtCore.QTimer()
        self.timer.setInterval(33)  # Set timer interval for 30 FPS
        self.timer.timeout.connect(self.render)
        self.counter = 0
        self.play = False
        self.init_ui(cfg_path)
        self.load_data(calib_path, data_dir)
    
    # Initialize the UI components
    def init_ui(self, cfg_path):
        renderer_before = QVTKRenderWindowInteractor(self)
        self.plotter_before = EgocentricPlotter(qt_widget=renderer_before)
        self.plotter_before.show()
        self.plotter_before.init_board()
        self.plotter_before.add_board()

        self.play_button = QtWidgets.QPushButton('Start')
        self.play_button.clicked.connect(self.on_play)
        
        layout = QtWidgets.QGridLayout()
        layout.addWidget(renderer_before, 0, 0, 1, 1)
        layout.addWidget(self.play_button, 1, 0, 1, 2)
        self.setLayout(layout)
    
    # Load data including calibration and marker information
    def load_data(self, calib_path, data_dir):
        # Initialize the pose reconstructor
        self.reconstructor = PoseReconstructor(calib_path)
        camera_params = self.reconstructor.camera_params
        self.plotter_before.load_cameras(camera_params)
        self.plotter_before.add_cameras()

        # Load marker data
        marker_dir = 'D:/workspace/lib-dt-apriltags/apriltag_100mm_detectedTagsAll.npy'
        marker_list = load_batch_marker(marker_dir)
        self.seq_len = len(marker_list)
        self.marker_list = marker_list[:self.seq_len]  # Truncate or use full sequence
        self.marker3d_list = []
    
    # Toggle play and stop functionality
    def on_play(self):
        if not self.play:
            self.play = True
            self.timer.start()
            self.play_button.setText('Stop')
        else:
            self.play = False
            self.timer.stop()
            self.play_button.setText('Start')
    
    # Render the next frame
    def render(self):
        if self.counter >= self.seq_len:
            self.counter = 0
            self.marker3d_list = []

        marker2d = self.marker_list[self.counter]  # Get 2D markers for the current frame
        marker3d = self.reconstructor(marker2d)    # Reconstruct 3D markers
        self.marker3d_list.append(marker3d)
        self.plotter_before.update(marker3d)

        self.counter += 1

# Main function to start the application
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('fusion')
    
    cfg_path = 'configs/coco_17.json'
    calib_path = 'D:/DATA/weapon_pose/initial_camera_params_2024-03-28_16_35_39.mat'
    data_dir = 'D:/DATA/weapon_pose/SEQUENCE-2024-03-27-15-24-12'
    
    window = Visualizer(cfg_path, calib_path, data_dir)
    window.setWindowTitle('Visualizer')
    window.setGeometry(100, 200, 600, 800)
    window.show()
    
    sys.exit(app.exec_())
