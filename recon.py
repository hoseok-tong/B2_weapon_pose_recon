import os
import cv2 as cv
import numpy as np
from utils import extract_camera_parameters_xml, read_camera_parameters, load_camera_params_mat

def construct_cam_matrices(camera_params):
    """
    Construct camera matrices from intrinsic and extrinsic parameters.
    
    :param camera_params: Dictionary containing camera parameters 'K', 'R', and 'T'.
    :return: Camera matrices.
    """
    K = camera_params['K']
    R, t = camera_params['R'], camera_params['T']
    num_cams = K.shape[0]
    cam_matrices = np.zeros((num_cams, 4, 3))
    for cidx in range(num_cams):
        extrinsic = np.vstack([R[cidx].T, -t[cidx]])
        cam_matrices[cidx] = extrinsic @ K[cidx].T
    return cam_matrices

def undistort(keypoints, camera_params, idx):
    """
    Undistort 2D keypoints.
    
    :param keypoints: List of (x, y) tuples or numpy array of shape (N, 2) representing the distorted keypoints.
    :param camera_params: Dictionary containing camera parameters 'K' and 'D'.
    :param idx: Camera index.
    :return: Undistorted keypoints as a numpy array of shape (N, 2).
    """
    camera_matrix = camera_params['K'][idx]
    dist_coeffs = camera_params['D'][idx]
    keypoints = np.array(keypoints, dtype=np.float32)
    undistorted = cv.undistortPoints(keypoints, camera_matrix, dist_coeffs, P=camera_matrix)
    
    # Convert from homogeneous coordinates to 2D points
    undistorted = undistorted.squeeze()
    
    return undistorted

class PoseReconstructor:
    def __init__(self, calib_path):
        if isinstance(calib_path, list):  # MVS .txt
            self._load_mvs_calibration(calib_path)
        else:
            file_extension = calib_path.split('.')[-1]
            if file_extension == 'xml':  # Metashape .xml
                self._load_metashape_calibration(calib_path)
            elif file_extension == 'mat':  # Matlab calibration .mat
                self._load_matlab_calibration(calib_path)
        
        os.makedirs('./output/cams', exist_ok=True)
        np.savez('./output/cameras.npz', intrinsics=self.camera_params['K'], extrinsics=self.camera_params['R'])
        self.save_camera_parameters(self.camera_params['K'], self.camera_params['R'])

    def _load_mvs_calibration(self, calib_path):
        """
        Load MVS calibration data.
        
        :param calib_path: List of calibration file paths.
        """
        print(calib_path[0])
        intrinsics, extrinsics = [], []
        for calib in calib_path:
            ii, ee = read_camera_parameters(calib)
            intrinsics.append(ii)
            extrinsics.append(ee)

        intrinsics = np.stack(intrinsics)   # (n, 3, 3)
        extrinsics = np.stack(extrinsics)   # (n, 4, 4)
        extrinsics[:, :3, 3] *= 0.001       # scale translation
        self.camera_params = {
            'K': intrinsics[:, :3, :3],
            'R': extrinsics[:, :3, :3],
            'T': extrinsics[:, :3, 3]
        }

    def _load_metashape_calibration(self, calib_path):
        """
        Load Metashape calibration data.
        
        :param calib_path: Path to the calibration file.
        """
        print(calib_path)
        camera_params = extract_camera_parameters_xml(calib_path)
        intrinsics, extrinsics = [], []
        for idx, cam_param in enumerate(camera_params):
            transform = cam_param['transform_matrix'].astype(np.float32)
            intrinsic = np.array(cam_param['intrinsic_matrix']).astype(np.float32)
            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = transform[:3, :3].T
            extrinsic[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)

        intrinsics = np.stack(intrinsics)   # (n, 3, 3)
        extrinsics = np.stack(extrinsics)   # (n, 4, 4)
        self.camera_params = {
            'K': intrinsics[:, :3, :3],
            'R': extrinsics[:, :3, :3],
            'T': extrinsics[:, :3, 3]
        }

    def _load_matlab_calibration(self, calib_path):
        """
        Load Matlab calibration data.
        
        :param calib_path: Path to the calibration file.
        """
        print(calib_path)
        camera_params = load_camera_params_mat(calib_path)
        intrinsics = camera_params['K']  # (n, 3, 3)
        extrinsics = np.tile(np.eye(4), (len(intrinsics), 1, 1))  # (n, 4, 4)
        extrinsics[:, :3, :3] = camera_params['R'] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        extrinsics[:, :3, 3] = camera_params['T']
        self.camera_params = {
            'K': intrinsics[:, :3, :3],
            'D': camera_params['D'],
            'R': extrinsics[:, :3, :3],
            'T': extrinsics[:, :3, 3]
        }
        self.camera_matrices = construct_cam_matrices(self.camera_params)

    def save_camera_parameters(self, intrinsics, extrinsics):
        """
        Save camera parameters to text files.
        
        :param intrinsics: Intrinsic parameters of cameras.
        :param extrinsics: Extrinsic parameters of cameras.
        """
        for i, (K, E) in enumerate(zip(intrinsics, extrinsics)):
            filename = f"./output/cams/{i:08d}_cam.txt"
            with open(filename, 'w') as f:
                f.write("extrinsic\n")
                for row in E:
                    f.write(' '.join(map(str, row)) + '\n')
                f.write("\nintrinsic\n")
                for row in K:
                    f.write(' '.join(map(str, row)) + '\n')

    def __call__(self, batch_pose2d, cam_indices=None):
        """
        Reconstruct 3D pose from 2D poses across multiple camera views.
        
        :param batch_pose2d: Array of 2D poses with shape (num_cams, num_parts, 3).
        :param cam_indices: Indices of cameras to use for reconstruction.
        :return: 3D pose as a numpy array.
        """
        num_cams, num_parts = batch_pose2d.shape[:2]
        point_tracks = [{'ViewIds': [], 'Points': []} for _ in range(num_parts)]
        
        if cam_indices is None:
            cam_indices = range(num_cams)
        
        for cam_idx in cam_indices:
            pose2d = batch_pose2d[cam_idx].reshape([num_parts, 3])
            pose2d[:, :-1] = undistort(pose2d[:, :-1], self.camera_params, cam_idx)
            
            for part_idx in range(num_parts):
                if pose2d[part_idx, 2] < 0.01:
                    continue
                point_tracks[part_idx]['ViewIds'].append(cam_idx)
                point_tracks[part_idx]['Points'].append(pose2d[part_idx, :2])
        
        for part_idx in range(num_parts):
            point_tracks[part_idx]['Points'] = np.array(point_tracks[part_idx]['Points'])
        
        num_tracks = len(point_tracks)
        pose3d = np.zeros((num_tracks, 3), dtype=np.float32)
        
        for track_idx in range(num_tracks):
            track = point_tracks[track_idx]
            view_ids = track['ViewIds']
            points = np.transpose(track['Points'])  # shape: (2, num_views)
            num_views = len(view_ids)
            
            B = np.zeros((num_views * 2, num_views), dtype=np.float32)
            C = np.zeros((num_views, 4), dtype=np.float32)
            P = np.zeros((num_views * 2, 4), dtype=np.float32)
            
            for view_idx in range(num_views):
                if view_ids[view_idx] >= self.camera_matrices.shape[0]:
                    raise Exception(f"vision:absolutePoses:missingViewId {view_ids[view_idx]}")
                
                P_tmp = np.transpose(self.camera_matrices[view_ids[view_idx]])
                idx = 2 * view_idx
                B[idx:idx+2, view_idx] = points[:, view_idx]
                C[view_idx, :] = P_tmp[2, :]
                P[idx:idx+2, :] = P_tmp[0:2, :]
            
            A = np.matmul(B, C) - P
            
            _, _, V = np.linalg.svd(A)
            X = V[-1, :]
            X /= X[-1]
            pose3d[track_idx, :] = X[:3]
        
        return pose3d * -1

