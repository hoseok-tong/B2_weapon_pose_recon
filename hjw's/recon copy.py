import cv2 as cv
import numpy as np
from utils import load_camera_params


def construct_cam_matrices(camera_params):
    K = camera_params['K']
    R, t = camera_params['R'], camera_params['T']
    num_cams = K.shape[0]
    cam_matrices = np.zeros((num_cams, 4, 3))
    for cidx in range(num_cams):
        extrinsic = np.vstack([R[cidx].T, -t[cidx]])
        cam_matrices[cidx] = extrinsic @ K[cidx].T
    return cam_matrices


def reprojection(pose3ds, camera_matrices):
    # S: sequence length, J: number of joints, N: number of cameras
    # pose3ds shape: (S, J, 3), camera_matrices shape: (N, 4, 3)
    pose3dhs = np.concatenate([pose3ds * -1, np.ones_like(pose3ds[:, :, [0]])], axis=-1) # shape: (S, J, 3) -> (S, J, 4)
    pose2dhs = np.einsum('ijk,lkm->iljm', pose3dhs, camera_matrices, optimize=True) # shape: (S, J, 3) x (N, 4, 3) -> (S, N, J, 3)
    pose2ds = pose2dhs[..., :2] / pose2dhs[..., 2:] # shape: (S, N, J, 3) -> (S, N, J, 2)
    return pose2ds


def undistort(keypoints, camera_params, idx):
    """
    Undistort 2D keypoints.
    
    :param keypoints: List of (x, y) tuples or numpy array of shape (N, 2) representing the distorted keypoints.
    :param camera_matrix: 3x3 numpy array representing the camera matrix.
    :param dist_coeffs: 1x5 numpy array representing the distortion coefficients [k1, k2, p1, p2, k3].
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
        self.camera_params = load_camera_params(calib_path)
        self.camera_matrices = construct_cam_matrices(self.camera_params)
    
    def __call__(self, batch_pose2d, cam_indices=None):
        # kpts: C X N X 3
        num_cams, num_parts = batch_pose2d.shape[:2]
        point_tracks = [{'ViewIds': [], 'Points': []} for _ in range(num_parts)]
        if cam_indices is None:
            cam_indices = range(num_cams)
        for idx2 in cam_indices:
            if idx2 in [6, 7, 8]:
                continue
            pose2d = batch_pose2d[idx2].copy()
            pose2d = pose2d.reshape([num_parts, 3])
            # Undistort
            pose2d[:, :-1] = undistort(pose2d[:, :-1], self.camera_params, idx2)
            for idx3 in range(num_parts):
                if pose2d[idx3, 2] < 0.1:
                    continue
                point_tracks[idx3]['ViewIds'].append(idx2)
                point_tracks[idx3]['Points'].append(pose2d[idx3, 0:2])
        for idx3 in range(num_parts):
            point_tracks[idx3]['Points'] = np.array(point_tracks[idx3]['Points'])
        numTracks = len(point_tracks)
        pose3d = np.zeros((numTracks, 3), dtype=np.float32)
        # can use parallel processing
        for idx4 in range(numTracks):
            track = point_tracks[idx4]
            viewIds = track['ViewIds']
            points = np.transpose(track['Points'])  # shape: (2, N)
            numViews = len(viewIds)
            B = np.zeros((numViews * 2, numViews), dtype=np.float32)
            C = np.zeros((numViews, 4), dtype=np.float32)
            P = np.zeros((numViews * 2, 4), dtype=np.float32)
            for idx5 in range(numViews):
                if not viewIds[idx5] < self.camera_matrices.shape[0]:
                    raise Exception("vision:absolutePoses:missingViewId" + viewIds[idx5])
                Ptmp = np.transpose(self.camera_matrices[viewIds[idx5]])
                idx = 2 * idx5
                B[idx:idx+2, idx5] = points[:, idx5].reshape(-1, )
                C[idx5, :] = np.transpose(Ptmp[2, :].reshape(-1, 1))
                P[idx:idx+2, :] = Ptmp[0:2, :]
            A = np.matmul(B, C) - P
            A1 = np.zeros((numViews * 2, 4), dtype=np.float32)
            for idx5 in range(numViews):
                if not viewIds[idx5] < self.camera_matrices.shape[0]:
                    raise Exception("vision:absolutePoses:missingViewId" + viewIds[idx5])
                P = np.transpose(self.camera_matrices[viewIds[idx5]])
                idx = 2 * idx5
                A1[idx:idx+2, :] = np.matmul(points[:, idx5].reshape(-1, 1),
                                            np.transpose(P[2, :].reshape(-1, 1))) - P[0:2, :]
            _, _, V = np.linalg.svd(A)
            V = np.transpose(V)
            X = V[:, -1]
            X = X/X[-1]
            pose3d[idx4, :] = np.transpose(X[0:3])
        return pose3d * -1


def check_reprojection():
    import cv2 as cv
    calib_path = 'data/camera_parameters.mat'
    # camera_params = load_camera_params(calib_path)
    camera_matrices = construct_cam_matrices((calib_path))
    
    data_path = 'data/pose3d.npz'
    pose3d_dict = np.load(data_path)
    pose3d_svd = pose3d_dict['pose3d']
    pose3d_optim = pose3d_dict['pose3d_optim'] * -1
    pose2ds = reprojection(pose3d_optim, camera_matrices)
    num_cam, num_joints = pose2ds.shape[1:-1]
    cv.namedWindow('check reprojection', cv.WINDOW_NORMAL)
    for pose2d in pose2ds:
        for cam_idx in range(num_cam):
            image = np.zeros((1920, 1080, 3), dtype=np.uint8)
            for joint_idx in range(num_joints):
                x, y = pose2d[cam_idx, joint_idx].astype(int)
                cv.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv.imshow('check reprojection', image)
            cv.waitKey(0)
    print()


if __name__ == '__main__':
    check_reprojection()
