import cv2 as cv
import numpy as np
import scipy.io as spio
import xml.etree.ElementTree as ET


def extract_camera_parameters_xml(xml_file_path):
    """
    Extracts camera parameters and calibration data from the given XML file.

    Parameters:
    xml_file_path (str): Path to the XML file.

    Returns:
    list: A list of dictionaries containing camera parameters and calibration data.
    """
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Namespace handling (if any namespaces are defined in the XML)
    namespaces = {'default': 'http://www.w3.org/2001/XMLSchema-instance'}

    # Find all camera elements
    cameras = root.findall('.//camera', namespaces)

    # Find all sensor elements
    sensors = root.findall('.//sensor', namespaces)

    # Create a dictionary to map sensor IDs to their calibration data
    sensor_calibration_data = {}

    for sensor in sensors:
        sensor_id = sensor.get('id')
        calibration = sensor.find('calibration')
        if calibration is not None:
            intrinsic_params = {
                'f': float(calibration.find('f').text),
                'cx': float(calibration.find('cx').text),
                'cy': float(calibration.find('cy').text)
            }
            intrinsic_matrix = np.array([
                [intrinsic_params['f'], 0, intrinsic_params['cx']],
                [0, intrinsic_params['f'], intrinsic_params['cy']],
                [0, 0, 1]
            ])
            sensor_calibration_data[sensor_id] = intrinsic_matrix

    # Extract camera parameters
    camera_parameters = []

    for camera in cameras:
        camera_id = camera.get('id')
        sensor_id = camera.get('sensor_id')
        component_id = camera.get('component_id')
        label = camera.get('label')

        # Find and convert transform elements to a 4x4 matrix
        transform = camera.find('transform').text.strip().split()
        transform = [float(x) for x in transform]
        transform_matrix = np.array(transform).reshape(4, 4)

        # Check for optional rotation_covariance and convert to a 3x3 matrix
        rotation_covariance_elem = camera.find('rotation_covariance')
        if rotation_covariance_elem is not None:
            rotation_covariance = rotation_covariance_elem.text.strip().split()
            rotation_covariance = [float(x) for x in rotation_covariance]
            rotation_covariance_matrix = np.array(rotation_covariance).reshape(3, 3)
        else:
            rotation_covariance_matrix = None

        # Check for optional location_covariance and convert to a 3x3 matrix
        location_covariance_elem = camera.find('location_covariance')
        if location_covariance_elem is not None:
            location_covariance = location_covariance_elem.text.strip().split()
            location_covariance = [float(x) for x in location_covariance]
            location_covariance_matrix = np.array(location_covariance).reshape(3, 3)
        else:
            location_covariance_matrix = None

        # Get the intrinsic matrix for the sensor
        intrinsic_matrix = sensor_calibration_data.get(sensor_id, None)

        camera_parameters.append({
            'camera_id': camera_id,
            'sensor_id': sensor_id,
            'component_id': component_id,
            'label': label,
            'transform_matrix': transform_matrix,
            'rotation_covariance_matrix': rotation_covariance_matrix,
            'location_covariance_matrix': location_covariance_matrix,
            'intrinsic_matrix': intrinsic_matrix
        })

    return camera_parameters


def read_camera_parameters(filename):
    """
    Reads camera parameters from a text file.

    Parameters:
    filename (str): Path to the text file.

    Returns:
    tuple: Intrinsics and extrinsics matrices.
    """
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]

    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsics, extrinsics


def create_projection_matrix(K, E):
    """
    Creates a projection matrix from intrinsics and extrinsics.

    Parameters:
    K (ndarray): Intrinsic matrix.
    E (ndarray): Extrinsic matrix.

    Returns:
    ndarray: Projection matrix.
    """
    K_4x4 = np.eye(4)
    K_4x4[:3, :3] = K

    P = np.dot(K_4x4, E).astype(np.float32)
    return P


def load_K_Rt_from_P(filename, P=None):
    """
    Loads intrinsics and extrinsics from a projection matrix.

    Parameters:
    filename (str): Path to the file.
    P (ndarray): Projection matrix.

    Returns:
    tuple: Intrinsics and pose matrices.
    """
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(d):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in d:
        if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])
    return d


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.MatlabFunction):
            d[strg] = elem
        elif isinstance(elem, spio.matlab.mio5_params.mat_struct):
            d[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray):
            d[strg] = _tolist(elem)
        else:
            d[strg] = elem
    return d


def _tolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem, np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list

def construct_cam_matrices(camera_params):
    K = camera_params['K']
    R, t = camera_params['R'], camera_params['T']
    num_cams = K.shape[0]
    cam_matrices = np.zeros((num_cams, 4, 3))
    for cidx in range(num_cams):
        extrinsic = np.vstack([R[cidx].T, -t[cidx]])
        cam_matrices[cidx] = extrinsic @ K[cidx].T
    return cam_matrices

def load_camera_params(calibpath, is_meters=False):
    calibname = calibpath.split('/')[-1]
    root_key = 'calibrationOptimized' if calibname.startswith('optimized') \
        else 'calibration'
    print(root_key)
    calib_dict = loadmat(calibpath)[root_key]
    params_dict = {'K': [], 'D': [], 'R': [], 'T': []}
    num_cameras = len(calib_dict['ImageSize'])
    for idx in range(num_cameras):
        params_dict['K'].append(calib_dict['CameraParameters'][idx]['cameraMatrix'])
        params_dict['D'].append(calib_dict['CameraParameters'][idx]['distCoeffs'])
        rotmat, _ = cv.Rodrigues(np.array(calib_dict['ExtR'][idx], dtype=np.float32))
        params_dict['R'].append(rotmat.tolist())
        params_dict['T'].append(calib_dict['ExtT'][idx])
    for key in params_dict:
        params_dict[key] = np.array(params_dict[key])
    
    scale = 1 if is_meters else 0.001
    # params_dict['K'][:2, 2] *= scale
    params_dict['R'] = params_dict['R']
    params_dict['T'] *= scale
    return params_dict


def load_camera_params_mat(calibpath, is_meters=False):
    # .mat
    calibname = calibpath.split('/')[-1]
    root_key = 'calibrationOptimized' if calibname.startswith('optimized') \
        else 'calibration'
    print(root_key)
    calib_dict = loadmat(calibpath)[root_key]
    params_dict = {'K': [], 'D': [], 'R': [], 'T': []}
    num_cameras = len(calib_dict['ImageSize'])
    for idx in range(num_cameras):
        params_dict['K'].append(calib_dict['CameraParameters'][idx]['cameraMatrix'])
        params_dict['D'].append(calib_dict['CameraParameters'][idx]['distCoeffs'])
        rotmat, _ = cv.Rodrigues(np.array(calib_dict['ExtR'][idx], dtype=np.float32))
        params_dict['R'].append(rotmat.tolist())
        params_dict['T'].append(calib_dict['ExtT'][idx])
    for key in params_dict:
        params_dict[key] = np.array(params_dict[key])
    
    scale = 1 if is_meters else 0.001
    params_dict['R'] = params_dict['R']
    params_dict['T'] *= scale
    return params_dict


def load_obj(filename):
    """ Load OBJ file into vertices and faces. """
    vertices = []
    faces = []
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            elif parts[0] == 'v':  # This line describes a vertex
                # Convert the remaining parts to float and add to vertices list
                vertices.append([float(p) for p in parts[1:]])
            elif parts[0] == 'f':  # This line describes a face
                # Convert the remaining parts to integer index (subtract 1 because OBJ indexing starts at 1)
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)

    # Convert vertices list to a NumPy array for better performance
    vertices = np.array(vertices)
    
    return vertices, faces



def save_hair2pc(path, vert, color=None):
    """
    Save a 3D model in the OBJ format.
    Args:
        path (str): Path to save the OBJ file.
        vert (np.ndarray): Vertices of the model.
        color (np.ndarray, optional): Colors for the vertices.
    """
    with open(path, 'w') as f:
        if color is not None:
            for v, c in zip(vert, color):
                data = 'v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], c[0], c[1], c[2])
                f.write(data)
        else:
            for v in vert:
                data = 'v %f %f %f\n' % (v[0], v[1], v[2])
                f.write(data)