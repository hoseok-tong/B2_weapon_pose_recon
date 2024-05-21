import json
import numpy as np
import vedo


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


class Skeleton:
    def __init__(self, config_path, init_pose=None):
        self.config = load_config(config_path)
        self.skeleton = {'bones': [], 'joints': []}
        self.initialize(init_pose)
    
    def initialize(self, init_pose):
        if init_pose is None:
            num_joints = len(self.config['joint_names'])
            init_pose = np.zeros([num_joints, 3])
        
        for pair in self.config['bone_indices']:
            is_right = ['Right' in self.config['joint_names'][jidx] for jidx in pair]
            bone_lr = 0 if any(is_right) else 1
            color = [val / 255. for val in self.config['limb_colors'][bone_lr]]
            bone = vedo.Line(init_pose[pair], lw=self.config['bone_radius'], c=color)
            self.skeleton['bones'].append(bone)
        
        # Add joint actors to joints list
        joint_color = [val / 255. for val in self.config['joint_color']]
        for coords in init_pose:
            joint = vedo.Sphere(coords, r=self.config['joint_radius'], c=joint_color)
            self.skeleton['joints'].append(joint)
    
    def add(self, vp):
        for key in self.skeleton:
            vp.add(self.skeleton[key])
    
    def remove(self, vp):
        for key in self.skeleton:
            vp.remove(self.skeleton[key])
    
    def update(self, pose):
        world_pose = pose @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        for bone, pair in zip(self.skeleton['bones'], self.config['bone_indices']):
            bone.points(world_pose[pair])
        for joint, coords in zip(self.skeleton['joints'], world_pose):
            joint.pos(*coords)


class Board:
    def __init__(self, init_pose=None):
        self.kpts = []
        self.board = {'edges': [], 'vertices': []}
        # self.edge_indices = []
        self.edge_indices = [[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]]
        self.initialize(init_pose)
    
    def initialize(self, init_pose):
        if init_pose is None:
            # num_kpts = 5
            # init_pose = np.zeros([num_kpts, 3])

            num_kpts = 20
            init_pose = np.zeros([num_kpts, 3])

        num_kpts = len(init_pose)
        for pair in self.edge_indices:
            edge = vedo.Line(init_pose[pair], lw=1, c='w')
            self.board['edges'].append(edge)
        
        # Add joint actors to joints list
        color = ['w', 'b', 'yellow', 'g']
        for i, coords in enumerate(init_pose):
            vertex = vedo.Sphere(coords, r=0.03, c=color[i%4])
            self.board['vertices'].append(vertex)
    
    def add(self, vp):
        for key in self.board:
            vp.add(self.board[key])
    
    def remove(self, vp):
        for key in self.board:
            vp.remove(self.board[key])
    
    def update(self, aruco):
        world_aruco = aruco @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        for edge, pair in zip(self.board['edges'], self.edge_indices):
            edge.points(world_aruco[pair])
        for vertex, coords in zip(self.board['vertices'], world_aruco):
            vertex.pos(*coords)


# class Board:
#     def __init__(self, init_pose=None):
#         self.kpts = []
#         self.board = {'edges': [], 'vertices': []}
#         self.edge_indices = [[0, 1], [1, 2], [2, 3], [3, 0]]
#         self.initialize(init_pose)
    
#     def initialize(self, init_pose):
#         if init_pose is None:
#             num_kpts = 4
#             init_pose = np.zeros([num_kpts, 3])
#         num_kpts = len(init_pose)
#         for pair in self.edge_indices:
#             edge = vedo.Line(init_pose[pair], lw=4, c='w')
#             self.board['edges'].append(edge)
        
#         # Add joint actors to joints list
#         for coords in init_pose:
#             vertex = vedo.Sphere(coords, r=0.02, c='w')
#             self.board['vertices'].append(vertex)
    
#     def add(self, vp):
#         for key in self.board:
#             vp.add(self.board[key])
    
#     def remove(self, vp):
#         for key in self.board:
#             vp.remove(self.board[key])
    
#     def update(self, aruco):
#         world_aruco = aruco @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
#         for edge, pair in zip(self.board['edges'], self.edge_indices):
#             edge.points(world_aruco[pair])
#         for vertex, coords in zip(self.board['vertices'], world_aruco):
#             vertex.pos(*coords)


class Mesh:
    def __init__(self, vertices, faces):
        self.mesh = None
        self.initialize(vertices, faces)
    
    def initialize(self, vertices, faces):
        self.mesh = vedo.Mesh([vertices, faces], c=[225, 225, 225])
        # self.mesh.lighting('ambient')
    
    def add(self, vp):
        vp.add(self.mesh)

    def remove(self, vp):
        vp.remove(self.mesh)
    
    def update(self, vertices):
        self.mesh.points(vertices)


class Cameras:
    def __init__(self, camera_params):
        # self.camera_params = load_camera_params(params_path)
        self.camera_params = camera_params
        self.cameras = {'names': [], 'pyramids': [], 'axes': []}
        self.initialize()
    
    def initialize(self):
        rotmats = self.camera_params['R']
        transls = -self.camera_params['T']
        for idx, (rotmat, transl) in enumerate(zip(rotmats, transls)):
            nametxt = 'CAM{:02d}'.format(idx + 1)
            camera_transl = transl @ rotmat @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            camera_rotmat = rotmat @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            # print(camera_transl)
            self.cameras['names'].append(self.get_name(nametxt, camera_transl))
            self.cameras['pyramids'].append(self.get_pyramid(camera_rotmat.T, camera_transl))
            self.cameras['axes'].append(self.get_axes(camera_rotmat.T, camera_transl))
    
    def add(self, vp):
        for key in self.cameras:
            vp.add(self.cameras[key])
    
    def remove(self, vp):
        for key in self.cameras:
            vp.remove(self.cameras[key])
    
    @staticmethod
    def get_name(nametxt, transl, scale=0.05):
        name = vedo.Text3D(nametxt, pos=transl, s=scale)
        return name
    
    @staticmethod
    def get_pyramid(rotmat, transl, scale=0.1):
        pyramid = vedo.Pyramid(c='w', alpha=0.5, s=scale, axis=(0, 0, -1), height=scale * 1.5).rotate_z(45)
        trfm = np.eye(4)
        trfm[:3, :3] = rotmat
        trfm[:3, 3] = transl
        pyramid.apply_transform(trfm, concatenate=True)
        return pyramid

    @staticmethod
    def get_axes(rotmat, transl, scale=0.2):
        cam_x_axis = np.matmul([1, 0, 0], rotmat.T)
        cam_y_axis = np.matmul([0, 1, 0], rotmat.T)
        cam_z_axis = np.matmul([0, 0, 1], rotmat.T)
        x_axis = vedo.Arrow(transl, transl + cam_x_axis * scale, c='r')
        y_axis = vedo.Arrow(transl, transl + cam_y_axis * scale, c='g')
        z_axis = vedo.Arrow(transl, transl + cam_z_axis * scale, c='b')
        return [x_axis, y_axis, z_axis]


class PosePlotter:
    def __init__(self, qt_widget=None):
        self.vp = vedo.Plotter(qt_widget=qt_widget, bg=(30, 30, 30))
        self.skeleton = None
    
    def init_skeleton(self, config_path, init_pose=None):
        self.skeleton = Skeleton(config_path, init_pose)
    
    def add_skeleton(self):
        self.skeleton.add(self.vp)
        self.vp.render()
    
    def remove_skeleton(self):
        self.skeleton.remove(self.vp)
        self.vp.render()
    
    def show(self):
        self.vp.show()
    
    def update(self, pose):
        self.skeleton.update(pose)
        self.vp.render()


class MeshPlotter:
    def __init__(self, qt_widget=None):
        self.vp = vedo.Plotter(qt_widget=qt_widget, bg=(30, 30, 30))
        self.mesh = None
    
    def init_mesh(self, vertices, faces):
        self.mesh = Mesh(vertices, faces)
    
    def add_mesh(self):
        self.mesh.add(self.vp)
        self.vp.render(resetcam=1)
    
    def remove_mesh(self):
        self.mesh.remove(self.vp)
        self.vp.render()

    def show(self):
        self.vp.show()
    
    def update(self, vertices, restcam=False):
        self.mesh.update(vertices)
        self.vp.render(restcam)


class CameraPlotter:
    def __init__(self, qt_widget=None):
        self.vp = vedo.Plotter(qt_widget=qt_widget, bg=(30, 30, 30))
        self.cameras = None
    
    def load_cameras(self, camera_params):
        if self.cameras is not None:
            self.remove_cameras()
        self.cameras = Cameras(camera_params)
    
    def add_cameras(self):
        self.cameras.add(self.vp)
        self.vp.render()
    
    def remove_cameras(self):
        self.cameras.remove(self.vp)
        self.vp.render()

    def show(self):
        self.vp.show()


class EgocentricPlotter:
    def __init__(self, qt_widget=None):
        self.vp = vedo.Plotter(
            qt_widget=qt_widget, 
            bg=(30, 30, 30),
            axes=4
            )
        
        # Add origin, axes
        scale=5
        self.vp.add(vedo.Point([0.0, 0.0, 0.0], c='r'))
        self.cameras = None
        self.skeleton = None
        self.board = None
    
    def load_cameras(self, camera_params):
        if self.cameras is not None:
            self.remove_cameras()
        self.cameras = Cameras(camera_params)
    
    def init_skeleton(self, config_path, init_pose=None):
        self.skeleton = Skeleton(config_path, init_pose)
    
    def init_board(self, init_pose=None):
        self.board = Board(init_pose)
    
    def add_cameras(self):
        self.cameras.add(self.vp)
        self.vp.render()
    
    def add_skeleton(self):
        self.skeleton.add(self.vp)
        self.vp.render()
    
    def add_board(self):
        self.board.add(self.vp)
        self.vp.render()
    
    def remove_cameras(self):
        self.cameras.remove(self.vp)
        self.vp.render()
    
    def remove_skeleton(self):
        self.skeleton.remove(self.vp)
        self.vp.render()
    
    def remove_board(self):
        self.board.remove(self.vp)
        self.vp.render()
    
    def show(self):
        self.vp.show()
    
    def update(self, aruco):
        # self.skeleton.update(pose)
        self.board.update(aruco)
        self.vp.render()
