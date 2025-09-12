import genesis as gs
def initial():
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
    show_viewer = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True,
        world_frame_size = 1.0,
        show_link_frame  = False,
        show_cameras     = False,
        plane_reflection = True,
        ambient_light    = (0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.Rasterizer(),
   )
    plane = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file='urdf/red-duo-fixed.urdf'))
    scene.build()
    jnt_names = [
    'left1_duo_joint',    # 左前舵关节
    'left1_wheel_joint',  # 左前轮关节
    'left2_duo_joint',    # 左后舵关节
    'left2_wheel_joint',  # 左后轮关节
    'right1_duo_joint',   # 右前舵关节
    'right1_wheel_joint', # 右前轮关节
    'right2_duo_joint',   # 右后舵关节
    'right2_wheel_joint', # 右后轮关节
    ]
    dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]
    return scene, robot, plane, jnt_names, dofs_idx
def get_joint_indices(robot, jnt_names):
    return [robot.get_joint(name).dof_idx_local for name in jnt_names]
def get_duo_wheel_indices(dofs_idx):
    duo_joint_indices = [dofs_idx[0], dofs_idx[2], dofs_idx[4], dofs_idx[6]]  # 舵关节
    wheel_joint_indices = [dofs_idx[1], dofs_idx[3], dofs_idx[5], dofs_idx[7]]  # 轮关节
    return duo_joint_indices, wheel_joint_indices