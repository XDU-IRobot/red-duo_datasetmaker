import genesis as gs
import numpy as np
import math

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
robot = scene.add_entity(gs.morphs.URDF(file='urdf/red-duo.urdf'))
scene.build()

# 关节名称
jnt_names = [
    'left1_duo_joint',    # 索引0
    'left1_wheel_joint',  # 索引1
    'left2_duo_joint',    # 索引2
    'left2_wheel_joint',  # 索引3
    'right1_duo_joint',   # 索引4
    'right1_wheel_joint', # 索引5
    'right2_duo_joint',   # 索引6
    'right2_wheel_joint', # 索引7
]

dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]

print("=== 机器人控制初始化 ===")
print(f"关节DOF索引: {dofs_idx}")

# 设置增益
kp_values = np.ones(robot.n_dofs) * 100.0  # 基础增益
for idx in dofs_idx:
    kp_values[idx] = 2000.0  # 关节增益更高

robot.set_dofs_kp(kp_values)
print(f"增益设置完成: {kp_values}")

# 调整机器人初始位置
pos = robot.get_pos()
pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else pos
new_pos = pos_np + np.array([0.0, 0.0, 0.1])  
robot.set_pos(new_pos)

print("\n=== 开始运动控制 ===")

# 简化的运动控制
for i in range(500):
    t = i * 0.02
    
    # 生成简单的轮子运动
    wheel_angle = math.sin(t) * 10.0  # 正弦波运动
    
    # 为轮子关节设置目标角度
    wheel_indices = [dofs_idx[1], dofs_idx[3], dofs_idx[5], dofs_idx[7]]  # 轮子关节
    
    try:
        # 使用力矩控制
        current_pos = robot.get_dofs_position()
        target_pos = current_pos.clone()
        
        # 设置轮子目标角度
        for wheel_idx in wheel_indices:
            target_pos[wheel_idx] = wheel_angle
            
        robot.control_dofs_position(target_pos)
        
    except AttributeError:
        # 如果control_dofs_position不存在，尝试其他方法
        forces = np.zeros(robot.n_dofs)
        for wheel_idx in wheel_indices:
            current_angle = robot.get_dofs_position()[wheel_idx].item()
            error = wheel_angle - current_angle
            forces[wheel_idx] = kp_values[wheel_idx] * error
        
        robot.set_dofs_force(forces)
    
    scene.step()
    
    if i % 50 == 0:
        current_angles = robot.get_dofs_position()
        wheel_angles = [current_angles[idx].item() for idx in wheel_indices]
        print(f"步数: {i:3d}, 目标角度: {wheel_angle:6.3f}, 轮子角度: {wheel_angles}")

print("\n=== 运动控制测试完成 ===")
