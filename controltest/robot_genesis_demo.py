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

# Add a robot to the scene  
robot = scene.add_entity(
    gs.morphs.URDF(file='../urdf/red-duo-fixed.urdf')
)

scene.build()

jnt_names = [
    'left1_duo_joint',
    'left1_wheel_joint',
    'left2_duo_joint',
    'left2_wheel_joint',
    'right1_duo_joint',
    'right1_wheel_joint',
    'right2_duo_joint',
    'right2_wheel_joint',
]

dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]

# 获取初始位置并调整机器人高度
pos = robot.get_pos()
pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else pos
new_pos = pos_np + np.array([0.0, 0.0, 0.2])  # 只提升Z高度
robot.set_pos(new_pos)

print("=== 机器人控制信息 ===")
print(f"关节名称: {jnt_names}")
print(f"关节DOF索引: {dofs_idx}")
print(f"总DOF数量: {robot.n_dofs}")

# 设置关节控制参数
joint_kp = 1000.0  # 位置增益

# 为所有DOF设置增益（前6个是基座DOF，后8个是关节DOF）
all_kp = np.zeros(robot.n_dofs)

# 基座DOF保持较小增益（让它相对固定）
all_kp[:6] = 100.0  

# 关节DOF设置较大增益（用于控制）
for idx in dofs_idx:
    all_kp[idx] = joint_kp

# 设置增益
robot.set_dofs_kp(all_kp)

print(f"设置的位置增益: {all_kp}")

# 定义舵轮机器人运动控制
def swerve_drive_control(t, motion_mode="forward"):
    """
    舵轮机器人控制函数
    返回：[duo_angles, wheel_velocities]
    """
    # 4个舵关节角度 [left1, left2, right1, right2]
    duo_angles = [0.0, 0.0, 0.0, 0.0]
    # 4个轮子速度 [left1, left2, right1, right2] 
    wheel_velocities = [0.0, 0.0, 0.0, 0.0]
    
    if motion_mode == "forward":
        # 前进：舵角度为0，轮子同向转动
        duo_angles = [0.0, 0.0, 0.0, 0.0]
        speed = math.sin(t * 0.3) * 2.0  # 变速前进
        wheel_velocities = [speed, speed, speed, speed]
        
    elif motion_mode == "backward":
        # 后退：舵角度为0，轮子反向转动
        duo_angles = [0.0, 0.0, 0.0, 0.0] 
        speed = -1.5
        wheel_velocities = [speed, speed, speed, speed]
        
    elif motion_mode == "turn_left":
        # 左转：所有舵关节向左，轮子同向转动
        angle = -math.pi/6  # -30度
        duo_angles = [angle, angle, angle, angle]
        speed = 1.0
        wheel_velocities = [speed, speed, speed, speed]
        
    elif motion_mode == "turn_right":
        # 右转：所有舵关节向右，轮子同向转动
        angle = math.pi/6   # 30度
        duo_angles = [angle, angle, angle, angle]
        speed = 1.0
        wheel_velocities = [speed, speed, speed, speed]
        
    elif motion_mode == "rotate_cw":
        # 顺时针原地旋转：舵关节指向切线方向
        # 假设机器人中心在原点，轮子位置大致为：
        # left1: 前左, left2: 后左, right1: 前右, right2: 后右
        duo_angles = [math.pi/4, -math.pi/4, 3*math.pi/4, -3*math.pi/4]  # 切线方向
        speed = 1.0
        wheel_velocities = [speed, speed, speed, speed]  # 同向转动
        
    elif motion_mode == "rotate_ccw":
        # 逆时针原地旋转
        duo_angles = [-math.pi/4, math.pi/4, -3*math.pi/4, 3*math.pi/4]  # 切线方向
        speed = 1.0
        wheel_velocities = [speed, speed, speed, speed]  # 同向转动
        
    elif motion_mode == "strafe_left":
        # 左平移：所有舵关节指向左侧，轮子同向转动
        angle = -math.pi/2  # -90度
        duo_angles = [angle, angle, angle, angle]
        speed = 1.0
        wheel_velocities = [speed, speed, speed, speed]
        
    elif motion_mode == "strafe_right":
        # 右平移：所有舵关节指向右侧，轮子同向转动
        angle = math.pi/2   # 90度
        duo_angles = [angle, angle, angle, angle]
        speed = 1.0
        wheel_velocities = [speed, speed, speed, speed]
        
    elif motion_mode == "complex_motion":
        # 复杂运动：舵角和轮速都变化
        duo_angles = [
            math.sin(t * 0.2) * math.pi/4,      # left1: 摆动
            -math.sin(t * 0.25) * math.pi/4,    # left2: 反向摆动
            math.sin(t * 0.3) * math.pi/4,      # right1: 不同频率
            -math.sin(t * 0.35) * math.pi/4     # right2: 不同频率
        ]
        base_speed = 1.0
        wheel_velocities = [
            base_speed + math.cos(t * 0.4) * 0.5,    # left1
            base_speed + math.cos(t * 0.45) * 0.5,   # left2  
            base_speed + math.cos(t * 0.5) * 0.5,    # right1
            base_speed + math.cos(t * 0.55) * 0.5    # right2
        ]
        
    else:  # "stop"
        duo_angles = [0.0, 0.0, 0.0, 0.0]
        wheel_velocities = [0.0, 0.0, 0.0, 0.0]
    
    return duo_angles, wheel_velocities

# 运行仿真循环
print("\n=== 开始舵轮机器人仿真 ===")

# 定义运动序列
motion_sequence = [
    ("forward", 150),       # 前进
    ("turn_left", 100),     # 左转
    ("backward", 100),      # 后退
    ("turn_right", 100),    # 右转
    ("strafe_left", 100),   # 左平移
    ("strafe_right", 100),  # 右平移
    ("rotate_cw", 100),     # 顺时针旋转
    ("rotate_ccw", 100),    # 逆时针旋转
    ("complex_motion", 200), # 复杂运动
    ("stop", 50)            # 停止
]

current_motion_idx = 0
motion_step_count = 0

for i in range(1200):
    t = i * 0.02  # 时间变量
    
    # 确定当前运动模式
    if motion_step_count >= motion_sequence[current_motion_idx][1]:
        current_motion_idx = (current_motion_idx + 1) % len(motion_sequence)
        motion_step_count = 0
        
    current_motion = motion_sequence[current_motion_idx][0]
    motion_step_count += 1
    
    # 获取舵轮控制指令
    duo_angles, wheel_velocities = swerve_drive_control(t, current_motion)
    
    # 计算控制力矩
    forces = np.zeros(robot.n_dofs)
    
    # 舵关节位置控制（PD控制器）
    duo_joint_indices = [dofs_idx[0], dofs_idx[2], dofs_idx[4], dofs_idx[6]]  # duo joints
    current_dof_pos = robot.get_dofs_position()
    
    for j, idx in enumerate(duo_joint_indices):
        current_angle = current_dof_pos[idx].item()
        target_angle = duo_angles[j]
        angle_error = target_angle - current_angle
        
        # PD控制器
        kp_duo = 2000.0  # 位置增益
        kd_duo = 200.0   # 速度增益（简化为阻尼）
        forces[idx] = kp_duo * angle_error - kd_duo * 0.01  # 简单阻尼
    
    # 轮子关节速度控制（转换为力矩）
    wheel_joint_indices = [dofs_idx[1], dofs_idx[3], dofs_idx[5], dofs_idx[7]]  # wheel joints
    
    for j, idx in enumerate(wheel_joint_indices):
        target_velocity = wheel_velocities[j]
        # 简单的速度控制：力矩正比于目标速度
        forces[idx] = target_velocity * 1000.0  # 速度增益
    
    # 应用力矩
    robot.control_dofs_force(forces)
    
    # 仿真步进
    scene.step()
    
    # 打印信息
    if i % 50 == 0:
        current_pos = robot.get_dofs_position()
        base_pos = current_pos[:3].cpu().numpy()
        
        # 获取当前舵角和轮子角度
        current_duo_angles = [current_pos[idx].item() for idx in duo_joint_indices]
        current_wheel_angles = [current_pos[idx].item() for idx in wheel_joint_indices]
        
        print(f"\n=== 步数: {i:4d} | 运动模式: {current_motion:15s} ===")
        print(f"基座位置: [{base_pos[0]:6.3f}, {base_pos[1]:6.3f}, {base_pos[2]:6.3f}]")
        print(f"舵角目标: {[f'{x:6.3f}' for x in duo_angles]}")
        print(f"舵角当前: {[f'{x:6.3f}' for x in current_duo_angles]}")
        print(f"轮速目标: {[f'{x:6.3f}' for x in wheel_velocities]}")
        print(f"轮子角度: {[f'{x:6.3f}' for x in current_wheel_angles]}")

print("\n=== 舵轮机器人仿真完成！===")
print("已展示了舵轮机器人的各种运动模式：")
print("- 前进/后退：舵角为0，轮子同向/反向转动")
print("- 左转/右转：舵角统一转向，轮子同向转动") 
print("- 平移：舵角90度，轮子同向转动")
print("- 原地旋转：舵角切线方向，轮子同向转动")
print("- 复杂运动：舵角和轮速独立变化")