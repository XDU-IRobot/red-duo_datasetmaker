import genesis as gs
import numpy as np
import math
#控制效果差，未完成运动
# 初始化Genesis
gs.init(backend=gs.gpu)

# 创建场景
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

# 添加地面和机器人
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(gs.morphs.URDF(file='urdf/red-duo-fixed.urdf'))

# 构建场景
scene.build()

# 定义关节名称
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

# 获取关节DOF索引
dofs_idx = [robot.get_joint(name).dof_idx_local for name in jnt_names]

print("=== 分阶段前进控制 ===")
print(f"机器人总DOF: {robot.n_dofs}")
print(f"关节DOF索引: {dofs_idx}")

# 分离舵关节和轮关节索引
duo_joint_indices = [dofs_idx[0], dofs_idx[2], dofs_idx[4], dofs_idx[6]]  # 舵关节
wheel_joint_indices = [dofs_idx[1], dofs_idx[3], dofs_idx[5], dofs_idx[7]]  # 轮关节

print(f"舵关节索引: {duo_joint_indices}")
print(f"轮关节索引: {wheel_joint_indices}")

# 调整机器人初始位置
pos = robot.get_pos()
pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else pos
new_pos = pos_np + np.array([0.0, 0.0, 0.05])
robot.set_pos(new_pos)

# 设置增益 - 极高的舵关节控制增益
kp_values = np.ones(robot.n_dofs) * 10.0  # 基座极低增益

# 舵关节超强控制
for i in [0, 2, 4, 6]:  # 舵关节索引
    if dofs_idx[i] < robot.n_dofs:
        kp_values[dofs_idx[i]] = 5000.0  # 舵关节超高增益

# 轮关节禁用内置增益        
for i in [1, 3, 5, 7]:  # 轮关节索引
    if dofs_idx[i] < robot.n_dofs:
        kp_values[dofs_idx[i]] = 0.0   # 轮关节完全由外部控制

robot.set_dofs_kp(kp_values)

print(f"增益设置: 基座=10, 舵关节=5000, 轮关节=0(纯力控制)")

# 控制参数
target_duo_angle = 0.0      # 舵关节角度（0度=朝前）
target_wheel_speed = 0.8    # 目标轮速
stabilization_phase = 400   # 舵角稳定阶段步数
angle_tolerance = 0.02      # 舵角容许误差（约1度）

print(f"\n=== 开始分阶段前进仿真 ===")
print(f"阶段1: 前{stabilization_phase}步仅控制舵角稳定")
print(f"阶段2: 舵角稳定后才开始轮子转动")
print(f"舵角容许误差: {angle_tolerance:.3f} 弧度 ({angle_tolerance*57.3:.1f} 度)")

# 仿真循环
for i in range(800):
    # 计算控制力矩
    forces = np.zeros(robot.n_dofs)
    
    # 获取当前关节位置和速度
    current_dof_pos = robot.get_dofs_position()
    current_dof_vel = robot.get_dofs_velocity()
    
    # 1. 舵关节控制 - 极强的位置控制
    duo_angles = []
    for idx in duo_joint_indices:
        current_angle = current_dof_pos[idx].item()
        current_velocity = current_dof_vel[idx].item()
        angle_error = target_duo_angle - current_angle
        duo_angles.append(current_angle)
        
        # 超强PD控制器
        kp_duo = 8000.0  # 极高位置增益
        kd_duo = 1000.0  # 极高速度阻尼
        forces[idx] = kp_duo * angle_error - kd_duo * current_velocity
    
    # 检查舵角是否稳定
    max_duo_error = max(abs(angle) for angle in duo_angles)
    duo_stabilized = max_duo_error < angle_tolerance
    
    # 2. 轮关节控制 - 分阶段启动
    current_wheel_speed = 0.0
    
    if i < stabilization_phase:
        # 阶段1：专注于舵角稳定，轮子完全停止
        current_wheel_speed = 0.0
    elif duo_stabilized:
        # 阶段2：舵角稳定后才启动轮子
        progress = min(1.0, (i - stabilization_phase) / 200.0)
        current_wheel_speed = target_wheel_speed * progress
    else:
        # 如果舵角不稳定，轮子减速或停止
        current_wheel_speed = 0.1 if i > stabilization_phase + 100 else 0.0
    
    # 应用轮子力矩
    for idx in wheel_joint_indices:
        forces[idx] = current_wheel_speed * 400.0  # 适中的速度增益
    
    # 3. 基座稳定力
    forces[2] = -100.0  # Z方向轻微向下的力
    
    # 应用力矩
    robot.control_dofs_force(forces)
    
    # 仿真步进
    scene.step()
    
    # 每50步打印一次状态
    if i % 50 == 0:
        current_pos = robot.get_dofs_position()
        base_pos = current_pos[:3].cpu().numpy()
        
        # 获取轮角
        wheel_angles = [current_pos[idx].item() for idx in wheel_joint_indices]
        
        phase = "稳定舵角" if i < stabilization_phase else ("前进中" if duo_stabilized else "等待舵角")
        
        print(f"\n--- 步数: {i:3d} | 阶段: {phase} ---")
        print(f"基座位置: [{base_pos[0]:6.3f}, {base_pos[1]:6.3f}, {base_pos[2]:6.3f}]")
        print(f"舵角度: {[f'{x:6.3f}' for x in duo_angles]} (目标: {target_duo_angle:.3f})")
        print(f"舵角误差: {max_duo_error:.3f} 弧度 ({max_duo_error*57.3:.1f} 度)")
        print(f"当前轮速: {current_wheel_speed:.3f}")
        print(f"舵角稳定: {'是' if duo_stabilized else '否'}")
        
        # 检查侧向偏移
        lateral_drift = abs(base_pos[1])
        if lateral_drift > 0.2:
            print(f"警告：侧向偏移 {lateral_drift:.3f}m")

print("\n=== 分阶段前进仿真完成 ===")
print("策略说明：")
print("1. 先让舵轮稳定到0度，轮子完全停止")
print("2. 舵角稳定后才启动轮子前进")
print("3. 如果舵角再次偏离，轮子自动减速")
print("这种方法优先保证方向稳定性")
