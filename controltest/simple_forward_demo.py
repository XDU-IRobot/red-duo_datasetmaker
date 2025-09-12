#控制效果一般，无法直线运动
import genesis as gs
import numpy as np
import math

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
robot = scene.add_entity(gs.morphs.URDF(file='../urdf/red-duo-fixed.urdf'))

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

print("=== 简单前进控制初始化 ===")
print(f"机器人总DOF: {robot.n_dofs}")
print(f"关节DOF索引: {dofs_idx}")

# 分离舵关节和轮关节索引
duo_joint_indices = [dofs_idx[0], dofs_idx[2], dofs_idx[4], dofs_idx[6]]  # 舵关节
wheel_joint_indices = [dofs_idx[1], dofs_idx[3], dofs_idx[5], dofs_idx[7]]  # 轮关节

print(f"舵关节索引: {duo_joint_indices}")
print(f"轮关节索引: {wheel_joint_indices}")

# 调整机器人初始位置（轻微抬高避免地面穿透）
pos = robot.get_pos()
pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else pos
new_pos = pos_np + np.array([0.0, 0.0, 0.05])  # 只抬高5cm
robot.set_pos(new_pos)

# 设置增益 - 更保守的策略
kp_values = np.ones(robot.n_dofs) * 20.0  # 进一步降低基座增益

# 舵关节需要更强的控制
for i in [0, 2, 4, 6]:  # 舵关节索引
    if dofs_idx[i] < robot.n_dofs:
        kp_values[dofs_idx[i]] = 2000.0  # 舵关节高增益

# 轮关节适中的控制        
for i in [1, 3, 5, 7]:  # 轮关节索引
    if dofs_idx[i] < robot.n_dofs:
        kp_values[dofs_idx[i]] = 300.0   # 轮关节适中增益

robot.set_dofs_kp(kp_values)

print(f"增益设置: 基座=20, 舵关节=2000, 轮关节=300")

# 简单前进控制参数
target_duo_angle = 0.0      # 舵关节角度（0度=朝前）
target_wheel_speed = 1.0    # 轮子转速

print(f"\n=== 开始简单前进仿真 ===")
print(f"目标舵角: {target_duo_angle:.1f} 度")
print(f"目标轮速: {target_wheel_speed:.1f}")

# 仿真循环
for i in range(800):
    # 计算控制力矩
    forces = np.zeros(robot.n_dofs)
    
    # 获取当前关节位置
    current_dof_pos = robot.get_dofs_position()
    
    # 1. 舵关节位置控制 - 保持所有舵关节为0度（朝前）
    # 使用更强的PD控制器，确保舵角稳定
    current_dof_vel = robot.get_dofs_velocity()
    
    for idx in duo_joint_indices:
        current_angle = current_dof_pos[idx].item()
        current_velocity = current_dof_vel[idx].item()
        angle_error = target_duo_angle - current_angle
        
        # 更强的PD控制器
        kp_duo = 3000.0  # 大幅提高位置增益
        kd_duo = 500.0   # 大幅提高阻尼
        forces[idx] = kp_duo * angle_error - kd_duo * current_velocity
    
    # 2. 轮关节速度控制 - 更加谨慎的加速策略
    # 前500步缓慢加速，给舵角充足时间稳定
    if i < 500:
        current_wheel_speed = target_wheel_speed * (i / 500.0) * 0.5  # 降低最大速度
    else:
        current_wheel_speed = target_wheel_speed * 0.5  # 保持较低速度
        
    # 检测舵角偏差，动态调整轮速
    duo_angles = [current_dof_pos[idx].item() for idx in duo_joint_indices]
    max_duo_error = max(abs(angle) for angle in duo_angles)
    
    # 如果舵角偏差过大，大幅降低轮速
    if max_duo_error > 0.02:  # 超过约1度就开始减速
        speed_reduction = min(0.8, max_duo_error * 10.0)  # 最多减速80%
        current_wheel_speed *= (1.0 - speed_reduction)
    
    # 进一步限制轮速，确保稳定性优先
    current_wheel_speed = min(current_wheel_speed, 0.3)  # 最大轮速限制
        
    for idx in wheel_joint_indices:
        # 更保守的速度控制
        forces[idx] = current_wheel_speed * 500.0  # 降低速度增益
    
    # 3. 对基座施加轻微向下的力，保持接地稳定
    forces[2] = -200.0  # Z方向向下的力
    
    # 应用力矩
    robot.control_dofs_force(forces)
    
    # 仿真步进
    scene.step()
    
    # 每100步打印一次状态
    if i % 100 == 0:
        current_pos = robot.get_dofs_position()
        base_pos = current_pos[:3].cpu().numpy()
        
        # 获取舵角和轮角
        duo_angles = [current_pos[idx].item() for idx in duo_joint_indices]
        wheel_angles = [current_pos[idx].item() for idx in wheel_joint_indices]
        
        print(f"\n--- 步数: {i:3d} ---")
        print(f"基座位置: [{base_pos[0]:6.3f}, {base_pos[1]:6.3f}, {base_pos[2]:6.3f}]")
        print(f"当前轮速: {current_wheel_speed:6.3f}")
        print(f"舵角度: {[f'{x:6.3f}' for x in duo_angles]} (目标: {target_duo_angle:.3f})")
        print(f"轮角度: {[f'{x:6.3f}' for x in wheel_angles[-2:]]}")  # 只显示后两个轮子
        
        # 检查稳定性和偏移
        lateral_drift = abs(base_pos[1])  # Y方向偏移量
        if lateral_drift > 0.5:
            print(f"警告：侧向偏移过大 {lateral_drift:.3f}m")
        
        if base_pos[2] > 0.5:  # 如果Z坐标过高，说明跳起来了
            print("警告：机器人高度过高，可能不稳定!")
        
        max_duo_error = max(abs(x - target_duo_angle) for x in duo_angles)
        if max_duo_error > 0.1:  # 舵角误差大于0.1弧度
            print(f"注意：舵角误差较大 {max_duo_error:.3f} 弧度 ({max_duo_error*57.3:.1f}度)")

print("\n=== 简单前进仿真完成 ===")
print("如果机器人稳定前进，说明基本控制没问题")
print("如果还有跳跃，可能需要：")
print("1. 进一步降低增益")
print("2. 增加阻尼")
print("3. 检查URDF模型的物理参数")
