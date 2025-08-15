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
robot = scene.add_entity(gs.morphs.URDF(file='urdf/red-duo.urdf'))

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

print("=== 优化前进控制：强化舵角控制+增强驱动力 ===")
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

# 双重控制模式：舵关节使用极高增益位置控制，轮关节使用强化驱动控制
kp_values = np.ones(robot.n_dofs) * 20.0  # 基座适中增益

# 舵关节：超高位置增益
for i, idx in enumerate([0, 2, 4, 6]):  # 舵关节在jnt_names中的索引
    if dofs_idx[idx] < robot.n_dofs:
        kp_values[dofs_idx[idx]] = 25000.0  # 超高位置增益

# 轮关节：较低增益，但会用外部力增强        
for i, idx in enumerate([1, 3, 5, 7]):  # 轮关节在jnt_names中的索引
    if dofs_idx[idx] < robot.n_dofs:
        kp_values[dofs_idx[idx]] = 100.0    # 适中基础增益

robot.set_dofs_kp(kp_values)

print("控制模式: 舵关节=超高增益位置控制(25000), 轮关节=混合控制(100)")

# 增强的控制参数
target_duo_angle = 0.0          # 舵关节角度（0度=朝前）
target_wheel_speed = 1.5        # 提高目标轮速
stabilization_time = 150        # 减少舵角稳定时间
angle_tolerance = 0.1           # 收紧舵角容许误差（约5.7度）
max_angle_for_forward = 0.25    # 允许前进的最大舵角误差（约14度）
wheel_force_gain = 1200.0       # 增强轮子驱动力增益
max_wheel_force = 2000.0        # 提高最大轮子力矩

# PID控制器参数（用于舵角精确控制）
duo_integral_errors = [0.0] * 4
duo_prev_errors = [0.0] * 4
duo_kp = 500.0
duo_ki = 50.0
duo_kd = 100.0

print(f"\n=== 开始优化前进控制仿真 ===")
print(f"目标舵角: {target_duo_angle:.3f} 弧度")
print(f"目标轮速: {target_wheel_speed:.3f}")
print(f"舵角容许误差: {angle_tolerance:.3f} 弧度 ({angle_tolerance*57.3:.1f} 度)")
print(f"最大前进舵角: {max_angle_for_forward:.3f} 弧度 ({max_angle_for_forward*57.3:.1f} 度)")
print(f"轮子力增益: {wheel_force_gain:.0f}, 最大力矩: {max_wheel_force:.0f}")

# 仿真循环
for i in range(1500):
    # 获取当前关节位置和速度
    current_dof_pos = robot.get_dofs_position()
    current_dof_vel = robot.get_dofs_velocity()
    
    # 1. 增强舵关节控制 - 结合内置高增益控制和外部PID补偿
    target_positions = current_dof_pos.clone()
    duo_angles = []
    duo_forces = np.zeros(robot.n_dofs)
    
    for j, idx in enumerate(duo_joint_indices):
        current_angle = current_dof_pos[idx].item()
        duo_angles.append(current_angle)
        
        # 设置目标位置
        target_positions[idx] = target_duo_angle
        
        # 外部PID补偿
        error = target_duo_angle - current_angle
        duo_integral_errors[j] += error * 0.0167  # 假设60FPS
        duo_integral_errors[j] = np.clip(duo_integral_errors[j], -1.0, 1.0)
        
        derivative = error - duo_prev_errors[j]
        duo_prev_errors[j] = error
        
        pid_output = (duo_kp * error + 
                     duo_ki * duo_integral_errors[j] + 
                     duo_kd * derivative)
        
        duo_forces[idx] = np.clip(pid_output, -3000.0, 3000.0)
    
    # 应用舵关节控制
    robot.control_dofs_position(target_positions)
    
    # 2. 增强轮关节控制
    wheel_forces = np.zeros(robot.n_dofs)
    
    # 检查舵角稳定性
    max_duo_error = max(abs(angle - target_duo_angle) for angle in duo_angles)
    duo_stable = max_duo_error < angle_tolerance
    duo_can_forward = max_duo_error < max_angle_for_forward
    
    current_wheel_speeds = []
    
    if i > stabilization_time:
        if duo_stable:
            # 舵角稳定：全速前进
            ramp_progress = min(1.0, (i - stabilization_time) / 300.0)
            actual_target_speed = target_wheel_speed * ramp_progress
            speed_factor = 1.0
        elif duo_can_forward:
            # 舵角可前进：按误差比例降速
            error_ratio = (max_duo_error - angle_tolerance) / (max_angle_for_forward - angle_tolerance)
            speed_reduction = error_ratio * 0.7  # 最多降速70%
            ramp_progress = min(1.0, (i - stabilization_time) / 400.0)
            actual_target_speed = target_wheel_speed * ramp_progress * (1.0 - speed_reduction)
            speed_factor = 1.0 - speed_reduction * 0.5
        else:
            # 舵角误差过大：缓慢前进
            actual_target_speed = target_wheel_speed * 0.2 if i > stabilization_time + 100 else 0.0
            speed_factor = 0.3
    else:
        actual_target_speed = 0.0
        speed_factor = 0.0
    
    # 应用轮子速度控制
    for idx in wheel_joint_indices:
        current_velocity = current_dof_vel[idx].item()
        speed_error = actual_target_speed - current_velocity
        
        current_wheel_speeds.append(current_velocity)
        
        # 强化的速度控制力
        force = speed_error * wheel_force_gain * speed_factor
        
        # 增加前进偏向力（克服静摩擦和阻力）
        if actual_target_speed > 0.1:
            force += 200.0 * speed_factor
        
        wheel_forces[idx] = np.clip(force, -max_wheel_force, max_wheel_force)
    
    # 3. 基座稳定力
    wheel_forces[2] = -150.0  # Z方向向下力
    
    # 合并所有外部力
    total_forces = duo_forces + wheel_forces
    robot.control_dofs_force(total_forces)
    
    # 仿真步进
    scene.step()
    
    # 每100步打印状态
    if i % 100 == 0:
        current_pos = robot.get_dofs_position()
        base_pos = current_pos[:3].cpu().numpy()
        
        lateral_drift = abs(base_pos[1])
        avg_wheel_speed = np.mean(current_wheel_speeds)
        
        if i < stabilization_time:
            status = "舵角稳定中"
        elif duo_stable:
            status = "稳定前进"
        elif duo_can_forward:
            status = "降速前进"
        else:
            status = "等待稳定"
        
        print(f"\n--- 步数: {i:3d} | {status} ---")
        print(f"基座位置: [{base_pos[0]:6.3f}, {base_pos[1]:6.3f}, {base_pos[2]:6.3f}]")
        print(f"舵角度: {[f'{x:6.3f}' for x in duo_angles]}")
        print(f"舵角误差: {max_duo_error:.4f} 弧度 ({max_duo_error*57.3:.2f} 度)")
        print(f"轮速: 目标={actual_target_speed:.3f}, 实际={avg_wheel_speed:.3f}")
        print(f"侧向偏移: {lateral_drift:.4f}m")
        print(f"舵角状态: 稳定={duo_stable}, 可前进={duo_can_forward}")
        print(f"前进距离: {base_pos[0]:.3f}m")
        
        if lateral_drift > 0.15:
            print("⚠️  侧向偏移较大！")

print("\n=== 优化前进仿真完成 ===")
print("优化策略总结：")
print("1. 舵关节使用超高增益位置控制+外部PID补偿")
print("2. 轮关节使用强化驱动力控制")
print("3. 动态速度调节和误差容忍机制")
print("4. 增加前进偏向力克服静摩擦")

# 最终状态评估
current_pos = robot.get_dofs_position()
base_pos = current_pos[:3].cpu().numpy()
final_duo_angles = [current_pos[idx].item() for idx in duo_joint_indices]
final_drift = abs(base_pos[1])
final_duo_error = max(abs(angle - target_duo_angle) for angle in final_duo_angles)

print(f"\n=== 最终状态评估 ===")
print(f"最终位置: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
print(f"最终舵角: {[f'{x:.4f}' for x in final_duo_angles]}")
print(f"最终舵角误差: {final_duo_error:.4f} 弧度 ({final_duo_error*57.3:.2f} 度)")
print(f"最终侧向偏移: {final_drift:.4f}m")

if final_drift < 0.05:
    print("✓ 侧向偏移优秀")
elif final_drift < 0.1:
    print("✓ 侧向偏移良好") 
else:
    print("✗ 侧向偏移仍然较大")

if final_duo_error < 0.05:
    print("✓ 舵角控制精度优秀")
elif final_duo_error < 0.1:
    print("✓ 舵角控制精度良好") 
else:
    print("✗ 舵角控制精度需要改进")

# 计算总前进距离
total_distance = base_pos[0]
print(f"总前进距离: {total_distance:.3f}m")

if total_distance > 1.5 and final_drift < 0.1 and final_duo_error < 0.15:
    print("🎉 前进任务成功完成！机器人实现了有效的直线前进")
elif total_distance > 0.5:
    print("✓ 前进任务部分成功，机器人有明显前进")
else:
    print("⚠️  前进距离不足，需要进一步优化")
