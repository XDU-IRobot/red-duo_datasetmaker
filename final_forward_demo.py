#控制效果差，未完成运动
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

print("=== 最终修复版：解决前进距离不足问题 ===")
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
new_pos = pos_np + np.array([0.0, 0.0, 0.03])  # 降低高度，增加轮地接触
robot.set_pos(new_pos)

# 高精度舵角控制 + 混合轮子控制
kp_values = np.ones(robot.n_dofs) * 50.0

# 舵关节：极高增益位置控制
for i, idx in enumerate([0, 2, 4, 6]):
    if dofs_idx[idx] < robot.n_dofs:
        kp_values[dofs_idx[idx]] = 15000.0

# 轮关节：中等增益配合外部力控制        
for i, idx in enumerate([1, 3, 5, 7]):
    if dofs_idx[idx] < robot.n_dofs:
        kp_values[dofs_idx[idx]] = 500.0    # 提高基础增益

robot.set_dofs_kp(kp_values)

print("控制模式: 舵关节=高增益位置控制(15000), 轮关节=混合控制(500)")

# 前进优化参数
target_duo_angle = 0.0
target_wheel_speed = 2.0        # 进一步提高目标轮速
stabilization_time = 100        # 更短的稳定时间
angle_tolerance = 0.08          # 更严格的舵角容许误差
max_angle_for_forward = 0.2     # 允许前进的最大舵角误差
wheel_force_gain = 2000.0       # 更强的轮子驱动力增益
max_wheel_force = 3000.0        # 更高的最大轮子力矩

print(f"\n=== 最终优化前进控制 ===")
print(f"目标轮速: {target_wheel_speed}")
print(f"轮子力增益: {wheel_force_gain}, 最大力矩: {max_wheel_force}")
print(f"舵角容许误差: {angle_tolerance:.3f} 弧度 ({angle_tolerance*57.3:.1f} 度)")

# 外部基座推进力
base_push_force = 0.0
total_push_applied = 0.0

# 仿真循环
for i in range(2000):  # 增加仿真时间
    # 获取当前关节位置和速度
    current_dof_pos = robot.get_dofs_position()
    current_dof_vel = robot.get_dofs_velocity()
    
    # 舵关节高精度位置控制
    target_positions = current_dof_pos.clone()
    duo_angles = []
    
    for idx in duo_joint_indices:
        target_positions[idx] = target_duo_angle
        duo_angles.append(current_dof_pos[idx].item())
    
    robot.control_dofs_position(target_positions)
    
    # 轮关节强化控制
    forces = np.zeros(robot.n_dofs)
    
    # 检查舵角稳定性
    max_duo_error = max(abs(angle - target_duo_angle) for angle in duo_angles)
    duo_stable = max_duo_error < angle_tolerance
    duo_can_forward = max_duo_error < max_angle_for_forward
    
    current_wheel_speeds = []
    
    if i > stabilization_time:
        if duo_stable:
            # 舵角稳定：全力前进
            ramp_progress = min(1.0, (i - stabilization_time) / 200.0)
            actual_target_speed = target_wheel_speed * ramp_progress
            speed_factor = 1.0
            
            # 添加基座推进力帮助前进
            if i > stabilization_time + 50 and total_push_applied < 100:
                base_push_force = 300.0  # 基座前进推力
                total_push_applied += 1
            else:
                base_push_force = 0.0
                
        elif duo_can_forward:
            # 舵角可前进：适度降速
            error_ratio = (max_duo_error - angle_tolerance) / (max_angle_for_forward - angle_tolerance)
            speed_reduction = error_ratio * 0.5
            ramp_progress = min(1.0, (i - stabilization_time) / 250.0)
            actual_target_speed = target_wheel_speed * ramp_progress * (1.0 - speed_reduction)
            speed_factor = 1.0 - speed_reduction * 0.3
            base_push_force = 0.0
        else:
            actual_target_speed = 0.0
            speed_factor = 0.0
            base_push_force = 0.0
    else:
        actual_target_speed = 0.0
        speed_factor = 0.0
        base_push_force = 0.0
    
    # 轮子控制：位置控制 + 外部力增强
    for idx in wheel_joint_indices:
        current_velocity = current_dof_vel[idx].item()
        speed_error = actual_target_speed - current_velocity
        
        current_wheel_speeds.append(current_velocity)
        
        if actual_target_speed > 0.1:
            # 外部驱动力
            force = speed_error * wheel_force_gain * speed_factor
            # 增加持续前进力克服阻力
            force += 800.0 * speed_factor  # 增强前进偏向力
            forces[idx] = np.clip(force, -max_wheel_force, max_wheel_force)
            
            # 同时使用位置控制（适当的目标速度转换为位置）
            if i > stabilization_time + 20:
                # 累积位置目标（模拟轮子转动）
                wheel_rotation = (i - stabilization_time - 20) * actual_target_speed * 0.0167 * 0.5
                target_positions[idx] = wheel_rotation
        else:
            forces[idx] = 0.0
    
    # 基座稳定和推进
    forces[0] = base_push_force  # X方向推进力
    forces[2] = -200.0           # Z方向向下力
    
    # 应用所有控制
    robot.control_dofs_position(target_positions)
    robot.control_dofs_force(forces)
    
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
            status = "全力前进"
        elif duo_can_forward:
            status = "适度前进"
        else:
            status = "等待稳定"
        
        print(f"\n--- 步数: {i:3d} | {status} ---")
        print(f"基座位置: [{base_pos[0]:6.3f}, {base_pos[1]:6.3f}, {base_pos[2]:6.3f}]")
        print(f"舵角误差: {max_duo_error:.4f} 弧度 ({max_duo_error*57.3:.2f} 度)")
        print(f"轮速: 目标={actual_target_speed:.3f}, 实际={avg_wheel_speed:.3f}")
        print(f"侧向偏移: {lateral_drift:.4f}m")
        print(f"前进距离: {base_pos[0]:.4f}m")
        print(f"推进力: {base_push_force:.0f}")
        
        if base_pos[0] > 0.5:
            print("🎉 显著前进！")

print("\n=== 最终修复版仿真完成 ===")

# 最终状态评估
current_pos = robot.get_dofs_position()
base_pos = current_pos[:3].cpu().numpy()
final_duo_angles = [current_pos[idx].item() for idx in duo_joint_indices]
final_drift = abs(base_pos[1])
final_duo_error = max(abs(angle - target_duo_angle) for angle in final_duo_angles)

print(f"\n=== 最终状态评估 ===")
print(f"最终位置: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
print(f"最终舵角误差: {final_duo_error:.4f} 弧度 ({final_duo_error*57.3:.2f} 度)")
print(f"最终侧向偏移: {final_drift:.4f}m")
print(f"总前进距离: {base_pos[0]:.3f}m")

# 综合评估
if base_pos[0] > 1.0 and final_drift < 0.1 and final_duo_error < 0.15:
    print("🎉 前进任务完全成功！机器人实现了稳定有效的直线前进")
elif base_pos[0] > 0.3:
    print("✓ 前进任务基本成功，机器人有明显前进，但仍有优化空间")
elif base_pos[0] > 0.1:
    print("⚠️  前进任务部分成功，机器人有一定前进")
else:
    print("❌ 前进任务失败，机器人前进距离不足")

print(f"\n机器人性能总结：")
print(f"- 舵角控制: {'优秀' if final_duo_error < 0.05 else '良好' if final_duo_error < 0.1 else '需改进'}")
print(f"- 侧向稳定: {'优秀' if final_drift < 0.05 else '良好' if final_drift < 0.1 else '需改进'}")  
print(f"- 前进能力: {'优秀' if base_pos[0] > 1.0 else '良好' if base_pos[0] > 0.3 else '需改进'}")
