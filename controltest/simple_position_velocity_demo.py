import genesis as gs
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

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

print("=== 简单位置速度增益控制 ===")
print(f"机器人总DOF: {robot.n_dofs}")
print(f"关节DOF索引: {dofs_idx}")

# 分离舵关节和轮关节索引
duo_joint_indices = [dofs_idx[0], dofs_idx[2], dofs_idx[4], dofs_idx[6]]  # 舵关节
wheel_joint_indices = [dofs_idx[1], dofs_idx[3], dofs_idx[5], dofs_idx[7]]  # 轮关节

print(f"舵关节索引: {duo_joint_indices}")
print(f"轮关节索引: {wheel_joint_indices}")

# 设置简单的位置和速度增益
# 位置增益 (kp): 控制关节朝向目标位置的力度
# 速度增益 (kd): 控制关节速度阻尼，减少振荡
kp_values = np.ones(robot.n_dofs) * 100.0   # 基础位置增益
kd_values = np.ones(robot.n_dofs) * 20.0    # 基础速度增益

# 舵关节：使用中等增益进行稳定控制
for i, idx in enumerate([0, 2, 4, 6]):
    if dofs_idx[idx] < robot.n_dofs:
        kp_values[dofs_idx[idx]] = 2000.0   # 舵关节位置增益
        kd_values[dofs_idx[idx]] = 100.0    # 舵关节速度阻尼

# 轮关节：使用位置控制实现转速
for i, idx in enumerate([1, 3, 5, 7]):
    if dofs_idx[idx] < robot.n_dofs:
        kp_values[dofs_idx[idx]] = 500.0    # 轮关节位置增益
        kd_values[dofs_idx[idx]] = 50.0     # 轮关节速度阻尼

# 应用增益设置
robot.set_dofs_kp(kp_values)
robot.set_dofs_kv(kd_values)

print("位置增益设置：舵关节=2000, 轮关节=500, 其他=100")
print("速度增益设置：舵关节=100, 轮关节=50, 其他=20")

# 控制参数
target_duo_angle = 0.0          # 舵关节目标角度（0度=正前方）
stabilization_time = 200        # 舵角稳定时间
wheel_rotation_speed = 2.0      # 轮子转动速度 (rad/s)

# 6DoF位姿数据存储
trajectory_data = {
    'time': [],           # 时间戳
    'position': [],       # 位置 [x, y, z]
    'quaternion': [],     # 四元数 [qx, qy, qz, qw] 
    'euler_angles': [],   # 欧拉角 [roll, pitch, yaw] (弧度)
    'linear_velocity': [], # 线速度 [vx, vy, vz]
    'angular_velocity': [], # 角速度 [wx, wy, wz]
    'step': []            # 仿真步数
}

# 仿真时间步长
dt = 1.0 / 60.0  # 假设60FPS

# 仿真循环
wheel_rotation_accumulator = 0.0    # 累积轮子转角

for i in range(2000):
    # === 收集6DoF位姿数据 ===
    # 获取位置和旋转
    pos = robot.get_pos()  # [x, y, z]
    quat = robot.get_quat()  # [qx, qy, qz, qw]
    
    # 获取线速度和角速度
    linear_vel = robot.get_vel()  # [vx, vy, vz] 
    angular_vel = robot.get_ang()  # [wx, wy, wz]
    
    # 转换为numpy数组
    pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else np.array(pos)
    quat_np = quat.cpu().numpy() if hasattr(quat, 'cpu') else np.array(quat)
    lin_vel_np = linear_vel.cpu().numpy() if hasattr(linear_vel, 'cpu') else np.array(linear_vel)
    ang_vel_np = angular_vel.cpu().numpy() if hasattr(angular_vel, 'cpu') else np.array(angular_vel)
    
    # 转换四元数为欧拉角 (roll, pitch, yaw)
    # Genesis的四元数格式可能是 [qx, qy, qz, qw]，需要转换为scipy格式 [qx, qy, qz, qw]
    try:
        scipy_quat = [quat_np[0], quat_np[1], quat_np[2], quat_np[3]]  # [qx, qy, qz, qw]
        rotation = R.from_quat(scipy_quat)
        euler_angles = rotation.as_euler('xyz', degrees=False)  # roll, pitch, yaw in radians
    except:
        # 如果转换失败，使用零值
        euler_angles = np.array([0.0, 0.0, 0.0])
    
    # 存储数据
    current_time = i * dt
    trajectory_data['time'].append(current_time)
    trajectory_data['position'].append(pos_np.copy())
    trajectory_data['quaternion'].append(quat_np.copy())
    trajectory_data['euler_angles'].append(euler_angles.copy())
    trajectory_data['linear_velocity'].append(lin_vel_np.copy())
    trajectory_data['angular_velocity'].append(ang_vel_np.copy())
    trajectory_data['step'].append(i)
    
    # 获取当前关节状态
    current_dof_pos = robot.get_dofs_position()
    
    # 设置目标位置
    target_positions = current_dof_pos.clone()
    
    # 1. 舵关节位置控制 - 始终朝向前方
    duo_angles = []
    for idx in duo_joint_indices:
        current_angle = current_dof_pos[idx].item()
        duo_angles.append(current_angle)
        target_positions[idx] = target_duo_angle  # 目标角度0
    
    # 2. 轮关节位置控制 - 通过累积位置实现连续转动
    if i > stabilization_time:
        # 舵角稳定后开始转动轮子
        dt = 1.0 / 60.0  # 假设60FPS
        wheel_rotation_accumulator += wheel_rotation_speed * dt
        
        # 设置所有轮子的目标转角
        for idx in wheel_joint_indices:
            target_positions[idx] = wheel_rotation_accumulator
    else:
        # 舵角稳定阶段，轮子保持静止
        for idx in wheel_joint_indices:
            target_positions[idx] = current_dof_pos[idx]
    
    # 应用位置控制（Genesis会自动使用设定的kp和kd增益）
    robot.control_dofs_position(target_positions)
    
    # 仿真步进
    scene.step()
    
    # 每100步打印状态信息
    if i % 100 == 0:
        base_pos = current_dof_pos[:3].cpu().numpy()
        
        # 计算舵角误差
        max_duo_error = max(abs(angle - target_duo_angle) for angle in duo_angles)
        duo_stable = max_duo_error < 0.1  # 约5.7度误差容限
        
        # 计算侧向偏移
        lateral_drift = abs(base_pos[1])
        
        # 运动状态
        if i < stabilization_time:
            status = "舵角稳定中"
        elif duo_stable:
            status = "稳定前进"
        else:
            status = "前进中"
        
        print(f"\n--- 步数: {i:3d} | {status} ---")
        print(f"基座位置: [{base_pos[0]:6.3f}, {base_pos[1]:6.3f}, {base_pos[2]:6.3f}]")
        print(f"舵角度: {[f'{x:6.3f}' for x in duo_angles]}")
        print(f"舵角误差: {max_duo_error:.4f} 弧度 ({max_duo_error*57.3:.2f}度)")
        print(f"轮子累积转角: {wheel_rotation_accumulator:.3f} 弧度")
        print(f"侧向偏移: {lateral_drift:.4f}m")
        print(f"前进距离: {base_pos[0]:.3f}m")
        
        # 检查是否达到良好前进状态
        if base_pos[0] > 0.5 and lateral_drift < 0.1:
            print("✓ 前进状态良好")

print("\n=== 简单位置速度增益控制仿真完成 ===")

# 最终评估
final_pos = robot.get_dofs_position()
final_base_pos = final_pos[:3].cpu().numpy()
final_duo_angles = [final_pos[idx].item() for idx in duo_joint_indices]
final_duo_error = max(abs(angle - target_duo_angle) for angle in final_duo_angles)
final_lateral_drift = abs(final_base_pos[1])
final_forward_distance = final_base_pos[0]

print(f"\n=== 最终结果评估 ===")
print(f"最终位置: [{final_base_pos[0]:.3f}, {final_base_pos[1]:.3f}, {final_base_pos[2]:.3f}]")
print(f"最终舵角误差: {final_duo_error:.4f} 弧度 ({final_duo_error*57.3:.2f}度)")
print(f"最终侧向偏移: {final_lateral_drift:.4f}m")
print(f"总前进距离: {final_forward_distance:.3f}m")

print(f"\n控制方法总结:")
print(f"• 舵关节：纯位置控制，kp=2000, kd=100")
print(f"• 轮关节：累积位置控制，kp=500, kd=50") 
print(f"• 无外部力或复杂PID，仅依靠Genesis内置增益控制")

# 性能评估
if final_forward_distance > 1.0:
    print("\n🎉 成功：机器人实现了有效前进！")
elif final_forward_distance > 0.3:
    print("\n✓ 部分成功：机器人有明显前进")
else:
    print("\n⚠️ 需要改进：前进距离较小")

if final_lateral_drift < 0.05:
    print("✓ 侧向稳定性优秀")
elif final_lateral_drift < 0.1:
    print("✓ 侧向稳定性良好")
else:
    print("⚠️ 侧向稳定性需要改进")

if final_duo_error < 0.1:
    print("✓ 舵角控制优秀")
elif final_duo_error < 0.2:
    print("✓ 舵角控制良好")
else:
    print("⚠️ 舵角控制需要改进")

# === 6DoF位姿数据分析和可视化 ===
print(f"\n=== 6DoF位姿数据分析 ===")
print(f"数据点总数: {len(trajectory_data['time'])}")
print(f"总仿真时间: {trajectory_data['time'][-1]:.2f} 秒")

# 转换列表为numpy数组便于分析
positions = np.array(trajectory_data['position'])
euler_angles = np.array(trajectory_data['euler_angles'])  
linear_velocities = np.array(trajectory_data['linear_velocity'])
angular_velocities = np.array(trajectory_data['angular_velocity'])
times = np.array(trajectory_data['time'])

# 位置统计
print(f"\n位置轨迹统计:")
print(f"起始位置: [{positions[0,0]:.3f}, {positions[0,1]:.3f}, {positions[0,2]:.3f}]")
print(f"结束位置: [{positions[-1,0]:.3f}, {positions[-1,1]:.3f}, {positions[-1,2]:.3f}]")
print(f"总位移: [{positions[-1,0]-positions[0,0]:.3f}, {positions[-1,1]-positions[0,1]:.3f}, {positions[-1,2]-positions[0,2]:.3f}]")
print(f"最大位移幅度: {np.max(np.linalg.norm(positions - positions[0], axis=1)):.3f}m")

# 姿态统计
print(f"\n姿态轨迹统计 (欧拉角，弧度):")
print(f"Roll  范围: [{np.min(euler_angles[:,0]):.4f}, {np.max(euler_angles[:,0]):.4f}] rad")
print(f"Pitch 范围: [{np.min(euler_angles[:,1]):.4f}, {np.max(euler_angles[:,1]):.4f}] rad") 
print(f"Yaw   范围: [{np.min(euler_angles[:,2]):.4f}, {np.max(euler_angles[:,2]):.4f}] rad")

# 速度统计
print(f"\n速度统计:")
linear_speed = np.linalg.norm(linear_velocities, axis=1)
angular_speed = np.linalg.norm(angular_velocities, axis=1)
print(f"平均线速度: {np.mean(linear_speed):.3f} m/s")
print(f"最大线速度: {np.max(linear_speed):.3f} m/s") 
print(f"平均角速度: {np.mean(angular_speed):.3f} rad/s")
print(f"最大角速度: {np.max(angular_speed):.3f} rad/s")

# 创建可视化图表
print(f"\n正在生成轨迹图表...")

# 创建多子图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Red-Duo机器人6DoF位姿数据分析', fontsize=14)

# 1. 3D轨迹图
ax = fig.add_subplot(2, 3, 1, projection='3d')
ax.plot(positions[:,0], positions[:,1], positions[:,2], 'b-', linewidth=2, label='轨迹')
ax.scatter(positions[0,0], positions[0,1], positions[0,2], color='green', s=100, label='起点')
ax.scatter(positions[-1,0], positions[-1,1], positions[-1,2], color='red', s=100, label='终点')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D轨迹')
ax.legend()

# 删除原来的subplot(2,3,1)，因为我们已经用add_subplot创建了3D图
axes[0,0].remove()

# 2. XY平面轨迹图
axes[0,1].plot(positions[:,0], positions[:,1], 'b-', linewidth=2)
axes[0,1].scatter(positions[0,0], positions[0,1], color='green', s=50, label='起点')
axes[0,1].scatter(positions[-1,0], positions[-1,1], color='red', s=50, label='终点')
axes[0,1].set_xlabel('X (m)')
axes[0,1].set_ylabel('Y (m)')
axes[0,1].set_title('XY平面轨迹')
axes[0,1].grid(True)
axes[0,1].legend()
axes[0,1].axis('equal')

# 3. 位置随时间变化
axes[0,2].plot(times, positions[:,0], 'r-', label='X')
axes[0,2].plot(times, positions[:,1], 'g-', label='Y')
axes[0,2].plot(times, positions[:,2], 'b-', label='Z')
axes[0,2].set_xlabel('时间 (s)')
axes[0,2].set_ylabel('位置 (m)')
axes[0,2].set_title('位置随时间变化')
axes[0,2].legend()
axes[0,2].grid(True)

# 4. 欧拉角随时间变化
axes[1,0].plot(times, euler_angles[:,0], 'r-', label='Roll')
axes[1,0].plot(times, euler_angles[:,1], 'g-', label='Pitch')
axes[1,0].plot(times, euler_angles[:,2], 'b-', label='Yaw')
axes[1,0].set_xlabel('时间 (s)')
axes[1,0].set_ylabel('角度 (rad)')
axes[1,0].set_title('欧拉角随时间变化')
axes[1,0].legend()
axes[1,0].grid(True)

# 5. 线速度随时间变化
axes[1,1].plot(times, linear_velocities[:,0], 'r-', label='Vx')
axes[1,1].plot(times, linear_velocities[:,1], 'g-', label='Vy')
axes[1,1].plot(times, linear_velocities[:,2], 'b-', label='Vz')
axes[1,1].plot(times, linear_speed, 'k--', label='|V|')
axes[1,1].set_xlabel('时间 (s)')
axes[1,1].set_ylabel('线速度 (m/s)')
axes[1,1].set_title('线速度随时间变化')
axes[1,1].legend()
axes[1,1].grid(True)

# 6. 角速度随时间变化
axes[1,2].plot(times, angular_velocities[:,0], 'r-', label='ωx')
axes[1,2].plot(times, angular_velocities[:,1], 'g-', label='ωy')
axes[1,2].plot(times, angular_velocities[:,2], 'b-', label='ωz')
axes[1,2].plot(times, angular_speed, 'k--', label='|ω|')
axes[1,2].set_xlabel('时间 (s)')
axes[1,2].set_ylabel('角速度 (rad/s)')
axes[1,2].set_title('角速度随时间变化')
axes[1,2].legend()
axes[1,2].grid(True)

plt.tight_layout()
plt.savefig('red_duo_trajectory_analysis.png', dpi=300, bbox_inches='tight')
print("轨迹分析图表已保存为 'red_duo_trajectory_analysis.png'")

# 保存原始数据为CSV文件
print(f"正在保存6DoF数据...")
import csv

csv_filename = 'red_duo_6dof_trajectory.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['step', 'time', 'pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w',
                  'roll', 'pitch', 'yaw', 'vel_x', 'vel_y', 'vel_z', 'omega_x', 'omega_y', 'omega_z']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for i in range(len(trajectory_data['time'])):
        writer.writerow({
            'step': trajectory_data['step'][i],
            'time': trajectory_data['time'][i],
            'pos_x': trajectory_data['position'][i][0],
            'pos_y': trajectory_data['position'][i][1],
            'pos_z': trajectory_data['position'][i][2],
            'quat_x': trajectory_data['quaternion'][i][0],
            'quat_y': trajectory_data['quaternion'][i][1],
            'quat_z': trajectory_data['quaternion'][i][2],
            'quat_w': trajectory_data['quaternion'][i][3],
            'roll': trajectory_data['euler_angles'][i][0],
            'pitch': trajectory_data['euler_angles'][i][1],
            'yaw': trajectory_data['euler_angles'][i][2],
            'vel_x': trajectory_data['linear_velocity'][i][0],
            'vel_y': trajectory_data['linear_velocity'][i][1],
            'vel_z': trajectory_data['linear_velocity'][i][2],
            'omega_x': trajectory_data['angular_velocity'][i][0],
            'omega_y': trajectory_data['angular_velocity'][i][1],
            'omega_z': trajectory_data['angular_velocity'][i][2]
        })

print(f"6DoF轨迹数据已保存为 '{csv_filename}'")
print(f"数据包含 {len(fieldnames)} 列，{len(trajectory_data['time'])} 行")

plt.show()

# 打印一些关键的轨迹指标
print(f"\n=== 关键轨迹指标 ===")
total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
straight_distance = np.linalg.norm(positions[-1] - positions[0])
path_efficiency = straight_distance / total_distance if total_distance > 0 else 0

print(f"总路径长度: {total_distance:.3f} m")
print(f"直线距离: {straight_distance:.3f} m") 
print(f"路径效率: {path_efficiency:.3f} (1.0为完全直线)")
print(f"平均转弯半径: {np.mean(np.abs(1.0 / (angular_velocities[:,2] + 1e-8))):.3f} m")
