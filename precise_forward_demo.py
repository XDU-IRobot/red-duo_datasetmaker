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

print("=== 精确前进控制（带积分PID和阻尼优化）===")
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

# 设置基础增益 - 关闭内置PD控制，改用外部精确控制
kp_values = np.zeros(robot.n_dofs)  # 关闭所有内置增益

robot.set_dofs_kp(kp_values)

print("增益设置: 全部使用外部PID控制，内置增益=0")

# PID控制器参数
class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.reset()
    
    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
    
    def update(self, error, dt=0.01):
        # 积分项
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        # 微分项
        derivative = (error - self.last_error) / dt
        self.last_error = error
        
        # PID输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return output

# 为每个舵关节创建PID控制器
duo_pids = [PIDController(kp=4000.0, ki=200.0, kd=800.0, integral_limit=0.5) 
            for _ in duo_joint_indices]

# 轮关节速度控制器（较简单的P控制）
wheel_speed_controllers = [PIDController(kp=600.0, ki=50.0, kd=100.0, integral_limit=0.2)
                          for _ in wheel_joint_indices]

print("PID控制器初始化完成")
print(f"舵关节PID: Kp=4000, Ki=200, Kd=800")
print(f"轮关节PID: Kp=600, Ki=50, Kd=100")

# 控制参数
target_duo_angle = 0.0      # 舵关节角度（0度=朝前）
target_wheel_speed = 0.5    # 目标轮速（较保守）
stabilization_time = 300    # 舵角稳定时间
angle_tolerance = 0.01      # 舵角容许误差（约0.6度）

print(f"\n=== 开始精确前进仿真 ===")
print(f"目标舵角: {target_duo_angle:.3f} 弧度")
print(f"目标轮速: {target_wheel_speed:.3f}")
print(f"舵角容许误差: {angle_tolerance:.3f} 弧度 ({angle_tolerance*57.3:.1f} 度)")

# 仿真循环
for i in range(1000):
    # 计算控制力矩
    forces = np.zeros(robot.n_dofs)
    
    # 获取当前关节位置和速度
    current_dof_pos = robot.get_dofs_position()
    current_dof_vel = robot.get_dofs_velocity()
    
    # 1. 舵关节精确PID控制
    duo_angles = []
    duo_stable = True
    
    for idx, (joint_idx, pid) in enumerate(zip(duo_joint_indices, duo_pids)):
        current_angle = current_dof_pos[joint_idx].item()
        current_velocity = current_dof_vel[joint_idx].item()
        angle_error = target_duo_angle - current_angle
        duo_angles.append(current_angle)
        
        # 检查是否稳定
        if abs(angle_error) > angle_tolerance:
            duo_stable = False
        
        # PID控制输出
        pid_output = pid.update(angle_error)
        
        # 添加速度阻尼
        velocity_damping = -current_velocity * 500.0
        
        forces[joint_idx] = pid_output + velocity_damping
        
        # 限制最大力矩防止震荡
        forces[joint_idx] = np.clip(forces[joint_idx], -8000.0, 8000.0)
    
    # 2. 轮关节控制 - 基于舵角稳定状态
    current_wheel_speeds = []
    
    # 只有在舵角稳定且过了稳定时间后才启动轮子
    if i > stabilization_time and duo_stable:
        # 渐进加速
        ramp_progress = min(1.0, (i - stabilization_time) / 200.0)
        actual_target_speed = target_wheel_speed * ramp_progress
        
        # 进一步检查舵角误差，动态调整轮速
        max_duo_error = max(abs(angle) for angle in duo_angles)
        if max_duo_error > angle_tolerance * 2:  # 如果舵角误差超过2倍容许值
            actual_target_speed *= 0.3  # 大幅降速
        elif max_duo_error > angle_tolerance:
            actual_target_speed *= 0.7  # 适度降速
    else:
        actual_target_speed = 0.0
    
    # 为每个轮子应用速度控制
    for idx, (joint_idx, speed_pid) in enumerate(zip(wheel_joint_indices, wheel_speed_controllers)):
        current_velocity = current_dof_vel[joint_idx].item()
        speed_error = actual_target_speed - current_velocity
        
        # 速度PID控制
        speed_output = speed_pid.update(speed_error)
        forces[joint_idx] = speed_output
        
        # 限制力矩
        forces[joint_idx] = np.clip(forces[joint_idx], -1000.0, 1000.0)
        
        current_wheel_speeds.append(current_velocity)
    
    # 3. 基座稳定 - 轻微向下的力保持接地
    forces[2] = -150.0  # Z方向向下
    
    # 应用力矩
    robot.control_dofs_force(forces)
    
    # 仿真步进
    scene.step()
    
    # 每80步打印一次详细状态
    if i % 80 == 0:
        current_pos = robot.get_dofs_position()
        base_pos = current_pos[:3].cpu().numpy()
        
        max_duo_error = max(abs(angle - target_duo_angle) for angle in duo_angles)
        lateral_drift = abs(base_pos[1])
        
        # 计算平均轮速
        avg_wheel_speed = np.mean(current_wheel_speeds)
        
        status = "舵角稳定中" if i < stabilization_time else ("前进中" if duo_stable else "等待舵角稳定")
        
        print(f"\n--- 步数: {i:3d} | {status} ---")
        print(f"基座位置: [{base_pos[0]:6.3f}, {base_pos[1]:6.3f}, {base_pos[2]:6.3f}]")
        print(f"舵角度: {[f'{x:6.3f}' for x in duo_angles]}")
        print(f"舵角误差: {max_duo_error:.4f} 弧度 ({max_duo_error*57.3:.2f} 度)")
        print(f"轮速: 目标={actual_target_speed:.3f}, 实际={avg_wheel_speed:.3f}")
        print(f"侧向偏移: {lateral_drift:.4f}m")
        print(f"舵角稳定: {'是' if duo_stable else '否'}")
        
        if lateral_drift > 0.3:
            print("警告：侧向偏移较大！")
        
        if base_pos[2] > 0.5:
            print("警告：机器人高度异常！")

print("\n=== 精确前进仿真完成 ===")
print("优化策略总结：")
print("1. 使用PID控制器（含积分项）消除舵角稳态误差")
print("2. 增加关节阻尼减少震荡")
print("3. 分阶段启动，确保舵角充分稳定")
print("4. 动态轮速调整，根据舵角误差自动减速")
print("5. 限制最大力矩防止过度震荡")

# 最终状态检查
current_pos = robot.get_dofs_position()
base_pos = current_pos[:3].cpu().numpy()
final_duo_angles = [current_pos[idx].item() for idx in duo_joint_indices]
final_drift = abs(base_pos[1])

print(f"\n=== 最终状态评估 ===")
print(f"最终位置: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
print(f"最终舵角: {[f'{x:.4f}' for x in final_duo_angles]}")
print(f"最终侧向偏移: {final_drift:.4f}m")

if final_drift < 0.1:
    print("✓ 侧向偏移良好")
else:
    print("✗ 侧向偏移仍然较大")

max_final_error = max(abs(angle) for angle in final_duo_angles)
if max_final_error < 0.02:
    print("✓ 舵角控制精度良好") 
else:
    print("✗ 舵角控制精度需要改进")
