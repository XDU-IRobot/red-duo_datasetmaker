import math
import numpy as np
import robotinitial
from logger import Logger
scene, robot, plane, jnt_names, dofs_idx = robotinitial.initial()
duo_joint_indices, wheel_joint_indices = robotinitial.get_duo_wheel_indices(dofs_idx)
print("=== 机器人控制器模块 ===")
print(f"机器人总DOF: {robot.n_dofs}")
print(f"关节DOF索引: {dofs_idx}")
print(f"舵机关节索引: {duo_joint_indices}")
print(f"轮子关节索引: {wheel_joint_indices}")
logger = Logger("./log/","rotate_traj_001.csv") # set the log file name
def posinitial(robot, duo_joint_indices, wheel_joint_indices):
    # 调整机器人初始位置（轻微抬高避免地面穿透）
    pos = robot.get_pos()
    pos_np = pos.cpu().numpy() if hasattr(pos, 'cpu') else pos
    robot.set_pos(pos_np + np.array([0.0, 0.0, 0.05])) 
    # #仿真步进测试
    # for i in range(500):
    #     # 左侧双足轮子关节角度
    #     duo_joint_angle = 0.0
    #     scene.step()
def forward_position(robot, duo_joint_indices, wheel_joint_indices,step):
    # initial position
    posinitial(robot, duo_joint_indices, wheel_joint_indices)
    #set joint buff
    joint_kp = 1000.0
    all_kp_values = np.zeros(robot.n_dofs) 
    all_kp_values[0:6] = 100.0
    for idx in dofs_idx:
        all_kp_values[idx] = joint_kp
    # all_kd_values[0:6] = 1.0
    # for idx in dofs_idx:
    #     all_kd_values[idx] = joint_kd
    # position control
    for i in range(step):
        t = i * 0.02
        wheel_angle = 40 * np.sin(t)
        current_pos = robot.get_dofs_position()
        target_pos = current_pos.clone()
        for idx in wheel_joint_indices:
            target_pos[idx] = wheel_angle
            robot.control_dofs_position(target_pos)
        logger.log(i, robot)
        scene.step()
    logger.plot_trajectory(show=True, save_as="./log/rotate_traj_001.png")
def rotate_position(robot, duo_joint_indices, wheel_joint_indices, step):
    posinitial(robot, duo_joint_indices, wheel_joint_indices)
    #set joint buff
    joint_kp = 1000.0
    all_kp_values = np.zeros(robot.n_dofs) 
    all_kp_values[0:6] = 100.0
    for idx in dofs_idx:
        all_kp_values[idx] = joint_kp
    for i in range(step):
        t = i * 0.02
        wheel_angle = 40 * np.sin(t)
        duo_angles = [math.pi/4, -math.pi/4, 3*math.pi/4, -3*math.pi/4]
        current_pos = robot.get_dofs_position()
        target_pos = current_pos.clone()
        for idx in duo_joint_indices:
            target_pos[idx] = duo_angles[idx%4]
            robot.control_dofs_position(target_pos)
        for idx in wheel_joint_indices:
            target_pos[idx] = wheel_angle
            robot.control_dofs_position(target_pos)
        scene.step()
# control test
forward_position(robot, duo_joint_indices, wheel_joint_indices, 100)
# rotate_position(robot, duo_joint_indices, wheel_joint_indices, 1000) #set the step
