import numpy as np
import os
import csv
import matplotlib.pyplot as plt
class Logger:
    def __init__(self, log_dir, log_file):
        self.log_dir = log_dir
        self.log_file = log_file
        os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, log_file)
        with open(self.file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['step','pos_x', 'pos_y', 'pos_z', 'ori_x','ori_y','ori_z','ori_w'])  # Header row

    def log(self, step, robot):
        # 获取位姿
        pos = robot.get_pos().cpu().numpy() if hasattr(robot.get_pos(), 'cpu') else robot.get_pos()
        ori = robot.get_quat().cpu().numpy() if hasattr(robot.get_quat(), 'cpu') else robot.get_quat()
        
        # 获取关节角度
        joint_pos = robot.get_dofs_position()
        if hasattr(joint_pos, 'cpu'):  # 兼容 torch
            joint_pos = joint_pos.cpu().numpy()
        joint_str = ";".join(map(str, joint_pos))  # 存成字符串，避免列数过多

        # 写入 CSV
        with open(self.file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                pos[0], pos[1], pos[2],
                ori[0], ori[1], ori[2], ori[3],
                joint_str
            ])
    def plot_trajectory(self,show=True,save_as=None):
        if not os.path.exists(self.file_path):
            print("No log file found.")
            return
        pos_arr = np.array(self.positions)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pos_arr[:,0], pos_arr[:,1], pos_arr[:,2], label="Trajectory", color="b")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        if save_as:
            plt.savefig(save_as)
        if show:
            plt.show()
        plt.close()
