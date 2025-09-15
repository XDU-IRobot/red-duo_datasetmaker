import cv2
import numpy as np
# 输入抽象层：msg or csv
state_dim  = 10
meas_dim   = 7
kf = cv2.KalmanFilter(state_dim, meas_dim, 0)
# 状态向量 [x, y, z, vx, vy, vz, qx, qy, qz, qw]
# 测量向量 [x, y, z, qx, qy, qz, qw]
dt = 0.02  # 时间步长
kf.transitionMatrix = np.eye(state_dim, dtype=np.float32)
kf.transitionMatrix[0,3] = 1.0
kf.transitionMatrix[1,4] = 1.0
kf.transitionMatrix[2,5] = 1.0

kf.measurementMatrix = np.eye(meas_dim, dtype=np.float3)
kf.measurementMatrix[0,0] = 1.0
kf.measurementMatrix[1,1] = 1.0
kf.measurementMatrix[2,2] = 1.0
kf.measurementMatrix[3,6] = 1.0
kf.measurementMatrix[4,7] = 1.0
kf.measurementMatrix[5,8] = 1.0
kf.measurementMatrix[6,9] = 1.0

kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * 1e-3
kf.measurementNoiseCov = np.eye(meas_dim, dtype=np.float32) * 1e-2

kf.errorCovPost = np.eye(state_dim, dtype=np.float32) * 1.0
kf.statePost = np.array([0, 0, 0,   # pos
                         0, 0, 0,   # vel
                         0, 0, 0, 1], dtype=np.float32)  # quat

def normalize_quaternion(q):
    q = q / np.linalg.norm(q)
    return q
measured_pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
measured_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

measurement = np.hstack([measured_pos, measured_quat]).astype(np.float32)
pred = kf.predict()
kf.correct(measurement)
filtered_state = kf.statePost
pos = filtered_state[0:3]
quat = normalize_quaternion(filtered_state[6:10])
print(pos)
print(quat)