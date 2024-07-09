import pandas as pd
import matplotlib.pyplot as plt

DOF_NAMES = [
    'R_hip_joint',
    'R_hip2_joint',
    'R_thigh_joint',
    'R_calf_joint',  
    'R_toe_joint',
    'L_hip_joint',
    'L_hip2_joint',
    'L_thigh_joint',
    'L_calf_joint',  
    'L_toe_joint'
]

file_name = 'gpugym/analysis/data/play_log_leg_2.csv'
df = pd.read_csv(file_name)

# fig, axs = plt.subplots(3, 1, figsize=(10, 8))

# df.plot(x='time_step', y='base_vx', ax=axs[0], legend=False)
# df.plot(x='time_step', y='base_vy', ax=axs[1], legend=False)
# df.plot(x='time_step', y='base_vz', ax=axs[2], legend=False)

# axs[0].set_xlabel('Time (s)')
# axs[0].set_ylabel('$V_x$ (m/s)')
# axs[0].grid(True)

# axs[1].set_xlabel('Time (s)')
# axs[1].set_ylabel('$V_y$ (m/s)')
# axs[1].grid(True)

# axs[2].set_xlabel('Time (s)')
# axs[2].set_ylabel('$V_z$ (m/s)')
# axs[2].grid(True)

# plt.tight_layout()
# plt.show()

n_dof = len(DOF_NAMES)
fig, axs = plt.subplots(5, 2, figsize=(10, 10))

axs = axs.flatten()
for index, key in enumerate(DOF_NAMES):
    column_name = 'pos_' + key
    df.plot(x='time_step', y=column_name, ax=axs[index], legend=False, linewidth=0.9)
    axs[index].set_xlabel('Time (s)')
    axs[index].set_ylabel(column_name)
    axs[index].grid(True)

plt.tight_layout()
plt.show()