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

def plot_base_velocity():
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    df.plot(x='time_step', y='base_vx', ax=axs[0], legend=False)
    df.plot(x='time_step', y='base_vy', ax=axs[1], legend=False)
    df.plot(x='time_step', y='base_wz', ax=axs[2], legend=False)

    axs[0].hlines(y=1, xmin=0, xmax=10, colors='r', linestyles='--', label='command vx')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('$V_x$ (m/s)')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].hlines(y=0, xmin=0, xmax=10, colors='r', linestyles='--', label='command vy')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('$V_y$ (m/s)')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].hlines(y=0, xmin=0, xmax=10, colors='r', linestyles='--', label='command wz')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('$w_z$ (m/s)')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def plot_dof_torque():
    n_dof = len(DOF_NAMES)
    fig, axs = plt.subplots(5, 2, figsize=(10, 10))

    axs = axs.flatten()
    for index, key in enumerate(DOF_NAMES):
        column_name = 'torque_' + key
        df.plot(x='time_step', y=column_name, ax=axs[index], legend=False, linewidth=0.9)
        axs[index].set_xlabel('Time (s)')
        axs[index].set_ylabel(column_name)
        axs[index].grid(True)

    plt.tight_layout()
    plt.show()    

def plot_joint_positions():
    n_dof = len(DOF_NAMES)
    fig, axs = plt.subplots(5, 2, figsize=(10, 10))

    axs = axs.flatten()
    for index, key in enumerate(DOF_NAMES):
        column_name = 'pos_' + key
        df.plot(x='time_step', y=column_name, ax=axs[index], legend=False, linewidth=0.9)
        axs[index].set_xlabel('Time (s)'); axs[index].set_ylabel(column_name)
        axs[index].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # plot_vel_tracking()
    # plot_base_velocity()
    plot_dof_torque()
