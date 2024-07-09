import pandas as pd
import matplotlib.pyplot as plt

csv_file = 'gpugym/analysis/data/play_log_leg.csv'
df = pd.read_csv(csv_file)
data = df.to_numpy()

JOINT_INDEX = {
    'R_hip_joint': 0,
    'R_hip2_joint': 1,
    'R_thigh_joint': 2,
    'R_calf_joint': 3,  # 0.6
    'R_toe_joint': 4,
    'L_hip_joint': 5,
    'L_hip2_joint': 6,
    'L_thigh_joint': 7,
    'L_calf_joint': 8,  # 0.6
    'L_toe_joint': 9,      
}


TIME_INDEX = 0
BASE_Z_INDEX = 1
BASE_LIN_VEL_INDEX = range(2, 5)
BASE_ANG_VEL_INDEX = range(5, 8)
PROJECTED_GRAVITY_INDEX = range(8, 11)
COMMAND_INDEX = range(11, 14)
DOF_POS_INDEX = range(17, 27)
DOF_VEL_INDEX = range(27, 37)
IN_CONTACT_INDEX = range(37, 39)
AGENT_ACTION_INDEX = range(39, 49)
CONTACT_FORCE_INDEX = range(62, 68)
JOINT_TORQUE_INDEX = range(68, 78)

def plot_vel_tracking():
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))


    time_line = data[:, 0]


    axs[0].plot(time_line, data[:, BASE_LIN_VEL_INDEX[0]], label='$v_x$')
    axs[0].plot(time_line, data[:, COMMAND_INDEX[0]], label='$v_x^d$')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('$V_x$ (m/s)')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(time_line, data[:, BASE_LIN_VEL_INDEX[1]], label='$v_y$')
    axs[1].plot(time_line, data[:, COMMAND_INDEX[1]], label='$v_y^d$')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('$V_y$ (m/s)')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(time_line, data[:, BASE_ANG_VEL_INDEX[2]], label='$\omega_z$')
    axs[2].plot(time_line, data[:, COMMAND_INDEX[2]], label='$\omega_z^d$')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('$\omega_z$ (rad/s)')
    axs[2].grid(True)
    axs[2].legend()

    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_torque():
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))


    time_line = data[25:250, 0]


    axs[0].plot(time_line, data[25:250, JOINT_TORQUE_INDEX[JOINT_INDEX['R_calf_joint']]])
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Torque R_calf (N.m)')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(time_line, data[25:250, JOINT_TORQUE_INDEX[JOINT_INDEX['R_toe_joint']]])
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Torque R_toe (N.m)')
    axs[1].grid(True)
    axs[1].legend()

    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # plot_vel_tracking()
    plot_torque()