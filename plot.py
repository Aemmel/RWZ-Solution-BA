import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation

def setup_ax(ax):
    ax.tick_params(direction="in")          # ticks inside
    ax.xaxis.set_ticks_position("both")     # ticks on lower and upper x axis
    ax.yaxis.set_ticks_position("both")     # ticks on left and right y axis
    ax.tick_params(grid_alpha=0.5)          # opacity of grid

DATA_PATH = "data/"
OUTPUT_PATH = "plots/"
ELL = 2                     # ell for perturbation
IV_SIGMA = 1                # sigma for initial value gauss packet
PARITY = "odd"              # odd (axial) or even (polar)
DETECTOR_POS = 280

################################
# detector output
def show_output(log_plot=False, save=False, file_name="detector"):
    output_data = np.transpose(np.array(pd.read_csv(DATA_PATH + "detector_output-pos=280.csv", header=None)))
    fig, ax = plt.subplots()

    # we are only interested in values x > 0
    id_x_greater_0 = np.abs(output_data[1]).argmin()
    output_data = output_data[:, id_x_greater_0:] 

    if log_plot:
        detector_signal = np.log(np.abs(output_data[2]))
    else:
        detector_signal = output_data[2]

    ax.plot(output_data[1], detector_signal)

    setup_ax(ax)

    ax.set_xlabel(r"retarded time $u=t-r_*$")
    ax.set_ylabel(r"$\Psi^{" + PARITY + r"}_{" + str(ELL) + r"0}$")

    ax.set_title(r"Detector output at $r_* = " + str(DETECTOR_POS) + r"$ for IV of $\sigma=" + str(IV_SIGMA) + r"$")

    if save:
        add_log = ""
        if log_plot:
            add_log = "_log"
        plt.savefig(OUTPUT_PATH + file_name + add_log + ".pdf")

    plt.show()

################################
# animate signal
def show_signal(save=False, file_name="signal"):
    fig, ax = plt.subplots()
    signal_data = np.transpose(np.array(pd.read_csv(DATA_PATH + "signal_evolution.csv")))
    p, = plt.plot([], [])

    def animate_pert(i):
        p.set_data(signal_data[0], signal_data[i])
        return p,

    ax.set_xlim((signal_data[0][0], signal_data[0][-1]))
    ax.set_ylim((-0.25, 0.25))

    ax.set_xlabel(r"$r_*$")
    ax.set_ylabel(r"$\Psi^{" + PARITY + r"}_{" + str(ELL) + r"0}$")
    
    ax.set_title(r"Signal evolution with IV of $\sigma=" + str(IV_SIGMA) + r"$")

    ani = animation.FuncAnimation(fig, animate_pert, np.arange(start=1, stop=len(signal_data)), interval=100)
    plt.show()

    if save:
        writer = animation.FFMpegFileWriter()
        ani.save(OUTPUT_PATH + file_name + ".mp4", writer=writer)


# show_signal(save=True, file_name="signal_sig1")
show_output(log_plot=True, save=True, file_name="detector_sig1")