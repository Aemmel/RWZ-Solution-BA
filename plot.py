import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import json
from scipy import stats
from scipy.optimize import curve_fit

def setup_ax(ax):
    ax.tick_params(direction="in")          # ticks inside
    ax.xaxis.set_ticks_position("both")     # ticks on lower and upper x axis
    ax.yaxis.set_ticks_position("both")     # ticks on left and right y axis
    ax.tick_params(grid_alpha=0.5)          # opacity of grid

def find_max(t, arr):
    if len(arr) != len(t):
        print("find_max, wrong dimensions")
        return -1

    maxima_x = []
    maxima_y = []
    for i in range(1, len(arr)-1):
        if arr[i-1] < arr[i] and arr[i+1] <= arr[i]:
            maxima_x.append(t[i])
            maxima_y.append(arr[i])
    
    return maxima_x, maxima_y

def fit_ln_hyperbola(x, a, b):
    return -a*np.log(x) + b

with open("params.json", "r") as json_file:
    data = json_file.read()
obj = json.loads(data)

DATA_PATH = "data/"
OUTPUT_PATH = "plots/"
ELL = obj["ell"]                        # ell for perturbation
IV_SIGMA = obj["gauss_sigma"]           # sigma for initial value gauss packet
PARITY = obj["parity"]                  # odd (axial) or even (polar)
DETECTOR_POS = obj["detector_pos"]      # position of detector
MASS = obj["mass"]

################################
# detector output
def show_output(log_plot=False, save=False, file_name="detector"):
    output_data = np.transpose(np.array(pd.read_csv(DATA_PATH + "detector_output-pos=280.csv", header=None)))
    fig, ax = plt.subplots(figsize=(5.6,4))

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

    if log_plot:
        ax.set_ylabel(r"$\log(\Psi^{" + PARITY + r"}_{" + str(ELL) + r"0})$")
    else:
        x_lim=(0, 250)
        ax.set_xlim(x_lim)
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
    # find the y limit dynamically
    limit = 0
    for i in range(1, len(signal_data)):
        temp_max = signal_data[i][np.abs(signal_data[i]).argmax()] # argmax returns index position
        if temp_max > limit:
            limit = temp_max 
    ax.set_ylim((-limit, limit))

    ax.set_xlabel(r"$r_*$")
    ax.set_ylabel(r"$\Psi^{" + PARITY + r"}_{" + str(ELL) + r"0}$")
    
    ax.set_title(r"Signal evolution with IV of $\sigma=" + str(IV_SIGMA) + r"$")

    ani = animation.FuncAnimation(fig, animate_pert, np.arange(start=1, stop=len(signal_data)), interval=80)
    plt.show()

    if save:
        writer = animation.FFMpegFileWriter()
        ani.save(OUTPUT_PATH + file_name + ".mp4", writer=writer)

################################
# find the QNMs
def QNMs(save=False, file_name="QNM"):
    output_data = np.transpose(np.array(pd.read_csv(DATA_PATH + "detector_output-pos=280.csv", header=None)))
    fig, ax = plt.subplots(figsize=(5.6,4))

    #fig.set_size((10,1))

    # we are only interested in values x > 0
    id_x_greater_0 = np.abs(output_data[1]).argmin()
    output_data = output_data[:, id_x_greater_0:] 

    # output_data = np.zeros((3, 400))
    # output_data[1] = np.linspace(0, 100, 400)
    # output_data[2] = np.sin(0.8*output_data[1])*np.exp(-0.5*output_data[1])

    detector_signal = np.log(np.abs(output_data[2]))


    max_x, max_y = find_max(output_data[1], detector_signal)
    min_x, min_y = find_max(output_data[1], -detector_signal)
    min_y = -1 * np.array(min_y)

    # manually set
    max_i_min = 7
    max_i_max = len(max_x) - 24
    min_i_min = 7
    min_i_max = len(min_x) - 24
    
    # caluclate the QNM
    # frequency
    freq = 0
    for i in range(min_i_min+1, min_i_max):
        freq += np.abs(min_x[i] - min_x[i-1])
    freq /= min_i_max-min_i_min
    freq = 1 / freq  * np.pi # omega not f, but divided by two (*2 * pi / 2) because we calculated double the frequency (since we measured every pass through 0, not every second pass)

    # dampening
    slope, intercept, r_value, p_value, std_err = stats.linregress(max_x[max_i_min:max_i_max], max_y[max_i_min:max_i_max])

    print(str(freq) + " + " + str(slope) + "*i")

    # maximums
    slope_x = np.linspace(max_x[max_i_min]-5, max_x[max_i_max-1]+5, 100)
    slope_y = intercept + slope*slope_x

    # plot the result
    ax.plot(output_data[1], detector_signal, label="signal")
    ax.scatter(max_x[max_i_min:max_i_max], max_y[max_i_min:max_i_max], color="tab:orange", label="$\omega_I=%.6f$"%np.abs(slope), s=15) # we measure -w_I, not w_I
    ax.scatter(min_x[min_i_min:min_i_max], min_y[min_i_min:min_i_max], color="tab:green", label="$\omega_R=%.6f$"%freq, s=20)
    ax.plot(slope_x, slope_y)

    # plot limit
    x_lim = (140, 230)
    y_lim = (-15.5, -1.2)

    # draw in lines showing spacing between downwar peaks (for frequency)
    line_pos_y = y_lim[0]+1
    ax.hlines(line_pos_y, min_x[min_i_min], min_x[min_i_max-1], linestyles="solid", color="tab:green")
    for i in range(min_i_min, min_i_max):
        ax.vlines(min_x[i], min_y[i], line_pos_y, linestyles="dashed", color="tab:green")

    setup_ax(ax)

    ax.set_xlabel(r"retarded time $u=t-r_*$")

    ax.set_ylabel(r"$\ln(|\Psi^{" + PARITY + r"}_{" + str(ELL) + r"0}|)$")

    ax.set_title(r"Detector output at $r_* = " + str(DETECTOR_POS) + r"$ for IV of $\sigma=" + str(IV_SIGMA) + r"$")

    ax.legend(loc="best")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save:
        plt.savefig(OUTPUT_PATH + file_name + ".pdf")

    plt.show()

################################
# tail
def tail(save=False, file_name="tail"):
    output_data = np.transpose(np.array(pd.read_csv(DATA_PATH + "detector_output-pos=280.csv", header=None)))
    fig, ax = plt.subplots(figsize=(5.5,4)) #7.5,4

    # we are only interested in values x > 0
    id_x_greater_0 = np.abs(output_data[1]).argmin()
    output_data = output_data[:, id_x_greater_0:] 

    detector_signal = np.log(np.abs(output_data[2]))

    ax.plot(output_data[1], detector_signal, label="complete signal $\Psi$")
    tail_start = 6500
    tail_end = -1
    # ax.scatter(output_data[1][tail_start:tail_end], detector_signal[tail_start:tail_end], color="tab:orange")

    # fit tail
    # pars, cov = curve_fit(fit_ln_hyperbola, output_data[1][tail_start:tail_end], detector_signal[tail_start:tail_end], p0=[4.1,11.1])
    # time_fit = np.linspace(output_data[1][tail_start], output_data[1][tail_end], 500)
    # ax.plot(time_fit, fit_ln_hyperbola(time_fit, pars[0], pars[1]), linestyle=(0, (5, 5)), alpha=0.8, label="tail fit $\Psi\sim u^{-%.2f}$"%pars[0])

    setup_ax(ax)

    ax.set_xlabel(r"retarded time $u=t-r_*$")
    ax.set_ylabel(r"$\log(\Psi^{" + PARITY + r"}_{" + str(ELL) + r"0})$")
    ax.set_title(r"Detector output at $r_* = " + str(DETECTOR_POS) + r"$ for IV of $\sigma=" + str(IV_SIGMA) + r"$")

    ax.legend(loc="best")

    if save:
        plt.savefig(OUTPUT_PATH + file_name + ".pdf")

    plt.show()

file_name = "ell=" + str(ELL)
file_name += "_"
file_name += PARITY
file_name += "_"
file_name += "sig=" + str(IV_SIGMA)

save_all = True
#save_all = False

#show_signal(save=False, file_name="signal_" + file_name)
#show_output(log_plot=True, save=False, file_name="detector_" + file_name)
show_output(log_plot=False, save=True, file_name="N_x=5e4" + file_name)
#QNMs(save=True, file_name="QNM_" + file_name)
#tail(save=True, file_name="BC_tail_" + file_name)