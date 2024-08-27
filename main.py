import pyabf
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal

from scipy.signal import butter, iirnotch, lfilter, filtfilt, sosfiltfilt, hilbert

import numpy as np

def thresholding(array, th_val,thresh_wind):
    i = 0
    j = thresh_wind
    thresh = []

    while (True):

        go_in = []
        go_in = array[i:j]

        thresh_value = th_val
        thresh.append(np.mean(go_in) + (np.std(go_in) * thresh_value) )

        if j == len(array):
            break

        j = j + 1
        i = i + 1

        # print("what left \t", len(array) - j)

    return thresh

def first_point(array, thresh, thresh_wind):

    th = []
    st = 0


    for j in range (thresh_wind, len(array)):

        if (array[j] >= thresh[j]):

            # print("true start")

            st = j - 1
            th = thresh[j-1]
            print("thresh equals to\t", thresh[j])
            # x_axis.append(j)
            break

        # else:
        #
        #     break

    # start = st
    return st, th
    # end = find_region(STD, th, start)

def find_region(array, thresh, start):

    array = array[start+1:len(array)]

    for j in range(len(array)):

        if (array[j] < thresh):

            # y_axis.append(j)

            break

    return j

def oscillation_input(peaks):

    midle = int(len(peaks) / 2)
    stt = peaks[:midle]
    enn = peaks[midle:len(peaks)]

    start_t = []
    end_t = []

    for i in range(len(stt)):

        if (stt[i] < peaks[-1]):
            start_t.append(stt[i])

    for i in range(len(enn)):

        if (enn[i] < peaks[-1]):
            end_t.append(enn[i])

    return  start_t, end_t

from shapely.geometry import LineString

# matplotlib.use('TkAgg')
path = "C:\\Users\\Manef\\OneDrive - Aix-Marseille Université\\data\\jean-charles\\Manef\\14-02-04 - A - WT - KCC2 - P0\\"

ID = "2014_02_04_0000"

# id = "non-"+ ID
# # == load the epileptic patient
# with open(path + id + ".txt", 'r') as file1:
#     non = [float(i) for line in file1 for i in line.split('\n') if i.strip()]
#
# id = "mixte-" + ID
# # == load the epileptic patient
# with open(path + id + ".txt", 'r') as file1:
#     mixte = [float(i) for line in file1 for i in line.split('\n') if i.strip()]
#
# id = "loco-" + ID
# # == load the epileptic patient
# with open(path + id + ".txt", 'r') as file1:
#     loco = [float(i) for line in file1 for i in line.split('\n') if i.strip()]

path = "C:\\Users\\Manef\\OneDrive - Aix-Marseille Université\\data\\jean-charles\\Manef\\14-02-04 - B - WT - KCC2 - P0\\2014_02_04_0003.abf"

# path = "/home/ftay/data/jean-charle/14-02-14 - B - WT - KCC2 - P1/2014_02_14_0008.abf"
# path = "/home/ftay/data/jean-charle/14-02-14 - A - WT - KCC2 - P1/2014_02_14_0002.abf"
# path = "/home/ftay/data/jean-charle/14-02-04 - B - WT - KCC2 - P0/2014_02_04_0004.abf"
# path = "/home/ftay/data/jean-charle/14-02-04 - A - WT - KCC2 - P0/2014_02_04_0001.abf"

abf = pyabf.ABF(path)
last = 9
chanel = 7


STD_wind = 1000
th_val = 2

thresh_wind = 10
thresh_fft = 50

region = 1

abf.setSweep(sweepNumber= last, channel= chanel)
# print(abf.sweepY) # displays sweep data (ADC)
# print(abf.sweepX) # displays sweep times (seconds)
# print(abf.sweepC) # displays command waveform (DAC)
signal_xx = []
axis = []
# plt.plot(abf.sweepX, abf.sweepY)
# plt.show()

for i in range (last):
    print("i\t", i)
    # abf = pyabf.ABF("E:\\data\\jean-charles\\test3.abf")
    # abf.setSweep(sweepNumber=i, channel=0)
    # abf = pyabf.ABF(path)
    abf.setSweep(sweepNumber=i, channel=chanel)

    signal_xx.append(abf.sweepY)
    # print("length of signal\t", len(signal))

    axis.append(abf.sweepX)
    # print("length of axis\t", len(axis))

print("end of loop")

sig = np.array(signal_xx).flatten()



# start_non, end_non = oscillation_input(non)
# start_mixte, end_mixte = oscillation_input(mixte)
# start_loco, end_loco = oscillation_input(loco)

x = sig


fs = 5000
STD = []
# plt.plot(x)
print("computing tyhe STD")


i = 0
j = STD_wind
window = j
STD = []

while (True):

    go_in = []
    go_in = sig[i:j]

    std = np.std(go_in)

    STD.append(std)
    j = j + window
    i = i + window

    if j > len(sig):
        break



thresh = thresholding(STD, th_val, thresh_wind)

"here we need to make the STD array and the arr (which poresnet the threshold container equal by adding the same value 6 other times in the beging of the arr array"

for i in range( thresh_wind - 1):

    thresh.insert(0, thresh[0])

arr = thresh
# i = 0
# arr = []
# arr1_n = []

x = 0
# while (True):
#
#     if (x == 0):
#         for i in range(thresh_wind):
#             arr.append(thresh[x])
#             x = x + 1
#
#     else:
#         arr.append(thresh[x])
#
#         x = x + 1
#
#     if (x == len(STD)):
#         break



tt_healthy = []
first = 0
b = 0

while b <= len(arr) - 1:
    if (b == 0):
        tt_healthy.append(first)
        b = b + 1
    else:
        first = first + 1
        tt_healthy.append(first)
        b = b + 1



fig, axs = plt.subplots(2, 1)
axs[0].plot(sig)
axs[1].plot(STD)
axs[0].grid(True)
axs[1].grid(True)

axs[1].plot(tt_healthy, arr, 'r+-')

plt.show()

# =====================
"""
In this part, the goal is to find where the curves are above the threshold values: when it start to get higher and
when it get nornal ( normal oscvillations)

"""

x_axis = []
y_axis = []

STD1_xx = STD

while True:

    start, th= first_point(STD, arr, thresh_wind)
    print("start equals to\t", start)
    if (start > 1):

        end = find_region(STD, th, start)
        end = end + 1


        print("working on\t", start + end)

        # print("plotting")
        # fig, axs = plt.subplots(2, 1)
        # axs[0].plot(sig)
        # axs[1].plot(STD, label= "STD")
        # axs[0].grid(True)
        # axs[1].grid(True)
        # axs[1].plot( arr, label= "Threshold")
        # plt.show()

        x_axis.append(start)
        y_axis.append(end)

        "in this part w need to take out the region detected and the threshold associated to this region"
        STD1 = STD[0:start]
        STD2 = STD[start + end + 1:len(STD)]
        STD = STD1
        STD = np.append(STD, STD2)
        STD = STD.ravel()

        arr1 = arr[0:start]
        arr2 = arr[start + end + 1:len(arr)]
        arr = arr1
        arr = np.append(arr, arr2)
        arr = arr.ravel()

        # n = start + end + 1
        # tt_healthy = tt_healthy[:len(tt_healthy) - n]

        print("STD", len(STD))
        print("arr", len(arr))
        # print("tt_healthy", len(tt_healthy))

    else:
        print("from where to start cutting: \t", x_axis)
        print("length of how much to cut: \t", y_axis)
        break

    if((start+end+1) >= len(STD)):

        break


plt.show()
#---------------------------------------------------------------------------
# == preprae to plot

plt.show()
xx = x_axis
yy = []
for i in range(len(x_axis)):

    if (i > 0):

        arry = y_axis[0:i]
        sum = np.sum(arry) +i
        x_axis[i] = x_axis[i] + sum

print("x_axis\t", x_axis)
print("y_axis\t", y_axis)

ttt= []
first = 0
b = 0

while b <= len(sig) - 1:
    if (b ==0):
        ttt.append(first)
        b = b + 1
    else:
        first = first + 1
        ttt.append(first)
        b = b + 1

print("plotting")
# == the original plot: the original input signal and the STD of the signal used for the detection
# fig, axs = plt.subplots(2, 1)
# axs[0].plot( sig)
# axs[1].plot(STD1_xx)
# axs[0].grid(True)
# axs[1].grid(True)
#
# for i in range(len(x_axis)):
#     axs[1].axvline(x_axis[i], color='g', linestyle='--')
#     axs[1].axvline((x_axis[i] + y_axis[i]), color='g', linestyle='--')
#     # axs[1].text(x_axis[i], (x_axis[i] + y_axis[i]), "oscillation", style='italic', bbox={'facecolor': 'green', 'alpha': 0.5, 'pad': 10})
#     axs[1].text(x_axis[i], 3, i, style='italic')
#
# for i in range(len(x_axis)):
#     axs[0].axvline(x_axis[i] * window, color='g',label='own detection', linestyle='--')
#     axs[0].axvline(((x_axis[i] + y_axis[i]) * window), label='own detection', color='g', linestyle='--')
#     axs[0].text(x_axis[i]* window, 0.5, i, style='italic')
#
# for i in range(len(start_t)):
#     axs[0].axvline(start_t[i], color='magenta', label='locomotrice', linestyle='--')
#
# for i in range(len(end_t)):
#     axs[0].axvline((end_t[i]), color='magenta',label='locomotrice', linestyle='--')
#     # axs[0].text(x_axis[i]* window, 0.5, i, style='italic')
#
# axs[1].plot(tt_healthy, arr, color='blue', linestyle='--')

# result = [item / 5000 for item in x_axis]


# == just the detectiopn on the orioginal signal

#
# # ===================================================
# # == non locomotrice motrice regions plotting
#
# for i in range(len(start_non)):
#     axs.axvline(start_non[i], color='lightcoral', label='locomotrice', linestyle='--')
#     axs.text(start_non[i], 0.5, i, style='italic', color="lightcoral")
#
# for i in range(len(end_non)):
#     axs.axvline((end_non[i]), color='lightcoral',label='locomotrice', linestyle='--')
#
#
# # ===================================================
# # == Mixte regions plotting
#
#
# for i in range(len(start_mixte)):
#     axs.axvline(start_mixte[i], color='r', label='locomotrice', linestyle='--')
#     axs.text(start_mixte[i], 0.5, i, style='italic', color="r")
#
# for i in range(len(end_mixte)):
#     axs.axvline((end_mixte[i]), color='r',label='locomotrice', linestyle='--')

# ===================================================
# == locomotrice regions plotting

# for i in range(len(start_loco)):
#     axs.axvline(start_loco[i], color='lightsalmon', label='locomotrice', linestyle='--')
#     axs.text(start_loco[i], 0.5, i, style='italic', color="lightsalmon")
#
# for i in range(len(end_loco)):
#     axs.axvline((end_loco[i]), color='lightsalmon',label='locomotrice', linestyle='--')


print("regions\t", x_axis)
# plt.show()

# region = [3, 7, 10, 11, 15, 22, 24]
# import librosa

# for i in range(len(region)):
#     to_test = sig[(x_axis[region[i]] * window): ((x_axis[region[i]] + y_axis[region[i]]) * window)]
#     flatness = librosa.feature.spectral_flatness(y=to_test, win_length = len(to_test))
#     print("flatness\t", flatness)

# import librosa
# import numpy as np
#
# region = 0
# to_test = sig[(x_axis[region] * window): ((x_axis[region] + y_axis[region]) * window)]
#
# def spectral_flatness(signal, n_fft=512, hop_length=256):
#     # Calculate the power spectrum of the signal
#     power_spectrum = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))**2
#
#     # Compute the geometric mean of the power spectrum
#     geometric_mean = np.exp(np.mean(np.log(np.maximum(power_spectrum, 1e-10)), axis=0))
#
#     # Compute the arithmetic mean of the power spectrum
#     arithmetic_mean = np.mean(power_spectrum, axis=0)
#
#     # Compute the spectral flatness
#     spectral_flatness = geometric_mean / arithmetic_mean
#
#     return spectral_flatness
#
# # Example usage:
# # Load an audio file (replace 'your_audio_file.wav' with the path to your audio file)
# audio_file = 'your_audio_file.wav'
# signal, sr = librosa.load(to_test, sr=None)
#
# # Compute the spectral flatness of the signal
# flatness = spectral_flatness(signal)
#
# print("Spectral Flatness:", flatness)
# print("done")

# region = [6, 7, 9, 13, 15, 19]
# rest = []
# for i in range(len(region)):
#     to_test = sig[(x_axis[region[i]] * window): ((x_axis[region[i]] + y_axis[region[i]]) * window)]
#     area = np.trapz(to_test)
#     rest.append(area)
#
# print(rest)
#
# region = [5, 11, 17]
# rest = []
# for i in range(len(region)):
#     to_test = sig[(x_axis[region[i]] * window): ((x_axis[region[i]] + y_axis[region[i]]) * window)]
#     area = np.trapz(to_test)
#     rest.append(area)
#
# print(rest)


# region = 9

# == computing the aria under curve

# fft_result = np.fft.fft(to_test)
# freq = np.fft.fftfreq(len(to_test), to_test[1] - to_test[0])
#
# plt.figure(figsize=(10, 5))
# plt.plot(freq, np.abs(fft_result))
# plt.title('Frequency Spectrum')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()

def FFT_loc_class(array):
    qq = False
    X = np.fft.fft(array)

    N = len(array)
    n = np.arange(N)
    T = N / fs
    freq = n / T

    n_oneside = N // 2

    f_oneside = freq[1:n_oneside]

    tt = np.abs(X[1:n_oneside])

    # from scipy.interpolate import interp1d
    # linear_interpolator = interp1d(f_oneside, tt, kind='linear')
    # x_value = 0.5
    # y_value = linear_interpolator(x_value)
    # print(f"The interpolated value of the curve at x = {x_value} is {y_value}")

    # if (f_oneside[0] > 1):
    #     qq = True
    if(tt[0] > 790):
        qq = True


    return qq
def FFT_noise_class(array, thresh_fft):
    qq = True
    X = np.fft.fft(array)

    N = len(array)
    n = np.arange(N)
    T = N / fs
    freq = n / T

    n_oneside = N // 2
    # f_oneside = freq[:n_oneside]

    tt = np.abs(X[1:n_oneside])

    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(array)
    # axs[0].grid(True)
    # axs[0].grid(True)
    #
    # axs[1].plot(tt, 'b')
    # # axs[1].plot(xx, pl, 'ro')
    # axs[1].grid(True)
    # axs[1].grid(True)
    # axs[1].set_xlabel('Freq (Hz)')
    # axs[1].set_ylabel('FFT Amplitude |X(freq)|')
    # axs[1].axhline(50, color='g', linestyle='--')
    #
    #
    # plt.show()

    for i in range(len(tt)):

        if(tt[i] >= thresh_fft):
            qq = False
            break


    return qq

label = []
loc = []
for i in range (len(x_axis)):

    to_test = sig[(x_axis[i] * window): ((x_axis[i] + y_axis[i]) * window)]

    label.append(FFT_noise_class(to_test, thresh_fft))
    loc.append(FFT_loc_class(to_test))




fig, axs = plt.subplots()
axs.plot(sig)
axs.grid(True)

for i in range(len(x_axis)):


    if (label[i] == True):
        axs.axvline(x_axis[i] * window, color='r', label='own detection', linestyle='--')
        axs.axvline(((x_axis[i] + y_axis[i]) * window), label='own detection', color='r', linestyle='--')
        axs.text(x_axis[i] * window, 0.5, str(i) + " Noise", style='italic')

    #elif (non_loc[i] == True):
    if(loc[i] == True):
        axs.axvline(x_axis[i] * window, color='b', label='own detection', linestyle='--')
        axs.axvline(((x_axis[i] + y_axis[i]) * window), label='own detection', color='b', linestyle='--')
        axs.text(x_axis[i] * window, 0.5, str(i) + " non-loco", style='italic')

    else:
        axs.axvline(x_axis[i] * window, color='g', label='own detection', linestyle='--')
        axs.axvline(((x_axis[i] + y_axis[i]) * window), label='own detection', color='g', linestyle='--')
        axs.text(x_axis[i] * window, 0.5, i, style='italic')

plt.show()


    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(array)
    # axs[0].grid(True)
    # axs[0].grid(True)
    #
    # axs[1].plot(f_oneside, np.abs(X[1:n_oneside]), 'b')
    # axs[1].grid(True)
    # axs[1].grid(True)
    # axs[1].set_xlabel('Freq (Hz)')
    # axs[1].set_ylabel('FFT Amplitude |X(freq)|')
    #
    # axs[1].set_xlim(0, 5)
    # axs[1].set_ylim(0, 500)
    #
    # axs[1].axhline(200, color='g', linestyle='--')
    # axs[1].axvline(1.5, color='b', linestyle='--')



# res_test = FFT_class(to_test)


import numpy as np
import matplotlib.pyplot as plt


# plt.plot(to_test)
# plt.plot(y_smooth, color='red')
# plt.show()

# to_test = y_smooth
# == trying the FFT
tt = [1, 7, 12, 16, 19, 44, 52,53, 56, 59, 73, 85, 83, 88, 94, 102, 106, 113, 114, 123, 125]
ttt = ["loc","Mixte", "Mixte", "non loc", "Mixte", "loc", "Mixte", "Mixte", "non loc", "loc", "Mixte", "non loc", "loc", "non loc","non loc","non loc", "Mixte", "non loc", "Mixte", "Mixte", "non loc"]
"loc"
"Mixte"
"non loc"

# for i in range (len(tt)):
#
#     to_test = sig[(x_axis[tt[i]] * window): ((x_axis[tt[i]] + y_axis[tt[i]]) * window)]
#
#     X = np.fft.fft(to_test)
#
#     N = len(to_test)
#     n = np.arange(N)
#     T = N / fs
#     freq = n / T
#
#     n_oneside = N // 2
#
#     f_oneside = freq[1:n_oneside]
#
#     # freq = freq[1:n_oneside]

#
#     fig, axs = plt.subplots(2, 1)
#
#     # plt.rcParams["figure.figsize"] = (30,30)
#     axs[0].set_title('region number  ' + str(tt[i]) + " " + ttt[i] )
#
#     axs[0].plot( to_test)
#     axs[0].grid(True)
#     axs[0].grid(True)
#
#     axs[1].plot(f_oneside, np.abs(X[1:n_oneside]), 'b')
#
#     axs[1].grid(True)
#     axs[1].grid(True)
#     axs[1].set_xlabel('Freq (Hz)')
#     axs[1].set_ylabel('FFT Amplitude |X(freq)|')
#     axs[1].set_xlim(0, 15)
#     axs[1].set_ylim(0, 500)
    # plt.savefig('C:\\Users\\Manef\\OneDrive - Aix-Marseille Université\\data\\jean-charles\\oscillation\\'+ str(i) + ".png")
    # axs[1].axhline(200, color='g', linestyle='--')
    # axs[1].axvline(1.5, color='b', linestyle='--')



# plt.show()
region  = 82
to_test = sig[(x_axis[region] * window): ((x_axis[region] + y_axis[region]) * window)]

X = np.fft.fft(to_test)

N = len(to_test)
n = np.arange(N)
T = N / fs
freq = n / T

n_oneside = N // 2

te = np.abs(X[1:n_oneside])
freq = freq[:n_oneside]

# y_val = 0.5
# x_interp = np.interp(y_val, f_oneside,  freq)
# # x_interp = np.interp(y_vals, y, x)
# import scipy
# sig = np.abs(X[1:n_oneside])
# xx = scipy.signal.find_peaks(sig)

# from scipy.signal import find_peaks
# peaks2, _= find_peaks(te, prominence=50)
#
# fig, axs = plt.subplots(2, 1)
# axs[0].plot(to_test)
# axs[1].plot(te)
# axs[1].plot(peaks2, te[peaks2], "ob")
# axs[1].legend(['prominence'])
# axs[0].grid(True)
# axs[1].grid(True)
#
# axs[1].plot(peaks2, te[peaks2], "ob"); plt.plot(x); plt.legend(['prominence'])
#
# axs[1].set_xlim(0,15)
# axs[1].set_ylim(0,500)
# plt.show()
from scipy import signal

# from scipy.interpolate import CubicSpline
# cs = CubicSpline(freq,te,bc_type='natural')
# to_show =
# f = CubicSpline(freq, te, bc_type='natural')
# x_new = np.linspace(min(freq), max(freq), 200)
# y_new = f(x_new)

# plt.plot(cs.x, 'r')
# plt.scatter(freq, te)
# plt.title('Cubic Spline Interpolation')
# plt.show()

# fig, axs = plt.subplots(2, 1)
# #
# axs[0].plot(freq, te)
# axs[0].grid(True)
# axs[1].plot(cs, 'b')
# # axs[1].plot(xx, pl, 'ro')
# axs[1].grid(True)
#
# axs[1].set_xlim(0,5)
# axs[1].set_ylim(0,500)


fig, axs = plt.subplots(2, 1)
axs[0].plot(to_test)
axs[0].grid(True)

axs[1].plot(freq, np.abs(X[:n_oneside]), 'b')
# axs[1].plot(xx, pl, 'ro')
axs[1].grid(True)
axs[1].set_xlabel('Freq (Hz)')
axs[1].set_ylabel('FFT Amplitude |X(freq)|')
# axs[1].axhline(50, color='g', linestyle='--')
axs[1].set_xlim(0,5)
axs[1].set_ylim(0,500)

axs[1].axhline(100, color='g', linestyle='--')
axs[1].axvline(1, color='b', linestyle='--')
# axs[1].plot(x_interp, y_val, 'o', color='k')
# import numpy as np
# from vector import vector, plot_peaks
# from libs import detect_peaks
# print('Detect peaks with minimum height and distance filters.')
# indexes = detect_peaks.detect_peaks((np.abs(X[:N//300])), mph=7, mpd=2)
# print('Peaks are: %s' % (indexes))


# if intersection.geom_type == 'MultiPoint':
#     axs[1].plot(*LineString(intersection).xy, 'o')
# elif intersection.geom_type == 'Point':
#     axs[1].plot(*intersection.xy, 'o')
plt.show()

