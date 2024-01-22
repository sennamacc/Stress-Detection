import numpy as np # used for arrays
import matplotlib.pyplot as plt # used for plotting / creating visualizations
from pylsl import StreamInlet, resolve_byprop  # used to receive EEG data
import utils
from pynput.keyboard import Key, Controller # used to monitor/control keyboard inputs
from datetime import datetime
# The following libraries are used for controlling the lightbulb
import asyncio
from kasa import Discover, SmartBulb
import colorsys

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3

print('Starting program.')

averaged_stress = 1.5

""" SET UP BULB """
async def find_bulb():
    devices = await Discover.discover()
    for dev in devices.values():
        if isinstance(dev, SmartBulb):
            await dev.update()
            return dev.host
    return None

async def change_bulb_color_hsv(ip, hue, saturation, value):
    bulb = SmartBulb(ip)
    await bulb.update()

    # Ensure the values are within valid ranges
    hue = int(hue) % 360  # Hue value should be between 0 and 360
    saturation = int(saturation) % 100  # Saturation should be between 0 and 100
    value = int(value) % 100  # Value should be between 0 and 100

    await bulb.set_hsv(hue, saturation, value)
    await bulb.update()

async def control_bulb():
    bulb_ip = await find_bulb()
    if bulb_ip:
        saturation = 70
        value = 40
        while True:
            hue = (120 + averaged_stress * 50)
            await change_bulb_color_hsv(bulb_ip, hue, saturation, value)

    else:
        raise RuntimeError('No lightbulbs found on the network.')

""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
BUFFER_LENGTH = 5

# Length of epochs
# Epochs are segment of EEG data
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of which channel(s) (electrodes) being used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]

# How long we are averadging the stress over
AVG_LENGTH = 35

async def main():
    global averaged_stress
    
    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Aquiring data...")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are collected in a second.
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer_le = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state_le = None  # for use with the notch filter
    
    eeg_buffer_lf = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state_lf = None  # for use with the notch filter
    
    eeg_buffer_rf = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state_rf = None  # for use with the notch filter
    
    eeg_buffer_re = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state_re = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer_le = np.zeros((n_win_test, 4))
    band_buffer_lf = np.zeros((n_win_test, 4))
    band_buffer_rf = np.zeros((n_win_test, 4))
    band_buffer_re = np.zeros((n_win_test, 4))


    """ 3. GET DATA """

    print('Press Ctrl-C in the console to break the while loop.')

    print('I will print your overall stress level.')

    print('The higher the number, the more stressed you are.')

    print('The lightbulb will also change colour based on your stress.')

    print('Purple is high stress, green is low stress.')

    overall_stress = np.full(AVG_LENGTH, 1.5)

    asyncio.create_task(control_bulb())

    try:
        keyboard = Controller()
        count=0

        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:
        
            await asyncio.sleep(0.5)

            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            # 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
            # using data from all channels
            if np.array(eeg_data).ndim == 2:
                ch_data_le = np.array(eeg_data)[:, 0]
                ch_data_lf = np.array(eeg_data)[:, 1]
                ch_data_rf = np.array(eeg_data)[:, 2]
                ch_data_re = np.array(eeg_data)[:, 3]

            # Update EEG buffer with the new data
            
            #le
            eeg_buffer_le, filter_state_le = utils.update_buffer(
                eeg_buffer_le, ch_data_le, notch=True,
                filter_state=filter_state_le)
            
            #lf
            eeg_buffer_lf, filter_state_lf = utils.update_buffer(
                eeg_buffer_lf, ch_data_lf, notch=True,
                filter_state=filter_state_lf)

            #rf
            eeg_buffer_rf, filter_state_rf = utils.update_buffer(
                eeg_buffer_rf, ch_data_rf, notch=True,
                filter_state=filter_state_rf)

            #re
            eeg_buffer_re, filter_state_re = utils.update_buffer(
                eeg_buffer_re, ch_data_re, notch=True,
                filter_state=filter_state_re)

            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch_le = utils.get_last_data(eeg_buffer_le,
                                             EPOCH_LENGTH * fs)

            data_epoch_lf = utils.get_last_data(eeg_buffer_lf,
                                             EPOCH_LENGTH * fs)

            data_epoch_rf = utils.get_last_data(eeg_buffer_rf,
                                             EPOCH_LENGTH * fs)

            data_epoch_re = utils.get_last_data(eeg_buffer_re,
                                             EPOCH_LENGTH * fs)

            # Compute band powers
            band_powers_le = utils.compute_band_powers(data_epoch_le, fs)
            band_buffer_le, _ = utils.update_buffer(band_buffer_le,
                                                 np.asarray([band_powers_le]))

            band_powers_lf = utils.compute_band_powers(data_epoch_lf, fs)
            band_buffer_lf, _ = utils.update_buffer(band_buffer_lf,
                                                 np.asarray([band_powers_lf]))

            band_powers_rf = utils.compute_band_powers(data_epoch_rf, fs)
            band_buffer_rf, _ = utils.update_buffer(band_buffer_rf,
                                                 np.asarray([band_powers_rf]))

            band_powers_re = utils.compute_band_powers(data_epoch_re, fs)
            band_buffer_re, _ = utils.update_buffer(band_buffer_re,
                                                 np.asarray([band_powers_re]))
            # Compute the average band powers for all epochs in buffer
            # This helps to smooth out noise
            smooth_band_powers_le = np.mean(band_buffer_le, axis=0)
            smooth_band_powers_lf = np.mean(band_buffer_lf, axis=0)
            smooth_band_powers_rf = np.mean(band_buffer_rf, axis=0)
            smooth_band_powers_re = np.mean(band_buffer_re, axis=0)

            """ 4. DETECTING STRESS """
            # Detecting stress through symmetry

            alpha_symmetry_forehead = (smooth_band_powers_lf[2] / smooth_band_powers_rf[2])

            alpha_symmetry_ear = (smooth_band_powers_le[2] / smooth_band_powers_re[2])

            beta_symmetry_forehead = (smooth_band_powers_lf[3] / smooth_band_powers_rf[3])

            beta_symmetry_ear = (smooth_band_powers_le[3] / smooth_band_powers_re[3])

            # Cleaning up our raw data

            clean_alpha_sym_f = abs(np.log(alpha_symmetry_forehead))
            clean_alpha_sym_e = abs(np.log(alpha_symmetry_ear))
            clean_beta_sym_f = abs(np.log(beta_symmetry_forehead))
            clean_beta_sym_e = abs(np.log(beta_symmetry_ear))

            new_value = (clean_alpha_sym_f + clean_alpha_sym_e + clean_beta_sym_f + clean_beta_sym_e) / 4
            overall_stress = np.append(overall_stress[1:], new_value)
            
            averaged_stress = np.mean(overall_stress)

            print(averaged_stress)
   
        
    except KeyboardInterrupt:
        print('\n Bye')

if __name__ == "__main__":
    asyncio.run(main())
