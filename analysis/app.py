import streamlit as st
from pathlib import Path
import time
import numpy as np
from utils import (read_npy_files, get_normalized_array, get_wavelength_range,
                    plot_simple_figure, find_wavelength_range_in_filename,
                    get_fig_with_lorentzian_trace, calculate_qf)
from fit_lorentzian import fit_lorentzian

CURRENT_DIR = Path('.')
DATA_DIR = CURRENT_DIR / '../data'
SAVED_IMAGES = CURRENT_DIR / 'images'
SAVED_IMAGES.mkdir(parents=True, exist_ok=True)

all_npy_files = read_npy_files(DATA_DIR)
filename2file = {file.name:file for file in all_npy_files}

st.title('QF measurement app')
filename = st.selectbox('Select npy file', list(filename2file.keys()))
file=filename2file[filename]
st.write('You selected:', file)

power_values = get_normalized_array(file)

st.warning('We detected these value in the filename as the start and end. If wrong, please correct them')
start_wav, end_wav = find_wavelength_range_in_filename(file)
start_wav_inp = st.text_input('Start wavelength', start_wav if start_wav else 'NOT FOUND')
end_wav_inp = st.text_input('End wavelength', end_wav if end_wav else 'NOT FOUND')



start_wav, end_wav = int(start_wav_inp), int(end_wav_inp)
wavelength_values = get_wavelength_range(start_wav, end_wav, power_values)

get_downsampled_rate = st.text_input('Downsample Rate', '1')
donwsampled_rate = int(get_downsampled_rate)
power_values = power_values[::donwsampled_rate]
wavelength_values = wavelength_values[::donwsampled_rate]

st.info(len(power_values))
plot_orig_fig = st.button('Plot original figure')


if plot_orig_fig:
    fig = plot_simple_figure(power_values, wavelength_values)    
    st.write(fig)



# st.header('Fitting Lorentzian to cuvrves')
chunk_start_wav_inp = st.text_input('Input selected chunk start wavelength', '')
chunk_end_wav_inp = st.text_input('Input selected chunk end wavelength', '')

chunk_power_values, chunk_wavelength_values = None, None
if chunk_start_wav_inp and chunk_end_wav_inp:
    chunk_start_wav, chunk_end_wav = int(chunk_start_wav_inp), int(chunk_end_wav_inp)

    chunk_start_index = np.abs(np.array(wavelength_values)-chunk_start_wav).argmin() 
    chunk_end_index = np.abs(np.array(wavelength_values)-chunk_end_wav).argmin()

    chunk_power_values = power_values[chunk_start_index:chunk_end_index]
    chunk_wavelength_values = wavelength_values[chunk_start_index:chunk_end_index]

plot_chunk_fig = st.button('Plot chunk figure')

if plot_chunk_fig:
    fig = plot_simple_figure(chunk_power_values, chunk_wavelength_values)    
    st.write(fig)

plot_lorentzian = st.button('Fit Lorentzian')

if plot_lorentzian:
    st.info('Fitting Lorentzian to the selected curve')
    fitted_curve, params = fit_lorentzian(chunk_power_values, chunk_wavelength_values)
    st.balloons()

    lorentzoian_fig = get_fig_with_lorentzian_trace(chunk_power_values, fitted_curve, chunk_wavelength_values)
    qf = calculate_qf(fitted_curve,chunk_wavelength_values,params)
    qf_y = 0.8
    qf_x = int(chunk_wavelength_values[0] + (0.95)*(chunk_wavelength_values[-1] - chunk_wavelength_values[0]))
    lorentzoian_fig.add_annotation(x=qf_x, y=qf_y,
                text=f'<b>QF={int(qf)}</b>',
                showarrow=False,
                arrowhead=1)

    st.write(lorentzoian_fig)

    RES_DIR = SAVED_IMAGES / filename
    RES_DIR.mkdir(parents=True, exist_ok=True)
    image_timestamp = time.strftime("%Y%m%d-%H%M%S")
    st.info(RES_DIR.absolute().as_posix()+f'/{image_timestamp}.png')
    lorentzoian_fig.write_image(RES_DIR.absolute().as_posix()+f'/{image_timestamp}.png')


