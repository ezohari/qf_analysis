from pathlib import Path, PosixPath
from typing import List
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go

CURRENT_DIR = Path('.')
DATA_DIR = CURRENT_DIR / '../data'

wav_pattern = r'(\d+)(_|-|to)(\d+).npy'

def find_wavelength_range_in_filename(file):
    mod_filename = file.name.replace('nm', '')
    res = re.findall(wav_pattern, mod_filename)
    if not res:
        print(file.name)
        return None, None
    if len(res[0]) != 3:
        print(file.name)
        return None, None

    return res[0][0], res[0][-1]


def read_npy_files(data_dir: PosixPath) -> List[PosixPath]:
    all_npy_files = [x for x in data_dir.rglob('*.npy')]

    return all_npy_files


def get_normalized_array(file):
    raw_power_values = np.load(file)
    # normalize raw_power_values
    normalized_power_values = raw_power_values  / raw_power_values.max()
    power_values = normalized_power_values.tolist()[0]
    
    return power_values


def get_wavelength_range(start_wav, end_wav, power_values):
    wavelenght_values = np.arange(start_wav, end_wav, (end_wav - start_wav) / len(power_values)).tolist()

    return wavelenght_values


def plot_simple_figure(power_values, wavelenght_values):


    fig = go.Figure([go.Scatter(x=wavelenght_values, y=power_values)])

    fig.update_layout(
        xaxis_title="Wavelength (nm)",
        yaxis_title="Normalized Power")

    return fig


def get_fig_with_lorentzian_trace(power_values, fitted_power_values, wavelength_values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name="Measured values",
        mode="lines", x=wavelength_values, y=power_values))

    fig.add_trace(go.Scatter(
        name="Fitted Lorentzian",
        mode="lines", x=wavelength_values, y=fitted_power_values))


    fig.update_layout(
        xaxis_title="Wavelength (nm)",
        yaxis_title="Normalized Power")


    return fig

def calculate_qf(fitted_lorentzian_values, wavelenght_values, params):
    x0 = wavelenght_values[np.argmin(fitted_lorentzian_values)]
    gamma = np.abs(params[-1])

    return x0 / gamma