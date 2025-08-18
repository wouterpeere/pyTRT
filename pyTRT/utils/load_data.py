"""
This file contains the base class for the loading and storing of the pyTRT measurements.
"""
from typing import Union

import warnings

import pandas as pd
import numpy as np


class TRTData:

    def __init__(self, file: Union[pd.DataFrame, str, bytes], col_time: str, col_temp_avg: str = None,
                 col_temp_in: str = None,
                 col_temp_out: str = None, col_power: str = None, average_power: float = None,
                 undisturbed_ground: float = None, start_index: int = 0, **kwargs):
        """
        This function loads and stores the pyTRT data from a csv file.
        This csv file should contain header values and columns for at least the timestamps and fluid temperature
        (either provided by an average fluid temperature or both inlet and outlet fluid temperatures).

        Parameters
        ----------
        file : str, pandas.DataFram, bytes
            Location of the csv file (or the file itself) with the pyTRT measurements.
        col_time : str
            Name of the column header with the time values in seconds.
        col_temp_avg : str
            Name of the column header with the average fluid temperatures in degrees Celsius.
        col_temp_in : str
            Name of the column header with the borehole inlet fluid temperatures in degrees Celsius.
        col_temp_out : str
            Name of the column header with the borehole outlet fluid temperatures in degrees Celsius.
        col_power : str
            Name of the column header with the power values in Watt.
        average_power : float
            The average power injected in the borehole during the pyTRT in Watt. When provided, this will overwrite
            any power measurements that are provided in the csv file.
        start_index : int
            The start index at which the steady state conditions of the borehole begins. (Default 0)
        undisturbed_ground : float
            The undisturbed ground temperature in degrees Celsius.
            When not provided, the first average fluid temperature is used.
        kwargs
            The seperator and decimal point can be provided as respectively 'sep' and 'decimal'
        """

        self._time_array: np.ndarray = np.array([])
        self._temperature_array: np.ndarray = np.array([])
        self._power_array: np.ndarray = np.array([])

        self.load_trt_data(file, col_time, col_temp_avg, col_temp_in, col_temp_out, col_power, **kwargs)

        if start_index < 0 or start_index > len(self._time_array):
            raise ValueError('Please provide a valid start index.')

        self.start_index = start_index

        if len(self._power_array) == 0 and average_power is None:
            raise ValueError('Please provide either the power in the csv file or use an average power.')

        self._avg_power = average_power

        if len(self._power_array) != 0 and average_power is not None:
            warnings.warn('The power array will be overwritten with the average power.')
        self.undisturbed_ground_temperature: float = undisturbed_ground if undisturbed_ground is not None else \
            self._temperature_array[0]

    def load_trt_data(self, file: Union[pd.DataFrame, str, bytes], col_time: str, col_temp_avg: str = None,
                      col_temp_in: str = None,
                      col_temp_out: str = None, col_power: str = None, **kwargs) -> None:
        """
        This function loads the pyTRT data. Either the average fluid temperatures should be provided or both the
        inlet and outlet fluid temperatures.

        Parameters
        ----------
        file : str, pandas.DataFram, bytes
            Location of the csv file (or the file itself) with the pyTRT measurements.
        col_time : str
            Name of the column header with the time values in seconds.
        col_temp_avg : str
            Name of the column header with the average fluid temperatures in degrees Celsius.
        col_temp_in : str
            Name of the column header with the borehole inlet fluid temperatures in degrees Celsius.
        col_temp_out : str
            Name of the column header with the borehole outlet fluid temperatures in degrees Celsius.
        col_power : str
            Name of the column header with the power values in Watt.
        kwargs
            The seperator and decimal point can be provided as respectively 'sep' and 'decimal'. Default is ';' and '.'

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the temperature data is not provided in an accurate way.
        """

        if (col_temp_in is None and col_temp_out is not None) or (col_temp_out is None and col_temp_in is not None):
            raise ValueError('Both the column for the temperature in and temperature out measurements should be'
                             'provided.')
        if col_temp_avg is None and col_temp_out is None:
            raise ValueError('Either the average fluid temperature or the inlet and outlet fluid temperature'
                             'should be provided.')

        if isinstance(file, str):
            with open(file, 'rb') as f:
                df = pd.read_csv(f, sep=kwargs.get('sep', ';'), decimal=kwargs.get('decimal', '.'))
        elif isinstance(file, pd.DataFrame):
            df = file
        else:
            # assume it's already a file-like object
            df = pd.read_csv(file, sep=kwargs.get('sep', ';'), decimal=kwargs.get('decimal', '.'))

        self._time_array = np.array(df[col_time])

        if col_power is not None:
            self._power_array = np.array(df[col_power])

        if col_temp_avg is not None:
            # load data and store the load
            self._temperature_array = np.array(df[col_temp_avg])
            return

        self._temperature_array = np.array(df[col_temp_in] + df[col_temp_out]) / 2

    @property
    def time_array(self) -> np.ndarray:
        """
        This function returns the time values for the relevant period, i.e. when the borehole is in steady-state.

        Returns
        -------
        time values : np.ndarray
        """
        return self._time_array[self.start_index:]

    @property
    def temperature_array(self) -> np.ndarray:
        """
        This function returns the temperature values for the relevant period, i.e. when the borehole is in steady-state.

        Returns
        -------
        temperature values : np.ndarray
        """
        return self._temperature_array[self.start_index:]

    @property
    def power_array(self) -> np.ndarray:
        """
        This function returns the power values for the relevant period, i.e. when the borehole is in steady-state.

        Returns
        -------
        power values : np.ndarray
        """
        if self._avg_power is not None:
            return np.full(self._time_array[self.start_index:].shape, self._avg_power)
        return self._power_array[self.start_index:]

    @property
    def log_time_array(self) -> np.ndarray:
        """
        This function returns the natural logarithm of the time values.

        Returns
        -------
        logarithmic time: np.ndarray
        """
        return np.log(self.time_array)

    @property
    def average_power(self) -> float:
        """
        This function returns the average power during the relevant time of the pyTRT.

        Returns
        -------
        logarithmic time: float
        """
        if self._avg_power is not None:
            return self._avg_power
        return np.average(self.power_array)
