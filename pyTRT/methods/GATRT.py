"""
This file contains the code for the GA-TRT approach for analysing the TRT measurements.
This is based on the work of Focaccia et al. (2013)
Focaccia, Sara & Tinti, Francesco & Bruno, Roberto. (2013). A software tool for geostatistical analysis of thermal response test data: GA-TRT. Computers & Geosciences. 59. 163-170. 10.1016/j.cageo.2013.06.003.
"""
import math

import numpy as np

from GHEtool.VariableClasses.GroundData._GroundData import _GroundData
from pyTRT.utils import TRTData
from pyTRT.methods.baseclass import _Method
from typing import Union


class GATRT(_Method):
    """
    This class contains the GA-TRT method for the analysis of the TRT measurements.
    """

    def __init__(self, data: TRTData, borehole_length: float, borehole_radius: float,
                 volumetric_heat_capacity: Union[float, _GroundData], **kwargs):
        """
        Initialises the GA-TRT method for the analysis of the TRT measurements.
        This method is based on the work of (Focaccia, S., 2013) [#Focaccia]_.

        Parameters
        ----------
        data : TRTData
            Object with the TRT measurement data.
        borehole_length : float
            Length of the borehole heat exchanger [m]
        borehole_radius : float
            Radius of the borehole [m]
        volumetric_heat_capacity : float | _GroundData
            Volumetric heat capacity [J/(m³K)]

        References
        ----------
        .. [#Focaccia] Focaccia, Sara & Tinti, Francesco & Bruno, Roberto. (2013). A software tool for geostatistical analysis of thermal response test data: GA-TRT. Computers & Geosciences. 59. 163-170. 10.1016/j.cageo.2013.06.003.
        """

        self._declustered_time = np.array([])
        self._declustered_power = np.array([])
        self._declustered_temperature = np.array([])

        # prelaminary declustering
        self._decluster_array(data, **kwargs)

        a, b = np.polyfit(data.log_time_array, data.temperature_array, 1)

        ks = data.average_power / (4 * math.pi * borehole_length * a)

        gamma = 0.5772156649  # Euler-Mascheroni Constant
        Rb = (b - data.undisturbed_ground_temperature) * borehole_length / data.average_power - 1 / (
                4 * math.pi * ks) * (
                     np.log(4 * ks / volumetric_heat_capacity / (borehole_radius ** 2)) - gamma)

        # initiate baseclass
        super().__init__(Rb, ks)

    def _decluster_array(self, data: TRTData, n: int = 400, dt: float = 0.5, start: float = 3., **kwargs) -> None:
        """
        This function declusters the measured data to make the information uniform in time-log space.
        This declustering is done based on a random stratified sampling, using n random values in a time interval (dt)
        starting at start.

        Parameters
        ----------
        data : TRTData
            Object with the TRT measurement data.
        n : int
            Number of random samples in each time interval.
        dt : float
            Length of the time interval [log(s)].
        start : float
            Lower bound for the destructuring [log(s)].

        Returns
        -------
        None
        """
        number_of_measurements = len(data.log_time_array)
        start_boundary = start

        declustered_time = []
        declustered_temperature = []
        declustered_power = []

        while start_boundary < np.max(data.log_time_array):
            # Identify all indices in the current class interval
            mask = (data.log_time_array >= start_boundary) & (data.log_time_array < start_boundary + dt)
            indices = np.where(mask)[0]

            if len(indices) == 0:
                start_boundary += dt
                continue

            if len(indices) > n:
                selected = np.random.choice(indices, n, replace=False)
            else:
                selected = indices

            for idx in selected:
                declustered_time.append(data.log_time_array[idx])
                declustered_temperature.append(data.temperature_array[idx])
                declustered_power.append(data.power_array[idx])

            # Move to the next class interval
            start_boundary += dt

        # sort array
        declustered_time.sort(key=lambda x: x[0])
        declustered_temperature.sort(key=lambda x: x[0])
        declustered_power.sort(key=lambda x: x[0])

        # store array
        self._declustered_time = np.array(declustered_time)
        self._declustered_temperature = np.array(declustered_temperature)
        self._declustered_power = np.array(declustered_power)
