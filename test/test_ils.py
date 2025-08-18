import numpy as np
import pytest

from GHEtool import GroundConstantTemperature
from pyTRT import TRTData, FOLDER
from pyTRT.methods import ILS, _Method


def test_linz_ils():
    linz = TRTData(FOLDER.parent.joinpath("examples/data/Linz.csv"), 't [s]', 'Tf [degC]', col_power='P [W]',
                   decimal=',', undisturbed_ground=11.7)

    result = ILS(linz, 150, 0.133 / 2, 2.3e6)
    assert np.isclose(result.thermal_conductivity, 2.2144689487347056)
    assert np.isclose(result.borehole_resistance, 0.1104488374350138)


def test_dinsl_ils():
    linz = TRTData(FOLDER.parent.joinpath("examples/data/Dinsl.csv"), 't [s]', 'Tf [degC]', col_power='P [W]',
                   decimal=',', undisturbed_ground=11.8)

    result = ILS(linz, 99.3, 0.22 / 2, 2.35e6)
    assert np.isclose(result.thermal_conductivity, 2.305895592030428)
    assert np.isclose(result.borehole_resistance, 0.10489058718725794)


def test_ravensburg_ils():
    linz = TRTData(FOLDER.parent.joinpath("examples/data/Ravensburg.csv"), 't [s]', 'Tf [degC]', col_power='P [W]',
                   decimal=',', undisturbed_ground=14.7)

    result = ILS(linz, 193.5, 0.2 / 2, 2.26e6)
    assert np.isclose(result.thermal_conductivity, 2.267969906573817)
    assert np.isclose(result.borehole_resistance, 0.0817363638417728)

    ground = GroundConstantTemperature(2, 10, 2.26e6)
    result = ILS(linz, 193.5, 0.2 / 2, ground)
    assert np.isclose(result.thermal_conductivity, 2.267969906573817)
    assert np.isclose(result.borehole_resistance, 0.0817363638417728)
    assert np.isclose(result._a, 1.7454382119664755)
    assert np.isclose(result._b, 4.108257293968212)
