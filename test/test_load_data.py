from pyTRT import TRTData, FOLDER

import pytest

import numpy as np
import pandas as pd


def test_value_errors_trt():
    with pytest.raises(ValueError):
        TRTData(FOLDER.parent.joinpath("examples/data/Linz.csv"), 't [s]', 'Tf [degC]', col_power='P [W]', decimal=',',
                start_index=100000)
    with pytest.raises(ValueError):
        TRTData(FOLDER.parent.joinpath("examples/data/Linz.csv"), 't [s]', 'Tf [degC]', col_power='P [W]', decimal=',',
                start_index=-1)
    with pytest.raises(ValueError):
        TRTData(FOLDER.parent.joinpath("examples/data/Linz.csv"), 't [s]', 'Tf [degC]', decimal=',', )
    with pytest.raises(ValueError):
        TRTData(FOLDER.parent.joinpath("examples/data/Linz.csv"), 't [s]', decimal=',')
    with pytest.raises(ValueError):
        TRTData(FOLDER.parent.joinpath("examples/data/Linz.csv"), 't [s]', col_temp_in='Tf [degC]', decimal=',')
    with pytest.raises(ValueError):
        TRTData(FOLDER.parent.joinpath("examples/data/Linz.csv"), 't [s]', 'Tf [degC]', 'Tf [degC]', 'Tf [degC]',
                decimal=',')


def test_p_average():
    data = TRTData(FOLDER.parent.joinpath("examples/data/Linz.csv"), 't [s]', 'Tf [degC]', col_power='P [W]',
                   decimal=',')
    assert np.isclose(data.average_power, 7191.3840791032635)

    data = TRTData(FOLDER.parent.joinpath("examples/data/Linz.csv"), 't [s]', 'Tf [degC]', col_power='P [W]',
                   decimal=',', average_power=8000)
    assert np.isclose(data.average_power, 8000)
    assert np.allclose(np.full(len(data._time_array), 8000), data.power_array)
    assert np.allclose(np.log(data._time_array), data.log_time_array)
    data = TRTData(open(FOLDER.parent.joinpath("examples/data/Linz.csv"), "rb"), 't [s]', 'Tf [degC]',
                   col_power='P [W]',
                   decimal=',', average_power=8000)
    assert np.isclose(data.average_power, 8000)
    assert np.allclose(np.full(len(data._time_array), 8000), data.power_array)
    assert np.allclose(np.log(data._time_array), data.log_time_array)
    data = TRTData(pd.read_csv(open(FOLDER.parent.joinpath("examples/data/Linz.csv"), "rb"), decimal=',', sep=';'),
                   't [s]',
                   'Tf [degC]', col_power='P [W]', average_power=8000)
    assert np.isclose(data.average_power, 8000)
    assert np.allclose(np.full(len(data._time_array), 8000), data.power_array)
    assert np.allclose(np.log(data._time_array), data.log_time_array)


def test_avg_t():
    data = TRTData(FOLDER.parent.joinpath("examples/data/Linz.csv"), 't [s]', 'Tf [degC]', col_power='P [W]',
                   decimal=',')
    data_avg = TRTData(FOLDER.parent.joinpath("test/data/test_linz_dt.csv"), 't [s]', col_temp_in='Tfin [degC]',
                       col_temp_out='Tfout [degC]', col_power='P [W]', decimal=',')

    assert np.allclose(data.temperature_array, data_avg.temperature_array)

    data_avg.start_index = 100
    assert np.allclose(data.temperature_array[100:], data_avg.temperature_array)
    assert np.allclose(data.power_array[100:], data_avg.power_array)
    assert np.allclose(data.time_array[100:], data_avg.time_array)
    assert np.allclose(data.log_time_array[100:], data_avg.log_time_array)
