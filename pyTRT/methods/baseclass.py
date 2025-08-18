from dataclasses import dataclass


@dataclass(slots=True)
class _Method:
    """
    Baseclass for the different method of TRT analysis.

    Parameters
    ----------
    borehole_resistance : float
        The thermal resistance of the borehole in mK/W.
    thermal_conductivity : float
        The thermal conductivity of the ground in W/(mK).

    """
    borehole_resistance: float
    thermal_conductivity: float
    _a: float = 0
    _b: float = 0
