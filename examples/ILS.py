from pyTRT import TRTData, FOLDER
from pyTRT.methods import ILS

import matplotlib.pyplot as plt

linz = TRTData(FOLDER.parent.joinpath("examples/data/Dinsl.csv"), 't [s]', 'Tf [degC]', col_power='P [W]',
               decimal=',', undisturbed_ground=11.8)

result = ILS(linz, 99.3, 0.22 / 2, 2.35e6)
result = result.incremental(linz, 99.3, 0.22 / 2, 2.35e6)

# Time corresponding to each incremental fit
time = linz.time_array[99:] / 3600  # hours

fig, ax1 = plt.subplots(figsize=(9, 5))

time = linz.time_array[99:] / 3600

# Left axis
line1 = ax1.plot(time, result.get("ks"), label=r"$k_s$")
ax1.set_xlabel("Test duration [h]")
ax1.set_ylabel(r"Ground thermal conductivity [W/mK]")
ax1.grid(True)

# Right axis
ax2 = ax1.twinx()
line2 = ax2.plot(time, result.get("Rb"), color="C1", label=r"$R_b$")
ax2.set_ylabel(r"Effective borehole resistance [mK/W]")

# Combined legend
lines = line1 + line2
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels)

plt.tight_layout()
plt.show()