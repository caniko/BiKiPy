import matplotlib as mpl
import matplotlib.pyplot as plt

ticklabelpad = mpl.rcParams["xtick.major.pad"] * 0.3

fig, ax = plt.subplots()
ax.set_xlim([0, 5])

dx_in_points = 6
fontproperties = ax.xaxis.get_label().get_fontproperties()

ax.annotate(
    "XLabel",
    xy=(1, 0),
    xytext=(dx_in_points, -ticklabelpad),
    ha="left",
    va="top",
    xycoords="axes fraction",
    textcoords="offset points",
    fontproperties=fontproperties,
)

plt.tight_layout()
plt.show()
