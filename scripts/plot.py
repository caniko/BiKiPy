import matplotlib.pyplot as plt
import numpy as np

# Create some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Plot the data
ax.plot(x, y)

# Set the y-tick positions
yticks = np.linspace(-1, 1, 5)
ax.set_yticks(yticks)

# Set the y-tick labels in reverse order
# ax.set_yticklabels(yticks[::-1])

# Display the plot
plt.show()
