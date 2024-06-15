import matplotlib.pyplot as plt

# Data
severity = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
nds = [0.5762, 0.5379, 0.4798, 0.4364, 0.3988, 0.3715, 0.3730, 0.3117, 0.3178]
map_ = [0.5741, 0.5156, 0.4436, 0.3669, 0.3117, 0.2716, 0.2754, 0.2039, 0.2113]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(severity, nds, label='NDS', marker='o')
plt.plot(severity, map_, label='mAP', marker='s')

# Adding titles and labels
plt.title('NDS and mAP over different severity levels')
plt.xlabel('Degree')
plt.ylabel('Score')
plt.legend()

# Adding grid
plt.grid(True)

# Show plot
plt.show()
