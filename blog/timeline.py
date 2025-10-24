import matplotlib.pyplot as plt

# Data
methods = ["DQN", "DDPG", "TD3", "SAC", "CQL/Dreamer"]
descriptions = [
    "Discrete\nValue-based",
    "Deterministic\nActor",
    "Stabilized\nActor",
    "Entropy-\nRegularized",
    "Follow-ups"
]
colors = ["#1F78B4", "#33A02C", "#FB9A99", "#E31A1C", "#5AB4E4"] # Used darker, more contrasting colors

# --- Adjustments for Visibility and Spacing ---
num_methods = len(methods)
# Reduce the overall width of the timeline by scaling the x-positions
scale_factor = 0.5
scaled_num_methods = num_methods * scale_factor

# Data points on the x-axis, now closer together
x_positions = [i * scale_factor + (scale_factor / 2) for i in range(num_methods)]

# Create a wider figure to accommodate larger text
fig, ax = plt.subplots(figsize=(14, 5)) 

# 1. Draw the main timeline line
ax.hlines(0, 0, scaled_num_methods, color='gray', linewidth=2, zorder=1)

# 2. Draw vertical marker lines and colored dots (Lollipops)
for i, x in enumerate(x_positions):
    
    # Define larger vertical distances for better separation
    v_line_len = 0.5
    method_y_offset = 0.55
    desc_y_offset = 0.8
    
    if i % 2 == 0: # Place even index methods above the line
        ax.vlines(x, 0, v_line_len, color=colors[i], linestyle='-', linewidth=2, zorder=2)
        text_y_method = method_y_offset
        text_y_desc = desc_y_offset
        align = 'bottom'
    else: # Place odd index methods below the line
        ax.vlines(x, 0, -v_line_len, color=colors[i], linestyle='-', linewidth=2, zorder=2)
        text_y_method = -method_y_offset
        text_y_desc = -desc_y_offset
        align = 'top'

    # Colored Dot Marker (slightly larger)
    ax.plot(x, 0, 'o', markersize=10, color=colors[i], zorder=3)
    
    # Add Method Name (Increased fontsize)
    ax.text(x, text_y_method, methods[i], ha='center', va=align, color=colors[i], 
            fontsize=20, fontweight='bold')
    
    # Add Description (Increased fontsize)
    ax.text(x, text_y_desc, descriptions[i], ha='center', va=align, fontsize=18)

# Set limits 
ax.set_xlim(0, scaled_num_methods)
ax.set_ylim(-1.2, 1.2) # Adjusted limits for better padding

# Remove axes/ticks
ax.axis('off')

# Title
ax.set_title("Timeline of Continuous Control RL Methods", fontsize=22, pad=25)

plt.tight_layout()
plt.savefig("rl_timeline_improved.png")
plt.close()

print("Generated rl_timeline_improved.png with larger, closer labels.")