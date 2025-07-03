import numpy as np
import matplotlib.pyplot as plt

# -------- Parameters --------
n_days   = 1                 # 改成 365 可以画全年
n_hours  = 24 * n_days
typical_day_avg_load = 800.0
rand_amp = 0.05              # ±5 % 随机幅度

# 24 h 手工系数（可按需改动）
hourly_factor = np.array([
    0.19, 0.15, 0.10, 0.11, 0.13, 0.14,   # 0–5
    0.25, 0.50, 0.70, 0.88, 0.85, 0.82,   # 6–11
    0.80, 0.84, 0.82, 0.78, 0.72,         # 12–16
    0.68, 0.63, 0.58, 0.50, 0.43, 0.35, 0.30  # 17–23
])

# -------- Build profile with your double loop --------
random_variation = rand_amp * (np.random.rand(n_hours) - 0.5)
total_park_load_profile_year = np.zeros(n_hours)

for day in range(n_days):
    for hour in range(24):
        idx = day * 24 + hour
        base_load = typical_day_avg_load * hourly_factor[hour]
        total_park_load_profile_year[idx] = base_load * (1 + random_variation[idx])

# -------- Plotting --------
hours_axis = np.arange(24 * n_days)

plt.figure(figsize=(10, 4))
plt.step(hours_axis, total_park_load_profile_year, where='post')
plt.title("Park load profile ({} day{})".format(n_days, "" if n_days == 1 else "s"))
plt.xlabel("Hour")
plt.ylabel("Load (kW)")
plt.grid(ls='--', alpha=0.3)
plt.tight_layout()
plt.show()

# If single day, also draw nice 0‑23 plot with x‑ticks
if n_days == 1:
    plt.figure(figsize=(8, 4))
    plt.step(np.arange(24), total_park_load_profile_year, where='post')
    plt.title("Single‑day load profile (detail)")
    plt.xlabel("Hour of day")
    plt.ylabel("Load (kW)")
    plt.xticks(np.arange(0, 25, 2))
    plt.grid(ls='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
