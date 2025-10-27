# fuzzy_final_graph.py
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Вспомогательные функции (как в основном коде) ---
def low_temp_mf(t):
    if t <= 10: return 1.0
    elif t <= 30: return (30 - t) / 20
    else: return 0.0

def medium_temp_mf(t):
    if t <= 10 or t >= 50: return 0.0
    elif t <= 30: return (t - 10) / 20
    else: return (50 - t) / 20

def high_temp_mf(t):
    if t <= 30: return 0.0
    elif t <= 50: return (t - 30) / 20
    else: return 1.0

def low_speed_mf(v):
    if v <= 200: return 1.0
    elif v <= 600: return (600 - v) / 400
    else: return 0.0

def medium_speed_mf(v):
    if v <= 200 or v >= 800: return 0.0
    elif v <= 600: return (v - 200) / 400
    else: return (800 - v) / 200

def high_speed_mf(v):
    if v <= 600: return 0.0
    elif v <= 800: return (v - 600) / 200
    else: return 1.0

def defuzzify(x, y):
    num = np.trapz(x * y, x)
    den = np.trapz(y, x)
    return num / den if den != 0 else 0

def calculate_fan_speed(t):
    mu_l = low_temp_mf(t)
    mu_m = medium_temp_mf(t)
    mu_h = high_temp_mf(t)

    v_range = np.linspace(0, 1000, 1000)
    low = [min(low_speed_mf(v), mu_l) for v in v_range]
    med = [min(medium_speed_mf(v), mu_m) for v in v_range]
    high = [min(high_speed_mf(v), mu_h) for v in v_range]
    agg = [max(a, b, c) for a, b, c in zip(low, med, high)]
    return defuzzify(v_range, agg)

def plot_final_graph(root, variant=1):
    temps = []
    speeds = []
    t0 = 12 + 0.5 * variant
    for i in range(5):
        t = t0 + i * 0.3
        v = calculate_fan_speed(t)
        temps.append(t)
        speeds.append(v)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temps, speeds, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Температура (°C)')
    ax.set_ylabel('Скорость вентилятора (об/мин)')
    ax.set_title('Итоговый график: V(t) для 5 итераций')
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Вывод таблицы
    text = "\n".join([f"Итерация {i+1}: t={temps[i]:.1f}°C → V={speeds[i]:.1f} об/мин" for i in range(5)])
    result_label = ttk.Label(root, text=text, justify="left", font=("Courier", 10))
    result_label.pack(pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Итоговый график V(t)")
    plot_final_graph(root, variant=1)
    root.mainloop()
