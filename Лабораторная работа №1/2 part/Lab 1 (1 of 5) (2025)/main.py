# fuzzy_fuzzification.py
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Функции принадлежности для температуры
def low_temp_mf(t):
    if t <= 10:
        return 1.0
    elif 10 < t <= 30:
        return (30 - t) / 20
    else:
        return 0.0

def medium_temp_mf(t):
    if t <= 10 or t >= 50:
        return 0.0
    elif 10 < t <= 30:
        return (t - 10) / 20
    else:  # 30 < t < 50
        return (50 - t) / 20

def high_temp_mf(t):
    if t <= 30:
        return 0.0
    elif 30 < t <= 50:
        return (t - 30) / 20
    else:
        return 1.0

def plot_fuzzification(root, temperature=12.5):
    temp_range = np.linspace(0, 60, 600)
    low_vals = [low_temp_mf(t) for t in temp_range]
    med_vals = [medium_temp_mf(t) for t in temp_range]
    high_vals = [high_temp_mf(t) for t in temp_range]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(temp_range, low_vals, label='Низкая', linewidth=2)
    ax.plot(temp_range, med_vals, label='Средняя', linewidth=2)
    ax.plot(temp_range, high_vals, label='Высокая', linewidth=2)
    ax.axvline(x=temperature, color='red', linestyle='--', label=f't = {temperature}°C')
    ax.set_xlabel('Температура (°C)')
    ax.set_ylabel('Степень принадлежности')
    ax.set_title('Этап 1: Фаззификация')
    ax.legend()
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Вывод значений
    mu_low = low_temp_mf(temperature)
    mu_med = medium_temp_mf(temperature)
    mu_high = high_temp_mf(temperature)
    result_label = ttk.Label(root, text=f"μ_низкая={mu_low:.3f}, μ_средняя={mu_med:.3f}, μ_высокая={mu_high:.3f}")
    result_label.pack(pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Фаззификация (температура)")
    plot_fuzzification(root, temperature=12.5)
    root.mainloop()
