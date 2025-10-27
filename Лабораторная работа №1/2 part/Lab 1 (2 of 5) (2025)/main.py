# fuzzy_activation.py
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Функции принадлежности для скорости
def low_speed_mf(v):
    if v <= 200:
        return 1.0
    elif 200 < v <= 600:
        return (600 - v) / 400
    else:
        return 0.0

def medium_speed_mf(v):
    if v <= 200 or v >= 800:
        return 0.0
    elif 200 < v <= 600:
        return (v - 200) / 400
    else:  # 600 < v < 800
        return (800 - v) / 200

def high_speed_mf(v):
    if v <= 600:
        return 0.0
    elif 600 < v <= 800:
        return (v - 600) / 200
    else:
        return 1.0

# Степени принадлежности при t = 12.5°C
mu_low = 0.875
mu_med = 0.125
mu_high = 0.0

def plot_activation(root):
    speed_range = np.linspace(0, 1000, 1000)
    low_vals = [min(low_speed_mf(v), mu_low) for v in speed_range]
    med_vals = [min(medium_speed_mf(v), mu_med) for v in speed_range]
    high_vals = [min(high_speed_mf(v), mu_high) for v in speed_range]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(speed_range, low_vals, label='Низкая (усечена)', linewidth=2)
    ax.plot(speed_range, med_vals, label='Средняя (усечена)', linewidth=2)
    ax.plot(speed_range, high_vals, label='Высокая (не активна)', linewidth=2)
    ax.set_xlabel('Скорость (об/мин)')
    ax.set_ylabel('Степень принадлежности')
    ax.set_title('Этап 3: Активизация (усечение)')
    ax.legend()
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    result_label = ttk.Label(root, text=f"Активизация при μ_низкая={mu_low}, μ_средняя={mu_med}")
    result_label.pack(pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Активизация (скорость вентилятора)")
    plot_activation(root)
    root.mainloop()
