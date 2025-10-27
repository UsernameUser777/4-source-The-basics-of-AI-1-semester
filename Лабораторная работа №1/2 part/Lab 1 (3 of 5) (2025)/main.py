# fuzzy_accumulation.py
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def low_speed_mf(v):
    if v <= 200: return 1.0
    elif v <= 600: return (600 - v) / 400
    else: return 0.0

def medium_speed_mf(v):
    if v <= 200 or v >= 800: return 0.0
    elif v <= 600: return (v - 200) / 400
    else: return (800 - v) / 200

mu_low, mu_med, mu_high = 0.875, 0.125, 0.0

def plot_accumulation(root):
    speed_range = np.linspace(0, 1000, 1000)
    low_vals = [min(low_speed_mf(v), mu_low) for v in speed_range]
    med_vals = [min(medium_speed_mf(v), mu_med) for v in speed_range]
    agg = [max(l, m, 0.0) for l, m in zip(low_vals, med_vals)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(speed_range, agg, 'k--', label='Агрегированное множество', linewidth=2.5)
    ax.fill_between(speed_range, 0, agg, color='lightgray', alpha=0.5)
    ax.set_xlabel('Скорость (об/мин)')
    ax.set_ylabel('Степень принадлежности')
    ax.set_title('Этап 4: Аккумуляция')
    ax.legend()
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Аккумуляция")
    plot_accumulation(root)
    root.mainloop()
