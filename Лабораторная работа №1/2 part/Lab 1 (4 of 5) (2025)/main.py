# fuzzy_defuzzification.py
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

mu_low, mu_med = 0.875, 0.125

def defuzzify(x, y):
    num = np.trapz(x * y, x)
    den = np.trapz(y, x)
    return num / den if den != 0 else 0

def plot_defuzzification(root):
    speed_range = np.linspace(0, 1000, 1000)
    low_vals = [min(low_speed_mf(v), mu_low) for v in speed_range]
    med_vals = [min(medium_speed_mf(v), mu_med) for v in speed_range]
    agg = [max(l, m) for l, m in zip(low_vals, med_vals)]

    centroid = defuzzify(speed_range, agg)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(speed_range, agg, 'k--', linewidth=2)
    ax.fill_between(speed_range, 0, agg, color='lightblue', alpha=0.6)
    ax.axvline(x=centroid, color='red', linestyle='--', label=f'Центр тяжести = {centroid:.1f} об/мин')
    ax.set_xlabel('Скорость (об/мин)')
    ax.set_ylabel('Степень принадлежности')
    ax.set_title('Этап 5: Дефаззификация (метод центра тяжести)')
    ax.legend()
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    result_label = ttk.Label(root, text=f"Итоговая скорость: V = {centroid:.2f} об/мин")
    result_label.pack(pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Дефаззификация")
    plot_defuzzification(root)
    root.mainloop()
