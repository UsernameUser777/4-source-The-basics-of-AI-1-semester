# Import necessary libraries
import numpy as np  # for working with arrays and mathematical operations
import matplotlib.pyplot as plt  # for plotting graphs
from tkinter import *  # for creating the graphical interface
from tkinter import ttk, messagebox, filedialog, Toplevel, Text, StringVar, scrolledtext  # additional tkinter widgets
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # for embedding plots in tkinter
import logging  # for logging actions
from logging.handlers import RotatingFileHandler  # for limiting the log file size
import datetime  # for working with dates and times

class FuzzyLogicApp:
    def __init__(self, root):
        # Initialize the main application window
        self.root = root
        self.root.title("Fuzzy Logic for Fan Control")

        # Configure logging
        self.logger = logging.getLogger('fuzzy_logic')
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler('fuzzy_logic.log', maxBytes=1000000, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Configure interface style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TLabel', font=('Arial', 10))
        self.style.configure('TEntry', font=('Arial', 10))

        # Variables for storing temperature and variant number values
        self.temperature = DoubleVar(value=22.0)
        self.variant_number = IntVar(value=1)

        # Widgets for temperature and variant input
        ttk.Label(root, text="Temperature (°C):", font=('Arial', 10)).grid(column=0, row=0, padx=5, pady=5, sticky='w')
        ttk.Entry(root, textvariable=self.temperature, font=('Arial', 10)).grid(column=1, row=0, padx=5, pady=5)
        ttk.Button(root, text="Calculate", command=self.calculate).grid(column=2, row=0, padx=5, pady=5)

        ttk.Label(root, text="Variant Number:", font=('Arial', 10)).grid(column=0, row=1, padx=5, pady=5, sticky='w')
        ttk.Entry(root, textvariable=self.variant_number, font=('Arial', 10)).grid(column=1, row=1, padx=5, pady=5)
        ttk.Button(root, text="Calculate 5 Iterations", command=self.calculate_iterations).grid(column=2, row=1, padx=5, pady=5)

        # Settings and action buttons
        ttk.Button(root, text="Set Membership Functions", command=self.set_membership_parameters).grid(column=0, row=2, columnspan=3, padx=5, pady=5, sticky='ew')
        ttk.Button(root, text="Set Rules", command=self.set_rules).grid(column=0, row=3, columnspan=3, padx=5, pady=5, sticky='ew')

        # Labels for displaying results
        self.result_label = ttk.Label(root, text="Fan Speed: ", font=('Arial', 10))
        self.result_label.grid(column=0, row=4, columnspan=3, padx=5, pady=5, sticky='w')

        self.iteration_results_label = ttk.Label(root, text="Iteration Results: ", font=('Arial', 10))
        self.iteration_results_label.grid(column=0, row=5, columnspan=3, padx=5, pady=5, sticky='w')

        # Save and export buttons
        ttk.Button(root, text="Save Results", command=self.save_results).grid(column=0, row=6, columnspan=1, padx=5, pady=5, sticky='ew')
        ttk.Button(root, text="Export Plots", command=self.export_plots).grid(column=1, row=6, columnspan=1, padx=5, pady=5, sticky='ew')
        ttk.Button(root, text="Help", command=self.show_help).grid(column=2, row=6, columnspan=1, padx=5, pady=5, sticky='ew')
        ttk.Button(root, text="Generate Report", command=self.generate_report).grid(column=0, row=7, columnspan=3, padx=5, pady=5, sticky='ew')

        # Action log
        ttk.Label(root, text="Action Log:").grid(column=0, row=8, padx=5, pady=5, sticky='w')
        self.log_text = scrolledtext.ScrolledText(root, width=80, height=10, wrap=WORD)
        self.log_text.grid(column=0, row=9, columnspan=3, padx=5, pady=5, sticky='ew')
        ttk.Button(root, text="Clear Log", command=self.clear_log).grid(column=0, row=10, columnspan=3, padx=5, pady=5, sticky='ew')

        # Interface themes
        self.theme_var = StringVar(value="light")
        ttk.Button(root, text="Toggle Theme", command=self.toggle_theme).grid(column=0, row=11, columnspan=3, padx=5, pady=5, sticky='ew')

        # Create Canvas and Scrollbar for plots
        self.canvas_frame = Frame(root)
        self.canvas_frame.grid(column=0, row=12, columnspan=3, padx=5, pady=5, sticky='nsew')

        self.canvas = Canvas(self.canvas_frame)
        self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.root.grid_rowconfigure(12, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.figures = []  # List for storing plots

        # Membership function parameters
        self.low_temp_lower = DoubleVar(value=10.0)
        self.low_temp_upper = DoubleVar(value=30.0)
        self.medium_temp_lower = DoubleVar(value=10.0)
        self.medium_temp_upper = DoubleVar(value=50.0)
        self.high_temp_lower = DoubleVar(value=30.0)
        self.high_temp_upper = DoubleVar(value=50.0)

        # Fuzzy logic rules
        self.rules = [
            {"antecedent": "low_temp", "consequent": "low_speed"},
            {"antecedent": "medium_temp", "consequent": "medium_speed"},
            {"antecedent": "high_temp", "consequent": "high_speed"}
        ]

        # Update action log
        self.update_log("Program started")

    def update_log(self, message):
        # Update action log
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - {message}\n"
        self.log_text.insert(END, log_message)
        self.log_text.see(END)
        self.logger.info(message)

    def clear_log(self):
        # Clear action log
        self.log_text.delete(1.0, END)
        self.update_log("Log cleared")

    def calculate(self):
        # Calculate fan speed for a given temperature
        try:
            temperature = self.temperature.get()
            if temperature < 0 or temperature > 60:
                raise ValueError("Temperature must be in the range from 0 to 60 °C")

            self.update_log(f"Calculating fan speed for temperature: {temperature}°C")
            fan_speed, speed_range, low_speed_membership, medium_speed_membership, high_speed_membership, aggregated = calculate_fan_speed(temperature)
            self.result_label.config(text=f"Fan Speed: {fan_speed:.2f} RPM")

            # Plots for each stage of fuzzy inference
            fig_fuzzification = plot_fuzzification(temperature)
            self.add_figure_to_scrollable_frame(fig_fuzzification)

            fig_aggregation = plot_aggregation(speed_range, low_speed_membership, medium_speed_membership, high_speed_membership)
            self.add_figure_to_scrollable_frame(fig_aggregation)

            fig_activation = plot_activation(speed_range, low_speed_membership, medium_speed_membership, high_speed_membership)
            self.add_figure_to_scrollable_frame(fig_activation)

            fig_accumulation = plot_accumulation(speed_range, aggregated)
            self.add_figure_to_scrollable_frame(fig_accumulation)

            fig_defuzzification = plot_defuzzification(speed_range, aggregated)
            self.add_figure_to_scrollable_frame(fig_defuzzification)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.update_log(f"Error calculating fan speed: {e}")

    def calculate_iterations(self):
        # Calculate five iterations with changing temperature
        try:
            variant = self.variant_number.get()
            if variant < 1:
                raise ValueError("Variant number must be a positive number")

            self.update_log(f"Calculating 5 iterations for variant: {variant}")
            temperatures, fan_speeds, areas, centroids = calculate_iterations(variant)

            results_text = "Iteration Results:\n"
            for i, (t, v, area, centroid) in enumerate(zip(temperatures, fan_speeds, areas, centroids)):
                results_text += f"Iteration {i+1}: Temperature = {t:.2f}°C, Speed = {v:.2f} RPM, Area = {area:.2f}, Centroid = {centroid:.2f}\n"

            self.iteration_results_label.config(text=results_text)

            fig = plot_final_graph(temperatures, fan_speeds)
            self.add_figure_to_scrollable_frame(fig)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.update_log(f"Error calculating iterations: {e}")

    def add_figure_to_scrollable_frame(self, fig):
        # Add plot to scrollable frame
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.figures.append(fig)

    def save_results(self):
        # Save results to a file
        try:
            variant = self.variant_number.get()
            temperatures, fan_speeds, _, _ = calculate_iterations(variant)
            filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
            if filename:
                save_results(temperatures, fan_speeds, filename)
                messagebox.showinfo("Save", "Results saved successfully!")
                self.update_log("Results saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.update_log(f"Error saving results: {e}")

    def set_membership_parameters(self):
        # Set membership function parameters
        param_window = Toplevel(self.root)
        param_window.title("Set Membership Functions")

        # Low temperature
        ttk.Label(param_window, text="Low Temperature:").grid(column=0, row=0, padx=5, pady=5, sticky='w')
        ttk.Label(param_window, text="Lower Bound:").grid(column=0, row=1, padx=5, pady=5, sticky='w')
        ttk.Entry(param_window, textvariable=self.low_temp_lower).grid(column=1, row=1, padx=5, pady=5)
        ttk.Label(param_window, text="Upper Bound:").grid(column=0, row=2, padx=5, pady=5, sticky='w')
        ttk.Entry(param_window, textvariable=self.low_temp_upper).grid(column=1, row=2, padx=5, pady=5)

        # Medium temperature
        ttk.Label(param_window, text="Medium Temperature:").grid(column=0, row=3, padx=5, pady=5, sticky='w')
        ttk.Label(param_window, text="Lower Bound:").grid(column=0, row=4, padx=5, pady=5, sticky='w')
        ttk.Entry(param_window, textvariable=self.medium_temp_lower).grid(column=1, row=4, padx=5, pady=5)
        ttk.Label(param_window, text="Upper Bound:").grid(column=0, row=5, padx=5, pady=5, sticky='w')
        ttk.Entry(param_window, textvariable=self.medium_temp_upper).grid(column=1, row=5, padx=5, pady=5)

        # High temperature
        ttk.Label(param_window, text="High Temperature:").grid(column=0, row=6, padx=5, pady=5, sticky='w')
        ttk.Label(param_window, text="Lower Bound:").grid(column=0, row=7, padx=5, pady=5, sticky='w')
        ttk.Entry(param_window, textvariable=self.high_temp_lower).grid(column=1, row=7, padx=5, pady=5)
        ttk.Label(param_window, text="Upper Bound:").grid(column=0, row=8, padx=5, pady=5, sticky='w')
        ttk.Entry(param_window, textvariable=self.high_temp_upper).grid(column=1, row=8, padx=5, pady=5)

        ttk.Button(param_window, text="Apply", command=self.apply_membership_parameters).grid(column=0, row=9, columnspan=2, padx=5, pady=5)

    def apply_membership_parameters(self):
        # Apply new membership function parameters
        messagebox.showinfo("Settings", "Membership function parameters updated!")
        self.update_log("Membership function parameters updated")

    def export_plots(self):
        # Export plots as images
        if not self.figures:
            messagebox.showwarning("Export", "No plots to export!")
            return

        directory = filedialog.askdirectory()
        if directory:
            for i, fig in enumerate(self.figures):
                fig.savefig(f"{directory}/plot_{i+1}.png", bbox_inches='tight', dpi=300)
            messagebox.showinfo("Export", "Plots exported successfully!")
            self.update_log("Plots exported successfully")

    def toggle_theme(self):
        # Toggle interface theme
        if self.theme_var.get() == "light":
            self.style.theme_use('alt')
            self.theme_var.set("dark")
            self.update_log("Dark theme applied")
        else:
            self.style.theme_use('clam')
            self.theme_var.set("light")
            self.update_log("Light theme applied")

    def show_help(self):
        # Display help
        help_window = Toplevel(self.root)
        help_window.title("Help")

        help_text = Text(help_window, wrap=WORD, font=('Arial', 10))
        help_text.insert(END, """
        Fuzzy Logic Fan Control Program.

        Task Description:
        The task of the air conditioner is to maintain the optimal air temperature in the room,
        cooling it when it is hot and heating it when it is cold.

        Fan Speed Adjustment Rules:
        1. If the temperature is low, then the fan speed is low.
        2. If the temperature is medium, then the fan speed is medium.
        3. If the temperature is high, then the fan speed is high.

        Usage:
        1. Enter the temperature and click "Calculate".
        2. Enter the variant number and click "Calculate 5 Iterations".
        3. Use the "Set Membership Functions" button to change parameters.
        4. Use the "Save Results" button to save results to a file.
        5. Use the "Export Plots" button to save plots as images.
        """)
        help_text.config(state=DISABLED)
        help_text.pack(padx=10, pady=10)
        self.update_log("Help opened")

    def set_rules(self):
        # Set fuzzy logic rules
        rules_window = Toplevel(self.root)
        rules_window.title("Set Rules")

        self.rules_listbox = Listbox(rules_window, font=('Arial', 10))
        self.rules_listbox.pack(padx=10, pady=10, fill=BOTH, expand=True)

        for i, rule in enumerate(self.rules):
            self.rules_listbox.insert(END, f"Rule {i+1}: If {rule['antecedent']}, then {rule['consequent']}")

        ttk.Button(rules_window, text="Add Rule", command=self.add_rule).pack(padx=10, pady=5, fill=X)
        ttk.Button(rules_window, text="Delete Rule", command=self.delete_rule).pack(padx=10, pady=5, fill=X)
        self.update_log("Rules settings window opened")

    def add_rule(self):
        # Add a new rule
        add_rule_window = Toplevel(self.root)
        add_rule_window.title("Add Rule")

        self.antecedent_var = StringVar()
        self.consequent_var = StringVar()

        ttk.Label(add_rule_window, text="Antecedent:").grid(column=0, row=0, padx=5, pady=5, sticky='w')
        ttk.Entry(add_rule_window, textvariable=self.antecedent_var).grid(column=1, row=0, padx=5, pady=5)

        ttk.Label(add_rule_window, text="Consequent:").grid(column=0, row=1, padx=5, pady=5, sticky='w')
        ttk.Entry(add_rule_window, textvariable=self.consequent_var).grid(column=1, row=1, padx=5, pady=5)

        ttk.Button(add_rule_window, text="Add", command=self.apply_add_rule).grid(column=0, row=2, columnspan=2, padx=5, pady=5)

    def apply_add_rule(self):
        # Apply a new rule
        antecedent = self.antecedent_var.get()
        consequent = self.consequent_var.get()

        if antecedent and consequent:
            self.rules.append({"antecedent": antecedent, "consequent": consequent})
            self.update_rules_list()
            messagebox.showinfo("Rules", "New rule added!")
            self.update_log(f"New rule added: If {antecedent}, then {consequent}")

    def delete_rule(self):
        # Delete a rule
        selected_index = self.rules_listbox.curselection()
        if selected_index:
            del self.rules[selected_index[0]]
            self.update_rules_list()
            messagebox.showinfo("Rules", "Rule deleted!")
            self.update_log("Rule deleted")

    def update_rules_list(self):
        # Update the list of rules
        self.rules_listbox.delete(0, END)
        for i, rule in enumerate(self.rules):
            self.rules_listbox.insert(END, f"Rule {i+1}: If {rule['antecedent']}, then {rule['consequent']}")

    def generate_report(self):
        # Generate a report
        try:
            variant = self.variant_number.get()
            temperatures, fan_speeds, areas, centroids = calculate_iterations(variant)

            filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
            if filename:
                with open(filename, 'w') as file:
                    file.write("Report for Laboratory Work #1\n\n")
                    file.write("Task Description:\n")
                    file.write("The task of the air conditioner is to maintain the optimal air temperature in the room, cooling it when it is hot and heating it when it is cold.\n\n")

                    file.write("Fan Speed Adjustment Rules:\n")
                    for i, rule in enumerate(self.rules, start=1):
                        file.write(f"{i}. If {rule['antecedent']}, then {rule['consequent']}\n")
                    file.write("\n")

                    file.write("Description of Fuzzy Inference Stages:\n")
                    file.write("1. Fuzzification: calculating membership function values for fuzzy sets.\n")
                    file.write("2. Aggregation: determining the truth degree of a compound statement.\n")
                    file.write("3. Activation: modifying fuzzy sets for the right part of the rules.\n")
                    file.write("4. Accumulation: combining modified sets.\n")
                    file.write("5. Defuzzification: transition from a fuzzy set to a scalar value.\n\n")

                    file.write("Iteration Results:\n")
                    for i, (t, v, area, centroid) in enumerate(zip(temperatures, fan_speeds, areas, centroids), start=1):
                        file.write(f"Iteration {i}: Temperature = {t:.2f}°C, Speed = {v:.2f} RPM, Area = {area:.2f}, Centroid = {centroid:.2f}\n")

                messagebox.showinfo("Report", "Report generated successfully!")
                self.update_log("Report generated successfully")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.update_log(f"Error generating report: {e}")

# Function for plotting fuzzification
def plot_fuzzification(temperature):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    temp_range = np.linspace(0, 60, 1000)
    ax.plot(temp_range, [low_temp_mf(t) for t in temp_range], label='Low', linewidth=2)
    ax.plot(temp_range, [medium_temp_mf(t) for t in temp_range], label='Medium', linewidth=2)
    ax.plot(temp_range, [high_temp_mf(t) for t in temp_range], label='High', linewidth=2)
    ax.axvline(x=temperature, color='r', linestyle='--', label=f'Current t={temperature}°C', linewidth=2)
    ax.set_title('Stage 1: Fuzzification', fontsize=12)
    ax.set_xlabel('Temperature (°C)', fontsize=10)
    ax.set_ylabel('Membership Degree', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    return fig

# Function for plotting aggregation
def plot_aggregation(speed_range, low_speed_membership, medium_speed_membership, high_speed_membership):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(speed_range, low_speed_membership, label='Low Speed', linewidth=2)
    ax.plot(speed_range, medium_speed_membership, label='Medium Speed', linewidth=2)
    ax.plot(speed_range, high_speed_membership, label='High Speed', linewidth=2)
    ax.set_title('Stage 2: Aggregation', fontsize=12)
    ax.set_xlabel('Speed (RPM)', fontsize=10)
    ax.set_ylabel('Membership Degree', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    return fig

# Function for plotting activation
def plot_activation(speed_range, low_speed_membership, medium_speed_membership, high_speed_membership):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(speed_range, low_speed_membership, label='Low Speed', linewidth=2)
    ax.plot(speed_range, medium_speed_membership, label='Medium Speed', linewidth=2)
    ax.plot(speed_range, high_speed_membership, label='High Speed', linewidth=2)
    ax.set_title('Stage 3: Activation', fontsize=12)
    ax.set_xlabel('Speed (RPM)', fontsize=10)
    ax.set_ylabel('Membership Degree', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    return fig

# Function for plotting accumulation
def plot_accumulation(speed_range, aggregated):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(speed_range, aggregated, label='Aggregated', linestyle='--', linewidth=2)
    ax.set_title('Stage 4: Accumulation', fontsize=12)
    ax.set_xlabel('Speed (RPM)', fontsize=10)
    ax.set_ylabel('Membership Degree', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    return fig

# Function for plotting defuzzification
def plot_defuzzification(speed_range, aggregated):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(speed_range, aggregated, label='Aggregated', linestyle='--', linewidth=2)
    ax.set_title('Stage 5: Defuzzification', fontsize=12)
    ax.set_xlabel('Speed (RPM)', fontsize=10)
    ax.set_ylabel('Membership Degree', fontsize=10)
    ax.grid(True)

    # Find the centroid
    centroid = defuzzify(speed_range, aggregated)
    ax.axvline(x=centroid, color='r', linestyle='--', label=f'Centroid: {centroid:.2f} RPM')

    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig

# Membership functions for temperature
def low_temp_mf(t, low_lower=10.0, low_upper=30.0):
    if t <= low_lower:
        return 1.0
    elif low_lower < t <= low_upper:
        return (low_upper - t) / (low_upper - low_lower)
    else:
        return 0.0

def medium_temp_mf(t, medium_lower=10.0, medium_upper=50.0):
    if t <= medium_lower or t >= medium_upper:
        return 0.0
    elif medium_lower < t <= (medium_lower + medium_upper) / 2:
        return (t - medium_lower) / ((medium_lower + medium_upper) / 2 - medium_lower)
    elif (medium_lower + medium_upper) / 2 < t <= medium_upper:
        return (medium_upper - t) / (medium_upper - (medium_lower + medium_upper) / 2)
    else:
        return 1.0

def high_temp_mf(t, high_lower=30.0, high_upper=50.0):
    if t <= high_lower:
        return 0.0
    elif high_lower < t <= high_upper:
        return (t - high_lower) / (high_upper - high_lower)
    else:
        return 1.0

# Membership functions for fan speed
def low_speed_mf(v):
    result = np.zeros_like(v, dtype=float)
    result[v <= 200] = 1.0
    mask = (v > 200) & (v <= 600)
    result[mask] = (600 - v[mask]) / 400
    return result

def medium_speed_mf(v):
    result = np.zeros_like(v, dtype=float)
    mask1 = (v > 200) & (v <= 600)
    result[mask1] = (v[mask1] - 200) / 400
    mask2 = (v > 600) & (v <= 800)
    result[mask2] = (800 - v[mask2]) / 200
    return result

def high_speed_mf(v):
    result = np.zeros_like(v, dtype=float)
    mask = (v > 600) & (v <= 800)
    result[mask] = (v[mask] - 600) / 200
    result[v > 800] = 1.0
    return result

# Defuzzification using the centroid method
def defuzzify(speed_range, membership_values):
    numerator = np.trapz(speed_range * membership_values, speed_range)
    denominator = np.trapz(membership_values, speed_range)
    return numerator / denominator

# Main function for calculating fan speed
def calculate_fan_speed(temperature):
    low_temp = low_temp_mf(temperature)
    medium_temp = medium_temp_mf(temperature)
    high_temp = high_temp_mf(temperature)

    speed_range = np.linspace(0, 1000, 1000)

    low_speed_membership = np.fmin(low_speed_mf(speed_range), low_temp)
    medium_speed_membership = np.fmin(medium_speed_mf(speed_range), medium_temp)
    high_speed_membership = np.fmin(high_speed_mf(speed_range), high_temp)

    aggregated = np.fmax(low_speed_membership, np.fmax(medium_speed_membership, high_speed_membership))

    fan_speed = defuzzify(speed_range, aggregated)

    return fan_speed, speed_range, low_speed_membership, medium_speed_membership, high_speed_membership, aggregated

# Function for plotting membership functions
def plot_membership_functions(temperature, speed_range, low_speed_membership, medium_speed_membership, high_speed_membership, aggregated):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=100)
    temp_range = np.linspace(0, 60, 1000)
    axs[0].plot(temp_range, [low_temp_mf(t) for t in temp_range], label='Low', linewidth=2)
    axs[0].plot(temp_range, [medium_temp_mf(t) for t in temp_range], label='Medium', linewidth=2)
    axs[0].plot(temp_range, [high_temp_mf(t) for t in temp_range], label='High', linewidth=2)
    axs[0].axvline(x=temperature, color='r', linestyle='--', label=f'Current t={temperature}°C', linewidth=2)
    axs[0].set_title('Temperature Membership Functions', fontsize=12)
    axs[0].set_xlabel('Temperature (°C)', fontsize=10)
    axs[0].set_ylabel('Membership Degree', fontsize=10)
    axs[0].legend(fontsize=10)
    axs[0].grid(True)
    axs[1].plot(speed_range, low_speed_membership, label='Low Speed', linewidth=2)
    axs[1].plot(speed_range, medium_speed_membership, label='Medium Speed', linewidth=2)
    axs[1].plot(speed_range, high_speed_membership, label='High Speed', linewidth=2)
    axs[1].plot(speed_range, aggregated, label='Aggregated', linestyle='--', linewidth=2)
    axs[1].set_title('Fan Speed Membership Functions', fontsize=12)
    axs[1].set_xlabel('Speed (RPM)', fontsize=10)
    axs[1].set_ylabel('Membership Degree', fontsize=10)
    axs[1].legend(fontsize=10)
    axs[1].grid(True)
    plt.tight_layout()
    return fig

# Function for plotting the final graph
def plot_final_graph(temperatures, fan_speeds):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    ax.plot(temperatures, fan_speeds, marker='o', linewidth=2)
    ax.set_title('Final Graph of Fan Speed Change', fontsize=12)
    ax.set_xlabel('Temperature (°C)', fontsize=10)
    ax.set_ylabel('Fan Speed (RPM)', fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    return fig

# Function for calculating five iterations
def calculate_iterations(variant_number):
    temperatures = []
    fan_speeds = []
    areas = []
    centroids = []

    current_temp = 12 + 0.5 * variant_number
    temperatures.append(current_temp)

    for _ in range(5):
        fan_speed, speed_range, _, _, _, aggregated = calculate_fan_speed(current_temp)
        fan_speeds.append(fan_speed)

        # Calculate area and centroid
        area = np.trapz(aggregated, speed_range)
        centroid = defuzzify(speed_range, aggregated)
        areas.append(area)
        centroids.append(centroid)

        current_temp += 0.3
        temperatures.append(current_temp)

    return temperatures[:-1], fan_speeds, areas, centroids

# Function for saving results
def save_results(temperatures, fan_speeds, filename="results.txt"):
    with open(filename, 'w') as file:
        file.write("Temperature (°C)\tFan Speed (RPM)\n")
        for t, v in zip(temperatures, fan_speeds):
            file.write(f"{t:.2f}\t{v:.2f}\n")

# Run the application
root = Tk()
app = FuzzyLogicApp(root)
root.mainloop()
