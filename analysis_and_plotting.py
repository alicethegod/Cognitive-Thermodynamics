# -*- coding: utf-8 -*-
"""
Final Data Processing and Plotting Script for Cognitive Thermodynamics

This script serves as the final, definitive analysis tool for our research. It
separates the data analysis from the raw experiment generation, which is a key
scientific best practice.

It performs the following critical steps:
1.  **Loads Raw Data**: It loads the results JSON file from a path specified
    via a command-line argument, making the script portable.
2.  **Uses Provided Potential Energy**: It now uses the 'energy_distribution'
    key directly from the JSON file.
3.  **Generates Publication-Ready Plots**: It creates the main and supplementary
    plots, now with auto-adjusting Y-axis and smoothed trend lines for clarity.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
import argparse

def load_results(file_path):
    """Loads results from the specified JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at the specified path: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file: {file_path}")
        return None


def process_data(results):
    """
    Processes raw experimental data. It extracts scalar metrics and uses the
    potential energy distribution provided directly in the data file.
    """
    if not results:
        return {}

    data = {}
    scalar_keys = ['generation', 'accuracy', 'inter_conceptual_entropy', 'so', 'cce']
    for key in scalar_keys:
        value_list = [r.get(key, 0) for r in results]
        data[key] = np.array(value_list)
        
    if 'htse' in results[0] and 'hsie' in results[0]:
        print("Found pre-calculated average entropies. Using them for trend plots.")
        avg_htse_series = [r.get('htse', 0) for r in results]
        avg_hsie_series = [r.get('hsie', 0) for r in results]
    else:
        print("Calculating average entropies from distributions for trend plots.")
        avg_htse_series = [np.mean(r.get('htse_distribution', [0])) for r in results]
        avg_hsie_series = [np.mean(r.get('hsie_distribution', [0])) for r in results]
    
    data['htse'] = np.array(avg_htse_series)
    data['hsie'] = np.array(avg_hsie_series)

    potential_energy_distributions = []
    for r in results:
        energy_dist = r.get('energy_distribution', [])
        potential_energy_distributions.append(energy_dist)
            
    data['potential_energy_dist'] = potential_energy_distributions
    
    if data['htse'].size > 0 and data['hsie'].size > 0 and (np.max(data['htse']) - np.min(data['htse'])) > 1e-9 and (np.max(data['hsie']) - np.min(data['hsie'])) > 1e-9:
        htse_scaler = MinMaxScaler()
        hsie_scaler = MinMaxScaler()
        data['htse_norm'] = htse_scaler.fit_transform(data['htse'].reshape(-1, 1)).flatten()
        data['hsie_norm'] = hsie_scaler.fit_transform(data['hsie'].reshape(-1, 1)).flatten()
        w1, w2 = 0.5, 0.5
        data['cognitive_load_norm'] = np.sqrt(
            (w1 * data['htse_norm'])**2 + (w2 * data['hsie_norm'])**2
        )
    else:
        data['htse_norm'] = np.zeros_like(data['htse'])
        data['hsie_norm'] = np.zeros_like(data['hsie'])
        data['cognitive_load_norm'] = np.zeros_like(data['htse'])

    return data

def plot_definitive_results(data, output_dir="."):
    """Generates the final 2x3 'six-act narrative' plot for the paper."""
    if not data or 'generation' not in data or data['generation'].size == 0:
        print("No data to plot for definitive results.")
        return

    gens = data['generation']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    fig.suptitle('Definitive Experiment: The Thermodynamic Collapse of a Closed Cognitive System', fontsize=24, y=0.98)

    # Plot 1: Performance Degradation
    axes[0, 0].plot(gens, data['accuracy'], marker='o', c='r')
    axes[0, 0].set_title('1. Informational Collapse', fontsize=16)
    axes[0, 0].set_ylabel('Accuracy on Real Test Set (%)', fontsize=14)

    # Plot 2: System Temperature (ICE)
    axes[0, 1].plot(gens, data['inter_conceptual_entropy'], marker='x', c='darkred')
    axes[0, 1].set_title('2. Semantic Heat Death (ICE)', fontsize=16)
    axes[0, 1].set_ylabel('System Temperature (Entropy)', fontsize=14)

    # Plot 3: Organizational Collapse (SO)
    axes[0, 2].plot(gens, data['so'], marker='d', c='purple', alpha=0.6, label='Raw SO')
    so_smooth = sm.nonparametric.lowess(data['so'], gens, frac=0.4)[:, 1]
    axes[0, 2].plot(gens, so_smooth, c='indigo', lw=2.5, label='SO Trend (LOWESS)')
    axes[0, 2].set_title('3. Organizational Collapse (SO)', fontsize=16)
    axes[0, 2].set_ylabel('Avg. Specialization Similarity', fontsize=14)
    axes[0, 2].legend(loc='lower right', fontsize='small')

    # Plot 4: The Entropy Trade-off (Normalized)
    if data['htse_norm'].size > 0:
        axes[1, 0].plot(gens, data['htse_norm'], marker='s', ls='--', c='b', alpha=0.7, label="H_TSE' (Normalized)")
        axes[1, 0].plot(gens, data['hsie_norm'], marker='^', ls=':', c='g', alpha=0.7, label="H_SIE' (Normalized)")
    axes[1, 0].set_title("4. The Entropy Trade-off (Normalized)", fontsize=16)
    axes[1, 0].set_ylabel("Normalized Entropy Value", fontsize=14)
    axes[1, 0].legend(loc='upper right', fontsize='small')

    # Plot 5: Statistical Proof of Collapse
    if data['cognitive_load_norm'].size > 1:
        latter_half_idx = len(gens) // 2
        loads_smooth = sm.nonparametric.lowess(data['cognitive_load_norm'], gens, frac=0.4)[:, 1]
        try:
            if len(gens[latter_half_idx:]) > 1:
                slope, intercept, _, p_value, _ = linregress(gens[latter_half_idx:], loads_smooth[latter_half_idx:])
                axes[1, 1].scatter(gens, data['cognitive_load_norm'], c='purple', alpha=0.2, label='Raw Normalized Load')
                axes[1, 1].plot(gens, loads_smooth, c='indigo', lw=2.5, label='Smoothed Trend')
                axes[1, 1].plot(gens[latter_half_idx:], intercept + slope*gens[latter_half_idx:], 'r', lw=2, label='Linear Fit on Trend')
                result_text = f"Trend Analysis (Gens {gens[latter_half_idx]}-{gens[-1]}):\nSlope = {slope:+.5f}\np-value = {p_value:.4f}"
                if not np.isnan(p_value) and p_value < 0.05:
                    result_text += "\n(Statistically Significant)"
                    axes[1, 1].set_facecolor('#e6ffe6')
                else:
                    result_text += "\n(Not Significant)"
                    axes[1, 1].set_facecolor('#ffe6e6')
                axes[1, 1].text(0.05, 0.95, result_text, transform=axes[1, 1].transAxes, fontsize=11,
                               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
            else:
                 axes[1, 1].text(0.5, 0.5, "Not enough data for trend analysis.", ha='center', va='center')
        except ValueError:
            axes[1, 1].text(0.5, 0.5, "Trend analysis failed.\nData may be constant.", ha='center', va='center')
        axes[1, 1].set_title("5. Statistical Proof: The Second Law", fontsize=16)
        axes[1, 1].legend(loc='lower right', fontsize='small')

    # Plot 6: Semantic Potential Energy Distribution
    if 'potential_energy_dist' in data and len(data['potential_energy_dist']) > 0:
        energy_data = [np.array(item) for item in data['potential_energy_dist'] if isinstance(item, list) and len(item) > 0]
        if len(energy_data) > 0 and len(energy_data[0]) > 0 and len(energy_data[-1]) > 0:
            energy_gen0, energy_gen_final = energy_data[0], energy_data[-1]
            combined_energies = np.concatenate((energy_gen0, energy_gen_final))
            bins = np.linspace(np.min(combined_energies), np.max(combined_energies), 30)
            axes[1, 2].hist(energy_gen0, bins=bins, alpha=0.7, label=f'Healthy State (Gen 0)', color='blue', density=True)
            axes[1, 2].hist(energy_gen_final, bins=bins, alpha=0.7, label=f'Collapsed State (Gen {len(gens)-1})', color='red', density=True)
            axes[1, 2].legend(fontsize='small')
        else:
            axes[1, 2].text(0.5, 0.5, "Energy distribution data\nis missing or invalid.", ha='center', va='center')
    axes[1, 2].set_title('6. Semantic Potential Energy', fontsize=16)
    axes[1, 2].set_xlabel("E_pot (Unweighted Norm of State Vector)", fontsize=14)
    axes[1, 2].set_ylabel("Probability Density", fontsize=14)

    for ax in axes.flat:
        if ax != axes[1, 2]:
            ax.set_xlabel('Generation', fontsize=14)
        ax.grid(True, which="both", ls="--")
        ax.tick_params(axis='both', which='major', labelsize=12)
    axes[1, 1].set_ylabel("Total Cognitive Load (Normalized Norm)", fontsize=14)

    plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.08, hspace=0.3, wspace=0.25)

    plot_path = os.path.join(output_dir, "definitive_main_plot_norm_model.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nMain narrative plot (using Norm Model) saved to: {plot_path}")
    plt.show()

def plot_supplementary_results(data, output_dir="."):
    """Generates the supplementary plot with detailed statistical analyses."""
    if not data or 'so' not in data or data['so'].size == 0:
        print("Skipping supplementary plots due to missing data.")
        return
    
    gens = data['generation']
    htses = data['htse']
    hsies = data['hsie']
    sos = data['so']
    cces = data.get('cce', np.array([]))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Supplementary Material: Detailed Statistical Analysis of Key Metrics', fontsize=20, y=0.98)

    def plot_trend(ax, x, y, title, ylabel, color_scatter, color_smooth):
        """
        A robust function to plot raw data, a smoothed trend, and a linear
        regression on the latter half of the smoothed data. Handles low-variance data
        gracefully by auto-adjusting the y-axis.
        """
        if y.size < 2:
            ax.text(0.5, 0.5, "Not enough data for trend analysis.", ha='center', va='center')
            ax.set_title(title, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=12)
            return

        ax.scatter(x, y, alpha=0.3, label='Raw Data', color=color_scatter)

        try:
            # Re-introduce LOWESS smoothing
            smooth = sm.nonparametric.lowess(y, x, frac=0.4)[:, 1]

            # Check for no variance after smoothing
            if (np.max(smooth) - np.min(smooth)) < 1e-9:
                raise ValueError("Smoothed data has no variance.")

            ax.plot(x, smooth, lw=2.5, label='Smoothed Trend', color=color_smooth)
            
            # Perform regression on the latter half of the SMOOTHED data
            latter_half_idx = len(x) // 2
            if len(x[latter_half_idx:]) > 1:
                _, _, _, p_val, _ = linregress(x[latter_half_idx:], smooth[latter_half_idx:])
                ax.set_title(f"{title} (p={p_val:.3f})", fontsize=14)
            else:
                ax.set_title(title, fontsize=14)
        
        except (ValueError, np.linalg.LinAlgError):
            # If smoothing or regression fails, just set the title and legend.
            ax.set_title(title, fontsize=14)
            ax.text(0.5, 0.5, "Could not perform trend analysis.", ha='center', va='center')

        # Auto-adjust Y-axis for low-variance data to ensure visibility
        y_min, y_max = np.min(y), np.max(y)
        y_range = y_max - y_min
        if y_range > 0:
            padding = y_range * 0.1
            ax.set_ylim(y_min - padding, y_max + padding)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc='best', fontsize='small')

    plot_trend(axes[0, 0], gens, htses, "H_TSE' Trend", "Grounding Cost (H_TSE')", 'lightblue', 'navy')
    plot_trend(axes[0, 1], gens, hsies, "H_SIE' Trend", "Structural Complexity (H_SIE')", 'lightgreen', 'darkgreen')
    plot_trend(axes[1, 0], gens, sos, "SO Trend", "Organizational Collapse (SO)", 'thistle', 'indigo')
    
    if cces.any(): # Check if there are any non-zero values
        plot_trend(axes[1, 1], gens, cces, "CCE Trend", "Cognitive Cross-Entropy (CCE)", 'lightpink', 'teal')
    else:
        axes[1, 1].text(0.5, 0.5, "CCE data not found in JSON.", ha='center', va='center')
        axes[1, 1].set_title("CCE Trend", fontsize=14)

    for ax in axes.flat:
        ax.set_xlabel('Generation', fontsize=12)
        ax.grid(True, which="both", ls="--")
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.08, hspace=0.3, wspace=0.3)

    plot_path_supp = os.path.join(output_dir, "supplementary_plots_norm_model.png")
    plt.savefig(plot_path_supp, dpi=300)
    print(f"Supplementary plots (using Norm Model) saved to: {plot_path_supp}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and plot the results of the Cognitive Thermodynamics experiment.')
    parser.add_argument('--json_path', type=str, default='sgd_focused_results.json',
                        help='The full path to the results JSON file.')
    args = parser.parse_args()

    raw_results = load_results(args.json_path)
    if raw_results:
        processed_data = process_data(raw_results)
        output_folder = "final_analysis_norm_model"
        os.makedirs(output_folder, exist_ok=True)
        plot_definitive_results(processed_data, output_dir=output_folder)
        plot_supplementary_results(processed_data, output_dir=output_folder)

