import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import Dict, List
import seaborn as sns


class RobustnessPlotter:
    def __init__(self, results_dir: str):
        """Initialize plotter with results directory"""
        self.results_dir = results_dir
        self.plot_dir = os.path.join(results_dir, 'report_plots')
        os.makedirs(self.plot_dir, exist_ok=True)

        # Load results
        with open(os.path.join(results_dir, 'robustness_results.txt'), 'r') as f:
            self.raw_results = f.read()

        # Parse results into structured format
        self.results = self.parse_results()

    def parse_results(self) -> Dict:
        """Parse the results text file into a structured format"""
        results = {}
        current_perturb = None

        for line in self.raw_results.split('\n'):
            if line.endswith(':') and not line.startswith('Statistics'):
                current_perturb = line[:-1]
                results[current_perturb] = {
                    'levels': [],
                    'scores': [],
                    'stats': {}
                }
            elif line.startswith('Level'):
                continue
            elif line.strip() and current_perturb and not line.startswith(('-', '=', 'Statistics')):
                try:
                    level, score = map(float, line.strip().split('\t'))
                    results[current_perturb]['levels'].append(level)
                    results[current_perturb]['scores'].append(score)
                except:
                    if line.startswith('Baseline'):
                        results[current_perturb]['stats']['baseline'] = float(line.split(': ')[1])
                    elif line.startswith('Worst'):
                        results[current_perturb]['stats']['worst'] = float(line.split(': ')[1])
                    elif line.startswith('Maximum'):
                        results[current_perturb]['stats']['max_degradation'] = float(line.split(': ')[1])
                    elif line.startswith('Most'):
                        results[current_perturb]['stats']['worst_level'] = float(line.split(': ')[1])

        return results

    def plot_perturbation_comparison(self):
        """Create comparison plot of all perturbations"""
        plt.figure(figsize=(12, 8))

        for perturb, data in self.results.items():
            # Normalize x-axis to [0,1] for comparison
            x = np.array(data['levels'])
            x_norm = (x - x.min()) / (x.max() - x.min())
            plt.plot(x_norm, data['scores'], marker='o', label=perturb)

        plt.xlabel('Normalized Perturbation Strength')
        plt.ylabel('Mean Dice Score')
        plt.title('Model Robustness Comparison Across Perturbations')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'perturbation_comparison.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

    def plot_degradation_summary(self):
        """Create bar plot of maximum degradation for each perturbation"""
        perturbations = list(self.results.keys())
        degradations = [data['stats']['max_degradation'] for data in self.results.values()]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(perturbations, degradations)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Perturbation Type')
        plt.ylabel('Maximum Performance Degradation')
        plt.title('Robustness Analysis: Maximum Impact of Each Perturbation')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'degradation_summary.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

    def plot_vulnerability_profile(self):
        """Create heatmap showing vulnerability at different perturbation levels"""
        perturbations = list(self.results.keys())
        max_samples = max(len(data['scores']) for data in self.results.values())

        # Create normalized score matrix
        score_matrix = np.zeros((len(perturbations), max_samples))
        for i, data in enumerate(self.results.values()):
            scores = np.array(data['scores'])
            normalized_scores = (scores - scores[0]) / scores[0]  # Relative to baseline
            score_matrix[i, :len(normalized_scores)] = normalized_scores

        plt.figure(figsize=(12, 8))
        sns.heatmap(score_matrix, cmap='RdYlBu_r', center=0,
                    xticklabels=list(range(max_samples)),
                    yticklabels=perturbations)
        plt.xlabel('Perturbation Level Index')
        plt.ylabel('Perturbation Type')
        plt.title('Vulnerability Profile: Relative Performance Degradation')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'vulnerability_profile.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

    def generate_all_plots(self):
        """Generate all plots for the report"""
        print("Generating plots for report...")
        self.plot_perturbation_comparison()
        self.plot_degradation_summary()
        self.plot_vulnerability_profile()
        print(f"Plots saved in: {self.plot_dir}")

        # Generate summary statistics
        summary = "Robustness Analysis Summary\n"
        summary += "=========================\n\n"

        # Find most and least robust perturbations
        degradations = [(p, d['stats']['max_degradation'])
                        for p, d in self.results.items()]
        most_vulnerable = max(degradations, key=lambda x: x[1])
        least_vulnerable = min(degradations, key=lambda x: x[1])

        summary += f"Most vulnerable to: {most_vulnerable[0]} "
        summary += f"(degradation: {most_vulnerable[1]:.4f})\n"
        summary += f"Most robust to: {least_vulnerable[0]} "
        summary += f"(degradation: {least_vulnerable[1]:.4f})\n\n"

        summary += "Performance Degradation Summary:\n"
        for perturb, data in self.results.items():
            summary += f"{perturb}:\n"
            summary += f"  Baseline: {data['stats']['baseline']:.4f}\n"
            summary += f"  Worst: {data['stats']['worst']:.4f}\n"
            summary += f"  Degradation: {data['stats']['max_degradation']:.4f}\n"
            summary += f"  Critical Level: {data['stats']['worst_level']}\n\n"

        # Save summary
        with open(os.path.join(self.plot_dir, 'analysis_summary.txt'), 'w') as f:
            f.write(summary)


def main():
    results_dir = './robustness_results'  # Update this path as needed
    plotter = RobustnessPlotter(results_dir)
    plotter.generate_all_plots()


if __name__ == '__main__':
    main()