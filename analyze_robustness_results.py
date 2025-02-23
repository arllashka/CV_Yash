import matplotlib.pyplot as plt
import numpy as np
import os


def read_results(filepath):
    """Read and parse the robustness results"""
    results = {}
    current_section = None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.endswith(':') and not line.startswith('Statistics'):
            current_section = line[:-1]
            results[current_section] = {'levels': [], 'scores': []}
        elif line.startswith('Level'):
            continue
        elif current_section and not any(line.startswith(x) for x in ['-', '=', 'Statistics']):
            try:
                level, score = map(float, line.split('\t'))
                results[current_section]['levels'].append(level)
                results[current_section]['scores'].append(score)
            except:
                continue

    return results


def create_robustness_plot(name, levels, scores, x_label, save_path):
    """Create a single robustness plot"""
    plt.figure(figsize=(10, 6))
    plt.plot(levels, scores, marker='o', linewidth=2, markersize=8)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Labels and title
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Dice Score', fontsize=12)
    plt.title(f'Segmentation Performance vs {name}', fontsize=14)

    # Add baseline reference
    plt.axhline(y=scores[0], color='g', linestyle='--',
                label=f'Baseline Score: {scores[0]:.4f}')

    # Add minimum score reference
    min_score = min(scores)
    plt.axhline(y=min_score, color='r', linestyle='--',
                label=f'Minimum Score: {min_score:.4f}')

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Paths
    results_path = 'robustness_results.txt'
    output_dir = 'task4_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Read results
    results = read_results(results_path)

    # Plot configurations for each perturbation
    plot_configs = {
        'Gaussian Noise': {
            'x_label': 'Standard Deviation',
            'filename': 'gaussian_noise_plot.png'
        },
        'Gaussian Blur': {
            'x_label': 'Number of 3x3 Convolutions',
            'filename': 'gaussian_blur_plot.png'
        },
        'Contrast Increase': {
            'x_label': 'Contrast Factor',
            'filename': 'contrast_increase_plot.png'
        },
        'Contrast Decrease': {
            'x_label': 'Contrast Factor',
            'filename': 'contrast_decrease_plot.png'
        },
        'Brightness Increase': {
            'x_label': 'Brightness Addition',
            'filename': 'brightness_increase_plot.png'
        },
        'Brightness Decrease': {
            'x_label': 'Brightness Subtraction',
            'filename': 'brightness_decrease_plot.png'
        },
        'Occlusion': {
            'x_label': 'Square Edge Length',
            'filename': 'occlusion_plot.png'
        },
        'Salt and Pepper': {
            'x_label': 'Noise Amount',
            'filename': 'salt_and_pepper_plot.png'
        }
    }

    # Generate each plot
    print("Generating Task 4 plots...")
    for name, config in plot_configs.items():
        if name in results:
            print(f"Creating plot for {name}...")
            create_robustness_plot(
                name=name,
                levels=results[name]['levels'],
                scores=results[name]['scores'],
                x_label=config['x_label'],
                save_path=os.path.join(output_dir, config['filename'])
            )

    print(f"\nPlots saved in: {output_dir}")
    print("\nPlot summary:")
    print("-------------")
    for name in plot_configs:
        if name in results:
            scores = results[name]['scores']
            degradation = scores[0] - min(scores)
            print(f"{name}:")
            print(f"  - Baseline: {scores[0]:.4f}")
            print(f"  - Worst: {min(scores):.4f}")
            print(f"  - Degradation: {degradation:.4f}\n")


if __name__ == '__main__':
    main()