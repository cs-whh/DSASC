import os
import csv
import subprocess
import itertools
from datetime import datetime

# Hyperparameter search space
lambda_c_values = [0.1, 0.5, 1.0, 2.0, 5.0]
lr_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
epochs_values = [500, 1000, 2000, 5000]

# Dataset to test
dataset = 'cifar10'

# CSV file to save results
csv_file = 'hyperparameter_results.csv'
csv_columns = ['dataset', 'lambda_c', 'lr', 'epochs', 'kmeans_acc', 'kmeans_nmi', 'subspace_acc', 'subspace_nmi', 'timestamp']

# Create CSV file with header if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

# Generate all combinations
all_combinations = list(itertools.product(lambda_c_values, lr_values, epochs_values))
total_runs = len(all_combinations)

print(f"Total hyperparameter combinations to test: {total_runs}")
print(f"Results will be saved to: {csv_file}")
print("=" * 80)

# Iterate through all combinations
for idx, (lambda_c, lr, epochs) in enumerate(all_combinations, 1):
    print(f"\n[{idx}/{total_runs}] Testing: lambda_c={lambda_c}, lr={lr}, epochs={epochs}")
    print("-" * 80)

    # Prepare command
    cmd = [
        'conda', 'run', '-n', 'py312', '--no-capture-output',
        'python', 'baseline.py',
        '--dataset', dataset,
        '--lambda_c', str(lambda_c),
        '--lr', str(lr),
        '--epochs', str(epochs),
        '--save_viz', './viz_results'
    ]

    # Run the command and capture output
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        # Parse output to extract metrics
        output = result.stdout + result.stderr

        # Look for the summary lines
        kmeans_acc = kmeans_nmi = subspace_acc = subspace_nmi = None

        for line in output.split('\n'):
            if 'K-means:' in line and 'ACC' in line:
                parts = line.split('ACC =')[1].split(',')
                kmeans_acc = float(parts[0].strip())
                kmeans_nmi = float(parts[1].split('=')[1].strip())
            elif 'Subspace:' in line and 'ACC' in line:
                parts = line.split('ACC =')[1].split(',')
                subspace_acc = float(parts[0].strip())
                subspace_nmi = float(parts[1].split('=')[1].strip())

        # Save to CSV
        if subspace_acc is not None:
            row = {
                'dataset': dataset,
                'lambda_c': lambda_c,
                'lr': lr,
                'epochs': epochs,
                'kmeans_acc': kmeans_acc,
                'kmeans_nmi': kmeans_nmi,
                'subspace_acc': subspace_acc,
                'subspace_nmi': subspace_nmi,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                writer.writerow(row)

            print(f"Results saved: ACC={subspace_acc:.4f}, NMI={subspace_nmi:.4f}")
        else:
            print("Warning: Could not parse results from output")

    except subprocess.TimeoutExpired:
        print(f"Error: Command timed out for lambda_c={lambda_c}, lr={lr}, epochs={epochs}")
    except Exception as e:
        print(f"Error: {str(e)}")

print("\n" + "=" * 80)
print(f"Hyperparameter search completed! Results saved to {csv_file}")
print("=" * 80)
