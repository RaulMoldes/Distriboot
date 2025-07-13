import numpy as np
import pandas as pd
import time
import subprocess
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification, load_breast_cancer, load_wine
import matplotlib.pyplot as plt
import seaborn as sns

class AdaBoostBenchmark:
    def __init__(self):
        self.results = []

    def generate_test_data(self, n_samples=4000, n_features=10, n_classes=2, random_state=42):
        """Generate synthetic data similar to your C implementation"""
        np.random.seed(random_state)

        # Generate features
        X = np.random.uniform(-5, 5, (n_samples, n_features))

        # Generate labels based on sum > 0 (same as C code)
        y = (X.sum(axis=1) > 0).astype(int)

        return X, y

    def save_data_for_c(self, X, y, filename="benchmark_data.txt"):
        """Save data in format readable by C program"""
        with open(filename, 'w') as f:
            f.write(f"{X.shape[0]} {X.shape[1]} {len(np.unique(y))}\n")
            for i in range(X.shape[0]):
                # Write features
                for j in range(X.shape[1]):
                    f.write(f"{X[i,j]:.6f} ")
                # Write label
                f.write(f"{y[i]}\n")

    def run_sklearn_adaboost(self, X_train, y_train, X_test, y_test, n_estimators=20, max_depth=10):
        """Run scikit-learn AdaBoost"""
        print(f"Running scikit-learn AdaBoost with {n_estimators} trees...")

        # Create base classifier with same parameters as C implementation
        base_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

        # Create AdaBoost classifier
        ada_clf = AdaBoostClassifier(
            base_classifier,
            n_estimators=n_estimators,
            learning_rate=1.0,  # Default learning rate
            algorithm='SAMME',  # Discrete AdaBoost
            random_state=42
        )

        # Train
        start_time = time.time()
        ada_clf.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predict
        y_pred = ada_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            'method': 'scikit-learn',
            'train_time': train_time,
            'accuracy': accuracy,
            'n_estimators': n_estimators,
            'classifier': ada_clf
        }

    def run_mpi_adaboost(self, n_processes=4, n_estimators=20):
        """Run your MPI AdaBoost implementation"""
        print(f"Running MPI AdaBoost with {n_processes} processes...")

        # Compile the C program
        compile_cmd = ["mpic++", "-o", "boosted_tree", "main.cpp", "-lm"]
        try:
            subprocess.run(compile_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e}")
            return None

        # Run the program
        run_cmd = ["mpirun", "-np", str(n_processes), "./boosted_tree"]

        start_time = time.time()
        try:
            result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=300)
            train_time = time.time() - start_time

            if result.returncode == 0:
                # Parse output to extract accuracy
                output_lines = result.stdout.strip().split('\n')
                accuracy = None
                for line in output_lines:
                    if "Avg precision:" in line:
                        # Extract accuracy from line like "Avg precision: 0.7710 (77.10%)"
                        accuracy = float(line.split('(')[1].split('%')[0]) / 100.0
                        break

                return {
                    'method': f'MPI ({n_processes} processes)',
                    'train_time': train_time,
                    'accuracy': accuracy,
                    'n_estimators': n_estimators,
                    'output': result.stdout
                }
            else:
                print(f"MPI execution failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print("MPI execution timed out")
            return None

    def run_scalability_test(self, max_processes=8):
        """Test how MPI implementation scales with number of processes"""
        print("Running scalability test...")

        scalability_results = []

        for n_proc in range(1, max_processes + 1):
            print(f"Testing with {n_proc} processes...")
            result = self.run_mpi_adaboost(n_processes=n_proc, n_estimators=20)
            if result:
                scalability_results.append({
                    'n_processes': n_proc,
                    'train_time': result['train_time'],
                    'accuracy': result['accuracy']
                })

        return scalability_results

    def benchmark_on_real_datasets(self):
        """Benchmark on real datasets"""
        print("Benchmarking on real datasets...")

        datasets = [
            ("Breast Cancer", load_breast_cancer()),
            ("Wine", load_wine())
        ]

        real_data_results = []

        for name, dataset in datasets:
            print(f"\nTesting on {name} dataset...")
            X, y = dataset.data, dataset.target

            # Convert to binary classification for simplicity
            if len(np.unique(y)) > 2:
                y = (y > 0).astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

            # Run scikit-learn
            sklearn_result = self.run_sklearn_adaboost(X_train, y_train, X_test, y_test)
            sklearn_result['dataset'] = name
            real_data_results.append(sklearn_result)

            print(f"Scikit-learn on {name}: {sklearn_result['accuracy']:.4f} accuracy, {sklearn_result['train_time']:.4f}s")

        return real_data_results

    def compare_performance(self, n_samples_list=[1000, 2000, 4000], n_estimators_list=[10, 20, 50]):
        """Compare performance across different dataset sizes and tree counts"""
        print("Running comprehensive performance comparison...")

        comparison_results = []

        for n_samples in n_samples_list:
            for n_estimators in n_estimators_list:
                print(f"\nTesting: {n_samples} samples, {n_estimators} trees")

                # Generate data
                X, y = self.generate_test_data(n_samples=n_samples)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

                # Test scikit-learn
                sklearn_result = self.run_sklearn_adaboost(X_train, y_train, X_test, y_test, n_estimators)
                sklearn_result['n_samples'] = n_samples
                comparison_results.append(sklearn_result)

                # Test MPI (with 4 processes)
                mpi_result = self.run_mpi_adaboost(n_processes=4, n_estimators=n_estimators)
                if mpi_result:
                    mpi_result['n_samples'] = n_samples
                    comparison_results.append(mpi_result)

                print(f"Scikit-learn: {sklearn_result['accuracy']:.4f} accuracy, {sklearn_result['train_time']:.4f}s")
                if mpi_result:
                    print(f"MPI: {mpi_result['accuracy']:.4f} accuracy, {mpi_result['train_time']:.4f}s")

        return comparison_results

    def plot_results(self, results):
        """Create visualization of benchmark results"""
        df = pd.DataFrame(results)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Accuracy comparison
        if 'n_samples' in df.columns:
            sns.boxplot(data=df, x='method', y='accuracy', ax=axes[0,0])
            axes[0,0].set_title('Accuracy Comparison')
            axes[0,0].set_ylabel('Accuracy')

            # Training time comparison
            sns.boxplot(data=df, x='method', y='train_time', ax=axes[0,1])
            axes[0,1].set_title('Training Time Comparison')
            axes[0,1].set_ylabel('Training Time (seconds)')

            # Accuracy vs Dataset Size
            sns.lineplot(data=df, x='n_samples', y='accuracy', hue='method', ax=axes[1,0])
            axes[1,0].set_title('Accuracy vs Dataset Size')
            axes[1,0].set_xlabel('Number of Samples')
            axes[1,0].set_ylabel('Accuracy')

            # Training Time vs Dataset Size
            sns.lineplot(data=df, x='n_samples', y='train_time', hue='method', ax=axes[1,1])
            axes[1,1].set_title('Training Time vs Dataset Size')
            axes[1,1].set_xlabel('Number of Samples')
            axes[1,1].set_ylabel('Training Time (seconds)')

        plt.tight_layout()
        plt.savefig('adaboost_benchmark_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("="*60)
        print("AdaBoost Implementation Benchmark")
        print("="*60)

        all_results = []

        # 1. Basic comparison
        print("\n1. Basic Performance Comparison")
        print("-" * 40)
        X, y = self.generate_test_data(n_samples=2000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        sklearn_result = self.run_sklearn_adaboost(X_train, y_train, X_test, y_test)
        mpi_result = self.run_mpi_adaboost(n_processes=4)

        all_results.extend([sklearn_result, mpi_result])

        # 2. Scalability test
        print("\n2. MPI Scalability Test")
        print("-" * 40)
        scalability_results = self.run_scalability_test(max_processes=6)

        # 3. Dataset size comparison
        print("\n3. Dataset Size Performance")
        print("-" * 40)
        comparison_results = self.compare_performance(
            n_samples_list=[1000, 2000, 4000],
            n_estimators_list=[10, 20]
        )
        all_results.extend(comparison_results)

        # 4. Real datasets
        print("\n4. Real Dataset Performance")
        print("-" * 40)
        real_data_results = self.benchmark_on_real_datasets()
        all_results.extend(real_data_results)

        # Create summary report
        self.create_summary_report(all_results, scalability_results)

        return all_results, scalability_results

    def create_summary_report(self, results, scalability_results):
        """Create a summary report of all benchmark results"""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY REPORT")
        print("="*60)

        # Filter valid results
        valid_results = [r for r in results if r and 'accuracy' in r and r['accuracy'] is not None]

        if valid_results:
            df = pd.DataFrame(valid_results)

            print("\nAccuracy Summary:")
            print(df.groupby('method')['accuracy'].agg(['mean', 'std', 'min', 'max']))

            print("\nTraining Time Summary:")
            print(df.groupby('method')['train_time'].agg(['mean', 'std', 'min', 'max']))

        if scalability_results:
            print("\nScalability Results:")
            for result in scalability_results:
                print(f"  {result['n_processes']} processes: {result['train_time']:.4f}s, "
                      f"accuracy: {result['accuracy']:.4f}")

        print("\nRecommendations:")
        print("- Compare accuracy differences between implementations")
        print("- Analyze speedup with multiple MPI processes")
        print("- Consider memory usage and network communication overhead")
        print("- Test on larger datasets to see where MPI shines")

# Usage example
if __name__ == "__main__":
    benchmark = AdaBoostBenchmark()

    # Run full benchmark suite
    results, scalability = benchmark.run_full_benchmark()

    # Plot results if we have data
    if results:
        benchmark.plot_results([r for r in results if r and 'accuracy' in r])
