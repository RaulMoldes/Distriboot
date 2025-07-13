# Distriboot
A distributed training algorithm for boosted decision trees. Based on AdaBoost algorithm with MPI parallelization.

## Overview
Distriboot implements three different parallel training strategies for AdaBoost ensembles:
- **Sequential Parallelism**: Bulletproof round-robin approach
- **Batch Parallelism**: High-performance batch training with correctness guarantees
- **Pipelined Parallelism**: Maximum throughput asynchronous pipeline

## Features
- Pure C implementation with MPI for distributed computing
- Multiple parallelization strategies with different performance/reliability trade-offs
- Mathematically correct AdaBoost weight updates
- Deadlock-free MPI communication patterns
- Synthetic data generation for testing and benchmarking
- Comprehensive logging and performance metrics

## Compile
```bash
mpicc -o adaboost main.cpp -lm
```

## Run
```bash
# Run with 8 processes
mpirun -np 8 adaboost

# Run with different process counts
mpirun -np 4 adaboost   # 4 processes
mpirun -np 16 adaboost  # 16 processes
```

## Performance
- **Sequential**: 1/N processes active (N = process count)
- **Batch**: Nearly all processes active, maintains AdaBoost correctness
- **Pipeline**: 100% utilization once pipeline fills, uses staggered weights

## Configuration
Edit the following parameters in `main()`:
```c
int num_samples = 1000;    // Training samples per process
int num_features = 20;     // Feature dimensionality
int num_classes = 2;       // Number of classes
int num_trees = 16;        // Ensemble size
```

## Algorithm Selection
Choose the training function in `main()`:
```c
// For maximum reliability
AdaBoostModel model = train_adaboost_simple_parallel(&train_data, num_trees);

// For balanced performance (recommended)
AdaBoostModel model = train_adaboost_parallel(&train_data, num_trees);

// For maximum throughput
AdaBoostModel model = train_adaboost_pipelined(&train_data, num_trees);
```

## Expected Output
```
=== Simple Parallel AdaBoost Results ===
Training completed successfully
Training time: 2.1234 seconds
Average accuracy: 0.8750 (87.50%)
Processes used: 8
Trees per process: 2.0
Estimated speedup: 5.6x
========================================
```

## Requirements
- MPI implementation (OpenMPI, MPICH, etc.)
- C compiler with math library support
- POSIX-compliant system

## Author
Raul Moldes

## License
MIT License
