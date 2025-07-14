# Distriboot
A distributed training algorithm for boosted decision trees. Based on adaboosst algorithm with MPI parallelization.

## Overview
Distriboot implements three different parallel training strategies for distriboot.out ensembles:
- **Sequential Parallelism**: Bulletproof round-robin approach
- **Batch Parallelism**: High-performance batch training with correctness guarantees
- **Pipelined Parallelism**: Maximum throughput asynchronous pipeline


## Compile
```bash
mpicc distriboot.c -lm -o distriboot.out
```

## Run
```bash
# Run with 8 processes
mpirun -np 8 distriboot.out

# Run with different process counts
mpirun -np 4 distriboot.out   # 4 processes
mpirun -np 16 distriboot.out  # 16 processes
```

## Performance
- **Sequential**: 1/N processes active (N = process count)
- **Batch**: Nearly all processes active, maintains distriboot.out correctness
- **Pipeline**: 100% utilization once pipeline fills, uses staggered weights

## Benchmarks:
Use the script ```benchmark.sh```
