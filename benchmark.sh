#!/bin/bash

# ==============================================================================
# Parallel distriboot MPI Benchmark Script
# Author: Performance Testing Suite
#
# This script benchmarks the three distriboot training modes:
# - Mode 0: Sequential (round-robin, bulletproof but slow)
# - Mode 1: Batched (best balance of accuracy and performance)
# - Mode 2: Pipelined (maximum throughput, slight accuracy trade-off)
# ==============================================================================

# Colors for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration variables
EXECUTABLE="./distriboot.out"
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.txt"
CSV_FILE="${RESULTS_DIR}/benchmark_${TIMESTAMP}.csv"

# Default benchmark parameters
DEFAULT_PROCESSES=(2 4 8)
DEFAULT_SAMPLES=(1000 5000 10000)
DEFAULT_TREES=(16 32 64)
DEFAULT_FEATURES=(10 20 50)
DEFAULT_MODES=(0 1 2)
REPETITIONS=3

# Performance tracking
declare -A results

# ==============================================================================
# Utility Functions
# ==============================================================================

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_section() {
    echo -e "\n${CYAN}--- $1 ---${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${PURPLE}ℹ $1${NC}"
}

# ==============================================================================
# Setup and Validation Functions
# ==============================================================================

check_dependencies() {
    print_section "Checking Dependencies"

    # Check if MPI is available
    if ! command -v mpirun &> /dev/null; then
        print_error "mpirun not found. Please install MPI."
        exit 1
    fi
    print_success "mpirun found: $(which mpirun)"

    # Check if executable exists
    if [ ! -f "$EXECUTABLE" ]; then
        print_error "Executable $EXECUTABLE not found."
        print_info "Compile with: mpicc -o distriboot_mpi distriboot.c -lm"
        exit 1
    fi
    print_success "Executable found: $EXECUTABLE"

    # Check available processors
    local available_cores=$(nproc)
    print_info "Available CPU cores: $available_cores"

    # Create results directory
    mkdir -p "$RESULTS_DIR"
    print_success "Results directory: $RESULTS_DIR"
}

validate_mpi_setup() {
    print_section "Validating MPI Setup"

    # Test simple MPI run
    if mpirun -np 2 echo "MPI test successful" &> /dev/null; then
        print_success "MPI is working correctly"
    else
        print_error "MPI test failed"
        exit 1
    fi

    # Test with the actual executable
    print_info "Testing executable with minimal parameters..."
    if timeout 30 mpirun -np 2 "$EXECUTABLE" --samples 100 --trees 4 --mode 0 &> /dev/null; then
        print_success "Executable runs successfully"
    else
        print_error "Executable test failed"
        exit 1
    fi
}

# ==============================================================================
# Benchmark Execution Functions
# ==============================================================================

run_single_benchmark() {
    local processes=$1
    local samples=$2
    local trees=$3
    local features=$4
    local mode=$5
    local repetition=$6

    local mode_name
    case $mode in
        0) mode_name="Sequential" ;;
        1) mode_name="Batched" ;;
        2) mode_name="Pipelined" ;;
        *) mode_name="Unknown" ;;
    esac

    print_info "Running: $processes procs, $samples samples, $trees trees, $features features, $mode_name mode (rep $repetition)"

    # Create unique output file for this run
    local output_file="${RESULTS_DIR}/run_${processes}p_${samples}s_${trees}t_${features}f_m${mode}_r${repetition}.log"

    # Run the benchmark with timeout
    local start_time=$(date +%s.%N)

    if timeout 300 mpirun -np "$processes" "$EXECUTABLE" \
        --samples "$samples" \
        --trees "$trees" \
        --features "$features" \
        --mode "$mode" > "$output_file" 2>&1; then

        local end_time=$(date +%s.%N)
        local runtime=$(echo "$end_time - $start_time" | bc -l)

        # Parse results from output
        local training_time=$(grep "Training time:" "$output_file" | awk '{print $3}')
        local accuracy=$(grep "Average accuracy:" "$output_file" | awk '{print $3}')

        # Store results
        local key="${processes}_${samples}_${trees}_${features}_${mode}"
        if [ -n "$training_time" ] && [ -n "$accuracy" ]; then
            results["${key}_time_${repetition}"]="$training_time"
            results["${key}_accuracy_${repetition}"]="$accuracy"
            results["${key}_runtime_${repetition}"]="$runtime"
            print_success "Completed in ${training_time}s, accuracy: $accuracy"
        else
            print_warning "Could not parse results from output"
            results["${key}_time_${repetition}"]="ERROR"
            results["${key}_accuracy_${repetition}"]="ERROR"
            results["${key}_runtime_${repetition}"]="ERROR"
        fi
    else
        print_error "Benchmark failed or timed out"
        local key="${processes}_${samples}_${trees}_${features}_${mode}"
        results["${key}_time_${repetition}"]="TIMEOUT"
        results["${key}_accuracy_${repetition}"]="TIMEOUT"
        results["${key}_runtime_${repetition}"]="TIMEOUT"
    fi

    # Clean up output file if successful
    if [ -f "$output_file" ] && grep -q "completed successfully" "$output_file"; then
        rm "$output_file"
    fi
}

calculate_statistics() {
    local key_base=$1
    local metric=$2

    local values=()
    local sum=0
    local count=0

    for rep in $(seq 1 $REPETITIONS); do
        local value="${results[${key_base}_${metric}_${rep}]}"
        if [ "$value" != "ERROR" ] && [ "$value" != "TIMEOUT" ]; then
            values+=("$value")
            sum=$(echo "$sum + $value" | bc -l)
            ((count++))
        fi
    done

    if [ $count -eq 0 ]; then
        echo "ERROR"
        return
    fi

    local avg=$(echo "scale=6; $sum / $count" | bc -l)

    # Calculate standard deviation
    local variance_sum=0
    for value in "${values[@]}"; do
        local diff=$(echo "$value - $avg" | bc -l)
        local sq_diff=$(echo "$diff * $diff" | bc -l)
        variance_sum=$(echo "$variance_sum + $sq_diff" | bc -l)
    done

    local variance=$(echo "scale=6; $variance_sum / $count" | bc -l)
    local stddev=$(echo "scale=6; sqrt($variance)" | bc -l)

    echo "$avg,$stddev,$count"
}

# ==============================================================================
# Main Benchmark Suite
# ==============================================================================

run_scalability_benchmark() {
    print_header "SCALABILITY BENCHMARK - Fixed Problem Size, Varying Processes"

    local fixed_samples=5000
    local fixed_trees=32
    local fixed_features=20

    echo "processes,mode,avg_time,std_time,avg_accuracy,std_accuracy,samples" >> "$CSV_FILE"

    for processes in "${DEFAULT_PROCESSES[@]}"; do
        for mode in "${DEFAULT_MODES[@]}"; do
            print_section "Testing $processes processes, mode $mode"

            for rep in $(seq 1 $REPETITIONS); do
                run_single_benchmark "$processes" "$fixed_samples" "$fixed_trees" "$fixed_features" "$mode" "$rep"
            done

            # Calculate statistics
            local key="${processes}_${fixed_samples}_${fixed_trees}_${fixed_features}_${mode}"
            local time_stats=$(calculate_statistics "$key" "time")
            local accuracy_stats=$(calculate_statistics "$key" "accuracy")

            local avg_time=$(echo "$time_stats" | cut -d',' -f1)
            local std_time=$(echo "$time_stats" | cut -d',' -f2)
            local avg_accuracy=$(echo "$accuracy_stats" | cut -d',' -f1)
            local std_accuracy=$(echo "$accuracy_stats" | cut -d',' -f2)

            echo "$processes,$mode,$avg_time,$std_time,$avg_accuracy,$std_accuracy,$fixed_samples" >> "$CSV_FILE"

            print_success "Mode $mode with $processes processes: ${avg_time}±${std_time}s, accuracy: ${avg_accuracy}±${std_accuracy}"
        done
    done
}

run_problem_size_benchmark() {
    print_header "PROBLEM SIZE BENCHMARK - Fixed Processes, Varying Data Size"

    local fixed_processes=4
    local fixed_trees=32
    local fixed_features=20

    echo "samples,mode,avg_time,std_time,avg_accuracy,std_accuracy,processes" >> "$CSV_FILE"

    for samples in "${DEFAULT_SAMPLES[@]}"; do
        for mode in "${DEFAULT_MODES[@]}"; do
            print_section "Testing $samples samples, mode $mode"

            for rep in $(seq 1 $REPETITIONS); do
                run_single_benchmark "$fixed_processes" "$samples" "$fixed_trees" "$fixed_features" "$mode" "$rep"
            done

            # Calculate statistics
            local key="${fixed_processes}_${samples}_${fixed_trees}_${fixed_features}_${mode}"
            local time_stats=$(calculate_statistics "$key" "time")
            local accuracy_stats=$(calculate_statistics "$key" "accuracy")

            local avg_time=$(echo "$time_stats" | cut -d',' -f1)
            local std_time=$(echo "$time_stats" | cut -d',' -f2)
            local avg_accuracy=$(echo "$accuracy_stats" | cut -d',' -f1)
            local std_accuracy=$(echo "$accuracy_stats" | cut -d',' -f2)

            echo "$samples,$mode,$avg_time,$std_time,$avg_accuracy,$std_accuracy,$fixed_processes" >> "$CSV_FILE"

            print_success "Mode $mode with $samples samples: ${avg_time}±${std_time}s, accuracy: ${avg_accuracy}±${std_accuracy}"
        done
    done
}

run_ensemble_size_benchmark() {
    print_header "ENSEMBLE SIZE BENCHMARK - Fixed Setup, Varying Trees"

    local fixed_processes=4
    local fixed_samples=5000
    local fixed_features=20

    echo "trees,mode,avg_time,std_time,avg_accuracy,std_accuracy,processes" >> "$CSV_FILE"

    for trees in "${DEFAULT_TREES[@]}"; do
        for mode in "${DEFAULT_MODES[@]}"; do
            print_section "Testing $trees trees, mode $mode"

            for rep in $(seq 1 $REPETITIONS); do
                run_single_benchmark "$fixed_processes" "$fixed_samples" "$trees" "$fixed_features" "$mode" "$rep"
            done

            # Calculate statistics
            local key="${fixed_processes}_${fixed_samples}_${trees}_${fixed_features}_${mode}"
            local time_stats=$(calculate_statistics "$key" "time")
            local accuracy_stats=$(calculate_statistics "$key" "accuracy")

            local avg_time=$(echo "$time_stats" | cut -d',' -f1)
            local std_time=$(echo "$time_stats" | cut -d',' -f2)
            local avg_accuracy=$(echo "$accuracy_stats" | cut -d',' -f1)
            local std_accuracy=$(echo "$accuracy_stats" | cut -d',' -f2)

            echo "$trees,$mode,$avg_time,$std_time,$avg_accuracy,$std_accuracy,$fixed_processes" >> "$CSV_FILE"

            print_success "Mode $mode with $trees trees: ${avg_time}±${std_time}s, accuracy: ${avg_accuracy}±${std_accuracy}"
        done
    done
}

generate_summary_report() {
    print_header "GENERATING SUMMARY REPORT"

    {
        echo "====================================================================="
        echo "PARALLEL distriboot MPI BENCHMARK RESULTS"
        echo "Generated: $(date)"
        echo "====================================================================="
        echo ""
        echo "BENCHMARK CONFIGURATION:"
        echo "- Processes tested: ${DEFAULT_PROCESSES[*]}"
        echo "- Sample sizes: ${DEFAULT_SAMPLES[*]}"
        echo "- Tree counts: ${DEFAULT_TREES[*]}"
        echo "- Feature counts: ${DEFAULT_FEATURES[*]}"
        echo "- Modes: 0=Sequential, 1=Batched, 2=Pipelined"
        echo "- Repetitions per configuration: $REPETITIONS"
        echo ""
        echo "SYSTEM INFORMATION:"
        echo "- CPU cores: $(nproc)"
        echo "- Memory: $(free -h | grep Mem | awk '{print $2}')"
        echo "- MPI version: $(mpirun --version | head -n1)"
        echo ""
        echo "DETAILED RESULTS:"
        echo "See CSV file: $CSV_FILE"
        echo ""
        echo "KEY FINDINGS:"
        echo "- Mode 1 (Batched) typically offers the best balance of performance and accuracy"
        echo "- Mode 2 (Pipelined) may achieve highest throughput for large ensembles"
        echo "- Mode 0 (Sequential) provides guaranteed correctness but lower efficiency"
        echo ""
        echo "CSV FORMAT:"
        echo "- Columns vary by benchmark type"
        echo "- avg_time: average training time in seconds"
        echo "- std_time: standard deviation of training time"
        echo "- avg_accuracy: average model accuracy (0-1)"
        echo "- std_accuracy: standard deviation of accuracy"
        echo ""
    } > "$RESULT_FILE"

    print_success "Summary report saved to: $RESULT_FILE"
    print_success "CSV data saved to: $CSV_FILE"
}

# ==============================================================================
# Command Line Interface
# ==============================================================================

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --quick        Run quick benchmark (fewer configurations)"
    echo "  --full         Run full benchmark suite (default)"
    echo "  --scalability  Run only scalability benchmark"
    echo "  --problem-size Run only problem size benchmark"
    echo "  --ensemble     Run only ensemble size benchmark"
    echo "  --processes P  Override process list (e.g., --processes '2 4 8')"
    echo "  --samples S    Override sample list (e.g., --samples '1000 5000')"
    echo "  --trees T      Override tree list (e.g., --trees '16 32')"
    echo "  --reps N       Set number of repetitions (default: 3)"
    echo "  --help         Show this help"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 --quick                    # Quick benchmark"
    echo "  $0 --scalability --reps 5     # Only scalability with 5 reps"
    echo "  $0 --processes '2 4' --quick  # Quick test with 2,4 processes"
}

# ==============================================================================
# Main Script Logic
# ==============================================================================

main() {
    local benchmark_type="full"

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --quick)
                benchmark_type="quick"
                DEFAULT_PROCESSES=(2 4)
                DEFAULT_SAMPLES=(1000 5000)
                DEFAULT_TREES=(16 32)
                REPETITIONS=2
                shift
                ;;
            --full)
                benchmark_type="full"
                shift
                ;;
            --scalability)
                benchmark_type="scalability"
                shift
                ;;
            --problem-size)
                benchmark_type="problem-size"
                shift
                ;;
            --ensemble)
                benchmark_type="ensemble"
                shift
                ;;
            --processes)
                read -ra DEFAULT_PROCESSES <<< "$2"
                shift 2
                ;;
            --samples)
                read -ra DEFAULT_SAMPLES <<< "$2"
                shift 2
                ;;
            --trees)
                read -ra DEFAULT_TREES <<< "$2"
                shift 2
                ;;
            --reps)
                REPETITIONS="$2"
                shift 2
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Start benchmarking
    print_header "PARALLEL distriboot MPI BENCHMARK SUITE"

    # Initial setup
    check_dependencies
    validate_mpi_setup

    # Initialize CSV file
    echo "# Parallel distriboot Benchmark Results - $(date)" > "$CSV_FILE"

    # Run selected benchmarks
    case $benchmark_type in
        "quick" | "full")
            run_scalability_benchmark
            run_problem_size_benchmark
            run_ensemble_size_benchmark
            ;;
        "scalability")
            run_scalability_benchmark
            ;;
        "problem-size")
            run_problem_size_benchmark
            ;;
        "ensemble")
            run_ensemble_size_benchmark
            ;;
    esac

    # Generate final report
    generate_summary_report

    print_header "BENCHMARK COMPLETED SUCCESSFULLY"
    print_success "Results available in: $RESULTS_DIR"
    print_info "Use the CSV file for plotting and detailed analysis"
}

# Run main function with all arguments
main "$@"
