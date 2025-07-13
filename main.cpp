#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define MAX_FEATURES 100
#define MAX_SAMPLES 10000
#define MAX_TREES 100
#define MAX_DEPTH 10

// Struct to represent a data sample.
typedef struct {
    double features[MAX_FEATURES];
    int label;
    double weight;
} Sample;

// Node in the decision tree
typedef struct TreeNode {
    int feature_idx;
    double threshold; // Entropy threshold to make the decision.
    int is_leaf; // No comments
    int prediction;
    double leaf_value; // Value in the leaf

    // Pointers to left and right child
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

// Decision tree
typedef struct {
    TreeNode* root;
    double alpha;  // Weight of the tree in the AdaBoost ensemble
} DecisionTree;

// Dataset representation.
typedef struct {
    Sample* samples;
    int num_samples;
    int num_features;
    int num_classes;
} Dataset;

// Ada-Boost model.
typedef struct {
    DecisionTree* trees;
    int num_trees;
    int num_features;
    int num_classes;
} AdaBoostModel;

// Global MPI variables.
int rank, size;

// Helper function to compute entropy.
double calculate_entropy(Sample* samples, int num_samples, int num_classes) {
    double* class_weights = (double*)calloc(num_classes, sizeof(double)); // Pre-allocate an initialized chunk of memory for the weighted sum per class.
    double entropy = 0.0;
    double total_weight = 0.0;

    for (int i = 0; i < num_samples; i++) {
        class_weights[samples[i].label] += samples[i].weight; // Add the weight of the current sample to its class.
        total_weight += samples[i].weight; // Increment the weight for this sample on AdaBoost algorithm.
    }

    // Compute entropy. Formula: H = -Σ(p_i * log2(p_i))
    for (int i = 0; i < num_classes; i++) { // For each class
        if (class_weights[i] > 0) { // If we have samples with weight
            double p = class_weights[i] / total_weight; // Calculate probability as weighted proportion
            entropy -= p * log2(p);
        }
    }

    free(class_weights); // Free the allocated memory for safety
    return entropy;
}

// Function to find the best split.
// We define the best split as the one that causes greater entropy reduction (as entropy is a synonym of the disorder or chaos in the source distribution).
void find_best_split(Sample* samples, int num_samples, int num_features, int num_classes,
    int* best_feature, double* best_threshold, double* best_gain) {
    *best_gain = -1.0;      // Initialize with impossible value so any real gain will be better
    *best_feature = -1;     // Initialize with invalid feature index
    *best_threshold = 0.0;  // Initialize threshold

    double parent_entropy = calculate_entropy(samples, num_samples, num_classes); // Compute entropy of current node (before split)

    for (int feature = 0; feature < num_features; feature++) { // Try each feature as potential split criterion
        // Order samples according the current feature.
        // Simple bubble sort - could be optimized with qsort() for better performance
        for (int i = 0; i < num_samples - 1; i++) {
            for (int j = i + 1; j < num_samples; j++) {
                if (samples[i].features[feature] > samples[j].features[feature]) {
                    Sample temp = samples[i];  // Swap samples to sort by current feature
                    samples[i] = samples[j];
                    samples[j] = temp;
                }
            }
        }

        // Try different thresholds (Greedily)
        for (int i = 0; i < num_samples - 1; i++) { // For each adjacent pair of samples
            double threshold = (samples[i].features[feature] + samples[i + 1].features[feature]) / 2.0; // Create threshold as midpoint between consecutive values

            // Divide on left and right
            int left_count = 0, right_count = 0; // Count how many samples go to each side
            for (int j = 0; j < num_samples; j++) { // Count samples for left and right splits
                if (samples[j].features[feature] <= threshold) {
                    left_count++; // Sample goes to left child
                }
                else {
                    right_count++; // Sample goes to right child
                }
            }

            if (left_count == 0 || right_count == 0) continue; // Skip if split puts all samples on one side (no information gain)

            Sample* left_samples = (Sample*)malloc(left_count * sizeof(Sample));   // Allocate memory for left split samples
            Sample* right_samples = (Sample*)malloc(right_count * sizeof(Sample)); // Allocate memory for right split samples

            int left_idx = 0, right_idx = 0; // Indices for filling left and right arrays
            for (int j = 0; j < num_samples; j++) { // Actually split the samples into left and right groups
                if (samples[j].features[feature] <= threshold) {
                    left_samples[left_idx++] = samples[j]; // Add sample to left group
                }
                else {
                    right_samples[right_idx++] = samples[j]; // Add sample to right group
                }
            }

            double left_entropy = calculate_entropy(left_samples, left_count, num_classes);   // Calculate entropy of left split
            double right_entropy = calculate_entropy(right_samples, right_count, num_classes); // Calculate entropy of right split

            // Calculate total weights for each split (CORRECTED)
            double left_total_weight = 0.0, right_total_weight = 0.0;
            for (int j = 0; j < left_count; j++) left_total_weight += left_samples[j].weight;   // Sum weights in left split
            for (int j = 0; j < right_count; j++) right_total_weight += right_samples[j].weight; // Sum weights in right split
            double total_weight = left_total_weight + right_total_weight; // Total weight of all samples

            double weighted_entropy = (left_total_weight * left_entropy + right_total_weight * right_entropy) / total_weight; // Calculate weighted average entropy after split using proper weights
            double gain = parent_entropy - weighted_entropy; // Information gain = entropy reduction achieved by this split

            if (gain > *best_gain) { // If this split is better than previous best
                *best_gain = gain;        // Update best information gain
                *best_feature = feature;  // Update best feature to split on
                *best_threshold = threshold; // Update best threshold value
            }

            free(left_samples);  // Free allocated memory for left split
            free(right_samples); // Free allocated memory for right split
        }
    }
}

// Function to build the tree recursively.
TreeNode* build_tree(Sample* samples, int num_samples, int num_features, int num_classes, int depth) {
    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));
    node->left = NULL;
    node->right = NULL;
    node->is_leaf = 0;

    // Stop conditions.
    if (depth >= MAX_DEPTH || num_samples < 2) {
        node->is_leaf = 1;
        // Find the most relevant class using weights (CORRECTED)
        double* class_weights = (double*)calloc(num_classes, sizeof(double)); // Use weights instead of counts
        double total_weight = 0.0;
        for (int i = 0; i < num_samples; i++) {
            class_weights[samples[i].label] += samples[i].weight; // Add weight to corresponding class
            total_weight += samples[i].weight; // Sum total weights
        }
        int max_class = 0;
        for (int i = 1; i < num_classes; i++) {
            if (class_weights[i] > class_weights[max_class]) { // Find class with highest total weight
                max_class = i;
            }
        }
        node->prediction = max_class;
        node->leaf_value = class_weights[max_class] / total_weight; // Proportion of weight for majority class
        free(class_weights);
        return node;
    }

    // Find best split.
    int best_feature;
    double best_threshold, best_gain;
    find_best_split(samples, num_samples, num_features, num_classes,
        &best_feature, &best_threshold, &best_gain);

    if (best_gain <= 0) { // No good split found, make it a leaf
        node->is_leaf = 1;
        double* class_weights = (double*)calloc(num_classes, sizeof(double)); // Use weights instead of counts
        double total_weight = 0.0;
        for (int i = 0; i < num_samples; i++) {
            class_weights[samples[i].label] += samples[i].weight; // Add weight to corresponding class
            total_weight += samples[i].weight; // Sum total weights
        }
        int max_class = 0;
        for (int i = 1; i < num_classes; i++) {
            if (class_weights[i] > class_weights[max_class]) { // Find class with highest total weight
                max_class = i;
            }
        }
        node->prediction = max_class;
        node->leaf_value = class_weights[max_class] / total_weight; // Proportion of weight for majority class
        free(class_weights);
        return node;
    }

    node->feature_idx = best_feature;
    node->threshold = best_threshold;

    // Divide the data.
    int left_count = 0, right_count = 0;
    for (int i = 0; i < num_samples; i++) {
        if (samples[i].features[best_feature] <= best_threshold) {
            left_count++;
        }
        else {
            right_count++;
        }
    }

    Sample* left_samples = (Sample*)malloc(left_count * sizeof(Sample));
    Sample* right_samples = (Sample*)malloc(right_count * sizeof(Sample));

    int left_idx = 0, right_idx = 0;
    for (int i = 0; i < num_samples; i++) {
        if (samples[i].features[best_feature] <= best_threshold) {
            left_samples[left_idx++] = samples[i];
        }
        else {
            right_samples[right_idx++] = samples[i];
        }
    }

    // Recursivelly build trees.
    node->left = build_tree(left_samples, left_count, num_features, num_classes, depth + 1);
    node->right = build_tree(right_samples, right_count, num_features, num_classes, depth + 1);

    free(left_samples);
    free(right_samples);

    return node;
}

// Función para predecir con un árbol
int predict_tree(TreeNode* node, double* features) {
    if (node->is_leaf) {
        return node->prediction;
    }

    if (features[node->feature_idx] <= node->threshold) {
        return predict_tree(node->left, features);
    }
    else {
        return predict_tree(node->right, features);
    }
}

// Function to train a weak learner with AdaBoost.
DecisionTree train_weak_learner(Dataset* dataset) {
    DecisionTree tree;
    tree.root = build_tree(dataset->samples, dataset->num_samples,
        dataset->num_features, dataset->num_classes, 0);

    // Calculate the error of the tree.
    double error = 0.0;
    double total_weight = 0.0;

    for (int i = 0; i < dataset->num_samples; i++) {
        int prediction = predict_tree(tree.root, dataset->samples[i].features);
        if (prediction != dataset->samples[i].label) {
            error += dataset->samples[i].weight;
        }
        total_weight += dataset->samples[i].weight;
    }

    error /= total_weight;

    // Calculate the weight of the tree for the final prediction.
    if (error > 0 && error < 0.5) {
        tree.alpha = 0.5 * log((1.0 - error) / error);
    }
    else {
        tree.alpha = 0.0;
    }

    return tree;
}

// Dummy function to generate synthetic data. Just for testing.
Dataset generate_synthetic_data(int num_samples, int num_features, int num_classes) {
    Dataset dataset;
    dataset.samples = (Sample*)malloc(num_samples * sizeof(Sample));
    dataset.num_samples = num_samples;
    dataset.num_features = num_features;
    dataset.num_classes = num_classes;

    srand(time(NULL) + rank);

    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_features; j++) {
            dataset.samples[i].features[j] = (double)rand() / RAND_MAX * 10.0 - 5.0;
        }

        // The label is generated based on a simple function which we are willing to approximate.
        double sum = 0.0;
        for (int j = 0; j < num_features; j++) {
            sum += dataset.samples[i].features[j];
        }
        dataset.samples[i].label = (sum > 0) ? 1 : 0;
        dataset.samples[i].weight = 1.0 / num_samples;
    }

    return dataset;
}

// Function to trait the adaboost trees in distributed mode.
AdaBoostModel train_adaboost_distributed(Dataset* dataset, int num_trees) {
    AdaBoostModel model;
    model.trees = (DecisionTree*)malloc(num_trees * sizeof(DecisionTree));
    model.num_trees = num_trees;
    model.num_features = dataset->num_features;
    model.num_classes = dataset->num_classes;

    // Uniformly initialize the weights.
    for (int i = 0; i < dataset->num_samples; i++) {
        dataset->samples[i].weight = 1.0 / dataset->num_samples;
    }

    for (int t = 0; t < num_trees; t++) {
        if (rank == 0) {
            printf("Entrenando árbol %d/%d...\n", t + 1, num_trees);
        }

        // Each process trains a tree on its cached local data.
        DecisionTree local_tree = train_weak_learner(dataset);

        // Synchronize results between trees (processes)
        double local_error = 0.0;
        double local_total_weight = 0.0;

        for (int i = 0; i < dataset->num_samples; i++) {
            int prediction = predict_tree(local_tree.root, dataset->samples[i].features);
            if (prediction != dataset->samples[i].label) {
                local_error += dataset->samples[i].weight;
            }
            local_total_weight += dataset->samples[i].weight;
        }

        // Combine errors.
        double global_error, global_total_weight;
        MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_total_weight, &global_total_weight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        global_error /= global_total_weight;

        // Compute global error
        double alpha = 0.0;
        if (global_error > 0 && global_error < 0.5) {
            alpha = 0.5 * log((1.0 - global_error) / global_error);
        }

        model.trees[t] = local_tree;
        model.trees[t].alpha = alpha;

        // Update weights.
        double weight_sum = 0.0;
        for (int i = 0; i < dataset->num_samples; i++) {
            int prediction = predict_tree(local_tree.root, dataset->samples[i].features);
            if (prediction != dataset->samples[i].label) {
                dataset->samples[i].weight *= exp(alpha);
            }
            else {
                dataset->samples[i].weight *= exp(-alpha);
            }
            weight_sum += dataset->samples[i].weight;
        }

        // Normalize weights
        for (int i = 0; i < dataset->num_samples; i++) {
            dataset->samples[i].weight /= weight_sum;
        }

        if (rank == 0) {
            printf("Tree %d trained. Error: %.4f, Alpha: %.4f\n", t + 1, global_error, alpha);
        }
    }

    return model;
}

// Function to predict using adaboots
int predict_adaboost(AdaBoostModel* model, double* features) {
    double* class_scores = (double*)calloc(model->num_classes, sizeof(double));

    for (int t = 0; t < model->num_trees; t++) {
        int prediction = predict_tree(model->trees[t].root, features);
        class_scores[prediction] += model->trees[t].alpha;
    }

    int best_class = 0;
    for (int i = 1; i < model->num_classes; i++) {
        if (class_scores[i] > class_scores[best_class]) {
            best_class = i;
        }
    }

    free(class_scores);
    return best_class;
}

// Function to evaluate the model.
double evaluate_model(AdaBoostModel* model, Dataset* test_data) {
    int correct = 0;
    for (int i = 0; i < test_data->num_samples; i++) {
        int prediction = predict_adaboost(model, test_data->samples[i].features);
        if (prediction == test_data->samples[i].label) {
            correct++;
        }
    }
    return (double)correct / test_data->num_samples;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Parameters
    int num_samples = 1000;
    int num_features = 10;
    int num_classes = 2;
    int num_trees = 20;

    if (rank == 0) {
        printf("=== DIstributed Boosted Decision tree ===\n");
        printf("MPI processes: %d\n", size);
        printf("Samples per process: %d\n", num_samples);
        printf("Features: %d\n", num_features);
        printf("Classes: %d\n", num_classes);
        printf("Number of trees: %d\n", num_trees);
        printf("==========================================\n");
    }

    // Generate data
    Dataset train_data = generate_synthetic_data(num_samples, num_features, num_classes);

    // Train the tree
    double start_time = MPI_Wtime();
    AdaBoostModel model = train_adaboost_distributed(&train_data, num_trees);
    double end_time = MPI_Wtime();

    // Generate test data
    Dataset test_data = generate_synthetic_data(num_samples / 4, num_features, num_classes);

    // Evaluate model
    double accuracy = evaluate_model(&model, &test_data);

    // Get the results back
    double global_accuracy;
    MPI_Reduce(&accuracy, &global_accuracy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        global_accuracy /= size;
        printf("\n=== Results ===\n");
        printf("Training time: %.4f seconds\n", end_time - start_time);
        printf("Avg precision: %.4f (%.2f%%)\n", global_accuracy, global_accuracy * 100);
        printf("==================\n");
    }

    // Free memory
    free(train_data.samples);
    free(test_data.samples);
    free(model.trees);

    MPI_Finalize();
    return 0;
}
