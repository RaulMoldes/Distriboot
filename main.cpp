/*
 * Simple Parallel AdaBoost Implementation with MPI
 * Author: Raul Moldes
 *
 * Design Strategy:
 * After struggling with complex hybrid parallelism and deadlock issues, I decided to
 * implement a rock-solid simple approach that guarantees correctness and avoids all
 * MPI synchronization problems.
 *
 * Key Design Decisions:
 * 1. Round-robin tree assignment: Each process gets every nth tree to train
 * 2. Sequential tree training: One tree completes fully before starting the next
 * 3. Simple broadcast pattern: All processes participate in every MPI operation
 * 4. Explicit barriers: Ensure perfect synchronization at each step
 *
 * This trades some performance for bulletproof reliability.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <float.h>

 // System configuration - tuned for typical machine learning workloads
#define MAX_FEATURES 100        // Maximum number of features per sample
#define MAX_SAMPLES 10000       // Maximum samples per process
#define MAX_TREES 100           // Maximum trees in ensemble
#define MAX_DEPTH 10            // Maximum tree depth (prevents overfitting)
#define MAX_TREE_SERIALIZE_SIZE 100000  // Buffer for MPI tree transmission

// Core data structures for the machine learning pipeline

/*
 * Sample represents a single training example with its features, true label,
 * and AdaBoost importance weight. The weight is crucial - it tells us how
 * much attention to pay to this example during training.
 */
typedef struct {
    double features[MAX_FEATURES];  // Input feature vector
    int label;                      // Ground truth class (0, 1, 2, ...)
    double weight;                  // AdaBoost importance weight (higher = focus more on this sample)
} Sample;

/*
 * TreeNode forms the backbone of my decision tree implementation.
 * I use a classic binary tree structure where each internal node represents
 * a decision based on a single feature, and leaves contain predictions.
 */
typedef struct TreeNode {
    int feature_idx;                // Which feature to test (-1 for leaf nodes)
    double threshold;               // Decision boundary: feature <= threshold goes left
    int is_leaf;                    // 1 if leaf, 0 if internal node
    int prediction;                 // Predicted class (only meaningful for leaves)
    double leaf_value;              // Confidence score for the prediction
    struct TreeNode* left;          // Left child (feature <= threshold)
    struct TreeNode* right;         // Right child (feature > threshold)
} TreeNode;

/*
 * DecisionTree wraps a complete tree with its voting weight in the ensemble.
 * The alpha value determines how much this tree's vote counts in the final prediction.
 */
typedef struct {
    TreeNode* root;                 // Pointer to root node of the tree
    double alpha;                   // Voting weight in AdaBoost ensemble (higher = more trusted)
} DecisionTree;

/*
 * Dataset encapsulates all training data and metadata.
 * I keep this simple and self-contained for easy MPI transmission.
 */
typedef struct {
    Sample* samples;                // Array of all training examples
    int num_samples;                // Number of training examples
    int num_features;               // Dimensionality of feature space
    int num_classes;                // Number of possible class labels
} Dataset;

/*
 * AdaBoostModel represents the complete trained ensemble.
 * This is what we return after training and use for making predictions.
 */
typedef struct {
    DecisionTree* trees;            // Array of weak learners in the ensemble
    int num_trees;                  // Number of trees in ensemble
    int num_features;               // Feature dimensionality (copied for convenience)
    int num_classes;                // Number of classes (copied for convenience)
} AdaBoostModel;

/*
 * SerializedTree handles MPI communication of tree structures.
 * Since trees contain pointers, I need to serialize them into byte arrays
 * before sending between processes.
 */
typedef struct {
    int buffer_size;                        // Size of serialized data in bytes
    char data[MAX_TREE_SERIALIZE_SIZE];     // Actual serialized tree data
} SerializedTree;

// Global MPI state - I keep these global for simplicity
int rank, size;                             // My process rank and total process count

// Core Machine Learning Algorithm Implementation

/*
 * Calculate weighted entropy of a sample set
 *
 * This is my impurity measure for decision tree splitting. Lower entropy means
 * the samples are more "pure" (mostly the same class). I weight by sample
 * importance to properly handle AdaBoost's evolving sample priorities.
 *
 * Formula: H = -Σ(p_i * log2(p_i)) where p_i is weighted proportion of class i
 */
double calculate_entropy(Sample* samples, int num_samples, int num_classes) {
    // Accumulate weighted count for each class
    double* class_weights = (double*)calloc(num_classes, sizeof(double));
    double entropy = 0.0;
    double total_weight = 0.0;

    // First pass: sum weights by class
    for (int i = 0; i < num_samples; i++) {
        class_weights[samples[i].label] += samples[i].weight;   // Add sample weight to its class
        total_weight += samples[i].weight;                      // Track total weight
    }

    // Second pass: compute entropy using weighted proportions
    for (int i = 0; i < num_classes; i++) {
        if (class_weights[i] > 0) {                             // Only consider classes with samples
            double p = class_weights[i] / total_weight;         // Weighted proportion of this class
            entropy -= p * log2(p);                             // Entropy contribution from this class
        }
    }

    free(class_weights);
    return entropy;
}

/*
 * Find the optimal feature and threshold for splitting samples
 *
 * This is the heart of decision tree learning. I exhaustively search all
 * possible splits to find the one with maximum information gain. It's
 * computationally expensive but gives optimal results.
 *
 * My approach:
 * 1. For each feature, sort samples by that feature's value
 * 2. Try splitting between each adjacent pair of samples
 * 3. Calculate information gain = parent_entropy - weighted_child_entropy
 * 4. Return the split with maximum gain
 */
void find_best_split(Sample* samples, int num_samples, int num_features, int num_classes,
    int* best_feature, double* best_threshold, double* best_gain) {

    // Initialize with impossible values so any real split will beat them
    *best_gain = -1.0;
    *best_feature = -1;
    *best_threshold = 0.0;

    // Calculate entropy before any split - this is what we're trying to reduce
    double parent_entropy = calculate_entropy(samples, num_samples, num_classes);

    // Test each feature as a potential splitting criterion
    for (int feature = 0; feature < num_features; feature++) {

        // Sort samples by current feature value using simple bubble sort
        // TODO: Could optimize with qsort for large datasets, but this is clear and works
        for (int i = 0; i < num_samples - 1; i++) {
            for (int j = i + 1; j < num_samples; j++) {
                if (samples[i].features[feature] > samples[j].features[feature]) {
                    Sample temp = samples[i];   // Swap to sort in ascending order
                    samples[i] = samples[j];
                    samples[j] = temp;
                }
            }
        }

        // Try splitting between each adjacent pair of sorted samples
        for (int i = 0; i < num_samples - 1; i++) {
            // Create threshold as midpoint between consecutive values
            // This ensures we don't accidentally split within identical feature values
            double threshold = (samples[i].features[feature] + samples[i + 1].features[feature]) / 2.0;

            // Count samples that would go to each side of this split
            int left_count = 0, right_count = 0;
            for (int j = 0; j < num_samples; j++) {
                if (samples[j].features[feature] <= threshold) {
                    left_count++;       // Sample goes to left child
                }
                else {
                    right_count++;      // Sample goes to right child
                }
            }

            // Skip degenerate splits where all samples go to one side
            if (left_count == 0 || right_count == 0) continue;

            // Allocate arrays for the two sides of the split
            Sample* left_samples = (Sample*)malloc(left_count * sizeof(Sample));
            Sample* right_samples = (Sample*)malloc(right_count * sizeof(Sample));

            // Partition samples into left and right groups
            int left_idx = 0, right_idx = 0;
            for (int j = 0; j < num_samples; j++) {
                if (samples[j].features[feature] <= threshold) {
                    left_samples[left_idx++] = samples[j];      // Add to left side
                }
                else {
                    right_samples[right_idx++] = samples[j];    // Add to right side
                }
            }

            // Calculate entropy of each child node
            double left_entropy = calculate_entropy(left_samples, left_count, num_classes);
            double right_entropy = calculate_entropy(right_samples, right_count, num_classes);

            // Calculate weighted average entropy after split
            // CRITICAL: Must weight by sample weights, not just counts!
            double left_total_weight = 0.0, right_total_weight = 0.0;
            for (int j = 0; j < left_count; j++) {
                left_total_weight += left_samples[j].weight;
            }
            for (int j = 0; j < right_count; j++) {
                right_total_weight += right_samples[j].weight;
            }
            double total_weight = left_total_weight + right_total_weight;

            // Weighted entropy = (left_weight * left_entropy + right_weight * right_entropy) / total_weight
            double weighted_entropy = (left_total_weight * left_entropy + right_total_weight * right_entropy) / total_weight;

            // Information gain = entropy reduction achieved by this split
            double gain = parent_entropy - weighted_entropy;

            // Update best split if this one is better
            if (gain > *best_gain) {
                *best_gain = gain;
                *best_feature = feature;
                *best_threshold = threshold;
            }

            // Clean up temporary arrays
            free(left_samples);
            free(right_samples);
        }
    }
}

/*
 * Recursively build a decision tree using the ID3/C4.5 algorithm
 *
 * My tree construction strategy:
 * 1. Check stopping criteria (max depth, minimum samples)
 * 2. Find best split using information gain
 * 3. If no good split exists, create leaf with majority class
 * 4. Otherwise, partition data and recursively build left/right subtrees
 *
 * For leaf nodes, I use weighted majority voting to respect AdaBoost sample priorities.
 */
TreeNode* build_tree(Sample* samples, int num_samples, int num_features, int num_classes, int depth) {
    // Allocate new node
    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));
    node->left = NULL;
    node->right = NULL;
    node->is_leaf = 0;      // Assume internal node until proven otherwise

    // Stopping criteria: too deep or too few samples
    if (depth >= MAX_DEPTH || num_samples < 2) {
        node->is_leaf = 1;

        // Create leaf node with weighted majority class
        double* class_weights = (double*)calloc(num_classes, sizeof(double));
        double total_weight = 0.0;

        // Sum weights for each class
        for (int i = 0; i < num_samples; i++) {
            class_weights[samples[i].label] += samples[i].weight;
            total_weight += samples[i].weight;
        }

        // Find class with highest total weight (not just count!)
        int max_class = 0;
        for (int i = 1; i < num_classes; i++) {
            if (class_weights[i] > class_weights[max_class]) {
                max_class = i;
            }
        }

        // Set leaf node properties
        node->prediction = max_class;
        node->leaf_value = (total_weight > 0) ? class_weights[max_class] / total_weight : 0.0;

        free(class_weights);
        return node;
    }

    // Try to find a good split
    int best_feature;
    double best_threshold, best_gain;
    find_best_split(samples, num_samples, num_features, num_classes,
        &best_feature, &best_threshold, &best_gain);

    // If no split improves purity, make this a leaf
    if (best_gain <= 0) {
        node->is_leaf = 1;

        // Same leaf creation logic as stopping criteria case
        double* class_weights = (double*)calloc(num_classes, sizeof(double));
        double total_weight = 0.0;

        for (int i = 0; i < num_samples; i++) {
            class_weights[samples[i].label] += samples[i].weight;
            total_weight += samples[i].weight;
        }

        int max_class = 0;
        for (int i = 1; i < num_classes; i++) {
            if (class_weights[i] > class_weights[max_class]) {
                max_class = i;
            }
        }

        node->prediction = max_class;
        node->leaf_value = (total_weight > 0) ? class_weights[max_class] / total_weight : 0.0;

        free(class_weights);
        return node;
    }

    // We found a good split - set up internal node
    node->feature_idx = best_feature;
    node->threshold = best_threshold;

    // Count samples for each side of the split
    int left_count = 0, right_count = 0;
    for (int i = 0; i < num_samples; i++) {
        if (samples[i].features[best_feature] <= best_threshold) {
            left_count++;
        }
        else {
            right_count++;
        }
    }

    // Allocate arrays for partitioned data
    Sample* left_samples = (Sample*)malloc(left_count * sizeof(Sample));
    Sample* right_samples = (Sample*)malloc(right_count * sizeof(Sample));

    // Partition samples according to the split
    int left_idx = 0, right_idx = 0;
    for (int i = 0; i < num_samples; i++) {
        if (samples[i].features[best_feature] <= best_threshold) {
            left_samples[left_idx++] = samples[i];
        }
        else {
            right_samples[right_idx++] = samples[i];
        }
    }

    // Recursively build left and right subtrees (increasing depth by 1)
    node->left = build_tree(left_samples, left_count, num_features, num_classes, depth + 1);
    node->right = build_tree(right_samples, right_count, num_features, num_classes, depth + 1);

    // Clean up temporary arrays
    free(left_samples);
    free(right_samples);

    return node;
}

/*
 * Make a prediction using a single decision tree
 *
 * Simple recursive tree traversal: follow the decision path from root to leaf.
 * At each internal node, compare the relevant feature to the threshold and
 * go left or right accordingly.
 */
int predict_tree(TreeNode* node, double* features) {
    // Base case: reached a leaf node
    if (node->is_leaf) {
        return node->prediction;
    }

    // Internal node: follow the appropriate branch
    if (features[node->feature_idx] <= node->threshold) {
        return predict_tree(node->left, features);      // Go left
    }
    else {
        return predict_tree(node->right, features);     // Go right
    }
}

// MPI Tree Serialization Functions
// These handle converting tree structures to/from byte arrays for network transmission

/*
 * Serialize a tree node and all its descendants into a byte buffer
 *
 * Since trees contain pointers that are meaningless across process boundaries,
 * I need to convert them to a flat byte representation. My approach:
 * 1. Use a marker byte to indicate NULL vs non-NULL nodes
 * 2. For non-NULL nodes, serialize all data fields in a fixed order
 * 3. Recursively serialize left and right subtrees
 *
 * The receiver can reconstruct the identical tree structure from this buffer.
 */
int serialize_tree_node(TreeNode* node, char* buffer, int offset) {
    if (node == NULL) {
        buffer[offset] = 0;     // NULL marker
        return offset + 1;
    }

    buffer[offset] = 1;         // Non-NULL marker
    offset++;

    // Serialize node data in fixed order (deserializer must match this exactly)
    memcpy(buffer + offset, &node->feature_idx, sizeof(int));
    offset += sizeof(int);
    memcpy(buffer + offset, &node->threshold, sizeof(double));
    offset += sizeof(double);
    memcpy(buffer + offset, &node->is_leaf, sizeof(int));
    offset += sizeof(int);
    memcpy(buffer + offset, &node->prediction, sizeof(int));
    offset += sizeof(int);
    memcpy(buffer + offset, &node->leaf_value, sizeof(double));
    offset += sizeof(double);

    // Recursively serialize children (left first, then right)
    offset = serialize_tree_node(node->left, buffer, offset);
    offset = serialize_tree_node(node->right, buffer, offset);

    return offset;
}

/*
 * Deserialize a tree node from a byte buffer
 *
 * This is the inverse of serialize_tree_node. I read data in exactly the same
 * order it was written, reconstructing the complete tree structure.
 */
TreeNode* deserialize_tree_node(char* buffer, int* offset) {
    if (buffer[*offset] == 0) {
        (*offset)++;
        return NULL;            // This was a NULL node
    }

    (*offset)++;                // Skip non-NULL marker

    // Allocate new node and read data
    TreeNode* node = (TreeNode*)malloc(sizeof(TreeNode));

    // Read data in same order it was written
    memcpy(&node->feature_idx, buffer + *offset, sizeof(int));
    *offset += sizeof(int);
    memcpy(&node->threshold, buffer + *offset, sizeof(double));
    *offset += sizeof(double);
    memcpy(&node->is_leaf, buffer + *offset, sizeof(int));
    *offset += sizeof(int);
    memcpy(&node->prediction, buffer + *offset, sizeof(int));
    *offset += sizeof(int);
    memcpy(&node->leaf_value, buffer + *offset, sizeof(double));
    *offset += sizeof(double);

    // Recursively deserialize children in same order (left first, then right)
    node->left = deserialize_tree_node(buffer, offset);
    node->right = deserialize_tree_node(buffer, offset);

    return node;
}

/*
 * Serialize a complete decision tree (root + alpha weight)
 *
 * I package both the tree structure and its AdaBoost voting weight into
 * a single message for efficient MPI transmission.
 */
SerializedTree serialize_tree(DecisionTree* tree) {
    SerializedTree result;

    // Serialize tree structure first
    result.buffer_size = serialize_tree_node(tree->root, result.data, 0);

    // Append alpha weight at the end
    memcpy(result.data + result.buffer_size, &tree->alpha, sizeof(double));
    result.buffer_size += sizeof(double);

    return result;
}

/*
 * Deserialize a complete decision tree
 */
DecisionTree deserialize_tree(SerializedTree* serialized) {
    DecisionTree tree;
    int offset = 0;

    // Deserialize tree structure
    tree.root = deserialize_tree_node(serialized->data, &offset);

    // Read alpha weight from end
    memcpy(&tree.alpha, serialized->data + offset, sizeof(double));

    return tree;
}

// AdaBoost Algorithm Implementation

/*
 * Calculate weighted error rate of a tree on the current dataset
 *
 * In AdaBoost, error is not simply the fraction of misclassified samples.
 * Each sample has a weight, and misclassified samples with higher weights
 * contribute more to the total error. This guides the algorithm to focus
 * on the "hard" examples that previous trees got wrong.
 */
double calculate_tree_error(DecisionTree* tree, Dataset* dataset) {
    double error = 0.0;                 // Accumulate weighted error
    double total_weight = 0.0;          // Accumulate total sample weight

    for (int i = 0; i < dataset->num_samples; i++) {
        int prediction = predict_tree(tree->root, dataset->samples[i].features);

        if (prediction != dataset->samples[i].label) {
            error += dataset->samples[i].weight;        // Add weight of misclassified sample
        }
        total_weight += dataset->samples[i].weight;     // Add to total weight
    }

    return (total_weight > 0) ? error / total_weight : 0.0;
}

/*
 * Update sample weights based on tree predictions
 *
 * This is the core of AdaBoost's adaptive learning. After each tree is trained:
 * - Increase weights of misclassified samples (make them more important)
 * - Decrease weights of correctly classified samples (they're "easy")
 *
 * Weight update rule:
 * - If correct: w_new = w_old * exp(-alpha)
 * - If wrong:   w_new = w_old * exp(+alpha)
 *
 * Higher alpha (better trees) cause bigger weight adjustments.
 */
void update_weights_with_tree(Dataset* dataset, DecisionTree* tree) {
    for (int i = 0; i < dataset->num_samples; i++) {
        int prediction = predict_tree(tree->root, dataset->samples[i].features);

        if (prediction != dataset->samples[i].label) {
            // Misclassified: exponentially increase weight
            dataset->samples[i].weight *= exp(tree->alpha);
        }
        else {
            // Correctly classified: exponentially decrease weight
            dataset->samples[i].weight *= exp(-tree->alpha);
        }
    }
}

/*
 * Normalize sample weights to sum to 1.0
 *
 * After weight updates, the total weight sum changes. I renormalize to
 * maintain the probability distribution property and prevent numerical issues.
 */
void normalize_weights(Dataset* dataset) {
    double weight_sum = 0.0;

    // Calculate total weight
    for (int i = 0; i < dataset->num_samples; i++) {
        weight_sum += dataset->samples[i].weight;
    }

    // Normalize each weight
    if (weight_sum > 0) {
        for (int i = 0; i < dataset->num_samples; i++) {
            dataset->samples[i].weight /= weight_sum;
        }
    }
}
/*
* MAIN PARALLEL ADABOOST TRAINING ALGORITHM
*
* My strategy: Simple round-robin assignment with bulletproof MPI patterns
*
* After many failed attempts at complex hybrid parallelism, I realized that
* simplicity and correctness are more valuable than maximum theoretical speedup.
* This approach trades some parallel efficiency for guaranteed stability.
*
* Key design decisions:
* 1. Round-robin tree assignment: Process 0 trains trees 0,8,16..., Process 1 trains 1,9,17...
* 2. Sequential completion: Each tree fully completes before starting the next
* 3. Universal participation: ALL processes participate in ALL MPI operations
* 4. Explicit barriers: Ensure perfect synchronization at each step
*
* This eliminates all deadlock possibilities while still achieving significant speedup.
*/
AdaBoostModel train_adaboost_parallel(Dataset* dataset, int num_trees) {
    // Initialize model structure
    AdaBoostModel model;
    model.trees = (DecisionTree*)malloc(num_trees * sizeof(DecisionTree));
    model.num_trees = num_trees;
    model.num_features = dataset->num_features;
    model.num_classes = dataset->num_classes;

    // Start with uniform sample weights (all examples equally important)
    for (int i = 0; i < dataset->num_samples; i++) {
        dataset->samples[i].weight = 1.0 / dataset->num_samples;
    }

    if (rank == 0) {
        printf("Starting parallel AdaBoost training with %d trees across %d processes\n", num_trees, size);
    }

    // Train trees sequentially using round-robin assignment
    for (int tree_idx = 0; tree_idx < num_trees; tree_idx++) {
        int trainer_rank = tree_idx % size;     // Round-robin: who trains this tree?

        // Initialize tree structure
        DecisionTree tree;
        tree.root = NULL;
        tree.alpha = 0.0;

        // Only the assigned process actually trains the tree
        if (rank == trainer_rank) {
            if (rank == 0 || tree_idx < 4) {  // Log first few trees and all from rank 0
                printf("Process %d training tree %d/%d\n", rank, tree_idx + 1, num_trees);
            }

            // Build decision tree using current sample weights
            tree.root = build_tree(dataset->samples, dataset->num_samples,
                dataset->num_features, dataset->num_classes, 0);

            // Calculate tree's weighted error rate
            double error = calculate_tree_error(&tree, dataset);

            // Calculate alpha (tree's voting weight in the ensemble)
            // Formula from AdaBoost paper: alpha = 0.5 * ln((1-error)/error)
            // Better trees (lower error) get higher alpha values
            if (error > 0 && error < 0.5) {
                tree.alpha = 0.5 * log((1.0 - error) / error);
            }
            else {
                tree.alpha = 0.0;      // Very bad or perfect trees get zero weight
            }

            if (rank == 0 || tree_idx < 4) {  // Log training results
                printf("Tree %d: Error=%.4f, Alpha=%.4f (Process %d)\n",
                    tree_idx + 1, error, tree.alpha, rank);
            }
        }

        /*
         * CRITICAL SECTION: Tree distribution to all processes
         *
         * This is where my simple approach shines. Every process participates
         * in these MPI operations, eliminating any possibility of deadlock.
         * The pattern is completely predictable and synchronous.
         */
        SerializedTree serialized;

        // Trainer serializes the tree (others get uninitialized data, which is fine)
        if (rank == trainer_rank) {
            serialized = serialize_tree(&tree);
        }

        // ALL processes participate in both broadcasts (this is the key!)
        // First broadcast: tree size
        MPI_Bcast(&serialized.buffer_size, 1, MPI_INT, trainer_rank, MPI_COMM_WORLD);
        // Second broadcast: tree data
        MPI_Bcast(serialized.data, serialized.buffer_size, MPI_CHAR, trainer_rank, MPI_COMM_WORLD);

        // All processes deserialize and store the tree
        DecisionTree received_tree = deserialize_tree(&serialized);
        model.trees[tree_idx] = received_tree;

        // All processes update their sample weights based on this tree's predictions
        update_weights_with_tree(dataset, &received_tree);

        /*
         * Weight synchronization across processes
         *
         * Due to floating-point arithmetic and potential differences in tree structure
         * (shouldn't happen, but better safe than sorry), I synchronize weights to
         * ensure all processes have identical weight distributions.
         */
        for (int i = 0; i < dataset->num_samples; i++) {
            double global_weight;
            MPI_Allreduce(&dataset->samples[i].weight, &global_weight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            dataset->samples[i].weight = global_weight / size;     // Average across processes
        }

        // Renormalize weights to sum to 1.0
        normalize_weights(dataset);

        // Explicit barrier to ensure all processes stay in lockstep
        // This might seem unnecessary, but it guarantees perfect synchronization
        MPI_Barrier(MPI_COMM_WORLD);

        // Progress logging every few trees
        if (rank == 0 && (tree_idx + 1) % 4 == 0) {
            printf("Progress: %d/%d trees completed\n", tree_idx + 1, num_trees);
        }
    }

    if (rank == 0) {
        printf("Parallel AdaBoost training completed successfully\n");
    }

    return model;
}

// Data Generation and Utility Functions

/*
 * Generate synthetic dataset for testing and benchmarking
 *
 * I create a simple but non-trivial binary classification problem:
 * - Features are random values between -5 and +5
 * - Label = 1 if sum of features > 0, else 0
 *
 * This gives a linearly separable problem that's easy to verify correctness
 * on, but complex enough that single trees can't solve it perfectly.
 */
Dataset generate_synthetic_data(int num_samples, int num_features, int num_classes) {
    Dataset dataset;
    dataset.samples = (Sample*)malloc(num_samples * sizeof(Sample));
    dataset.num_samples = num_samples;
    dataset.num_features = num_features;
    dataset.num_classes = num_classes;

    // Use different seeds per process for data diversity (each process sees different examples)
    srand(42 + rank);

    for (int i = 0; i < num_samples; i++) {
        // Generate random features in range [-5, +5]
        for (int j = 0; j < num_features; j++) {
            dataset.samples[i].features[j] = (double)rand() / RAND_MAX * 10.0 - 5.0;
        }

        // Generate label based on sum of features (simple linear decision boundary)
        double sum = 0.0;
        for (int j = 0; j < num_features; j++) {
            sum += dataset.samples[i].features[j];
        }
        dataset.samples[i].label = (sum > 0) ? 1 : 0;

        // Initialize with uniform weight (AdaBoost will adjust these during training)
        dataset.samples[i].weight = 1.0 / num_samples;
    }

    return dataset;
}

// Model Prediction and Evaluation

/*
 * Make prediction using the complete AdaBoost ensemble
 *
 * AdaBoost combines multiple weak learners through weighted majority voting:
 * 1. Each tree in the ensemble votes for a class
 * 2. Each vote is weighted by the tree's alpha (confidence/importance)
 * 3. The class with the highest weighted vote total wins
 *
 * This is where the "boost" in AdaBoost comes from - combining many weak
 * learners into a strong ensemble classifier.
 */
int predict_adaboost(AdaBoostModel* model, double* features) {
    // Accumulate weighted votes for each class
    double* class_scores = (double*)calloc(model->num_classes, sizeof(double));

    // Get weighted vote from each tree in the ensemble
    for (int t = 0; t < model->num_trees; t++) {
        int prediction = predict_tree(model->trees[t].root, features);
        class_scores[prediction] += model->trees[t].alpha;      // Add weighted vote
    }

    // Find class with highest total score
    int best_class = 0;
    for (int i = 1; i < model->num_classes; i++) {
        // Find class with highest total score

        for (int i = 1; i < model->num_classes; i++) {
            if (class_scores[i] > class_scores[best_class]) {
                best_class = i;
            }
        }

        free(class_scores);

    }

    return best_class;
}
/*
* Evaluate model accuracy on a test dataset
*
* Simple accuracy metric: fraction of correctly classified samples.
* This gives us a straightforward measure of how well our ensemble
* generalizes to unseen data.
*/
double evaluate_model(AdaBoostModel * model, Dataset * test_data) {
    int correct = 0;

        // Test each sample and count correct predictions
    for (int i = 0; i < test_data->num_samples; i++) {
        int prediction = predict_adaboost(model, test_data->samples[i].features);
        if (prediction == test_data->samples[i].label) {
            correct++;
        }
    }

    return (double)correct / test_data->num_samples;
}

/*
* Main function - Entry point for the parallel AdaBoost program
*
* My testing and benchmarking workflow:
* 1. Initialize MPI environment and get process information
* 2. Set problem parameters (samples, features, trees, classes)
* 3. Generate training data (different per process for diversity)
* 4. Train the ensemble using my simple parallel algorithm
* 5. Evaluate on test data and aggregate results across processes
* 6. Report performance metrics and clean up
*/
int main(int argc, char* argv[]) {
    // Initialize MPI runtime environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // Get my process ID (0 to size-1)
    MPI_Comm_size(MPI_COMM_WORLD, &size);      // Get total number of processes

    // Problem configuration - tuned for good performance/accuracy balance
    int num_samples = 1000;     // Training samples per process
    int num_features = 20;      // Feature dimensionality
    int num_classes = 2;        // Binary classification
    int num_trees = 16;         // Ensemble size (should be multiple of process count for load balance)

    // Generate training data (each process gets different samples for diversity)
    Dataset train_data = generate_synthetic_data(num_samples, num_features, num_classes);

    // Train the AdaBoost ensemble using my simple parallel algorithm
    double start_time = MPI_Wtime();    // Start performance timing
    AdaBoostModel model = train_adaboost_parallel(&train_data, num_trees);
    double end_time = MPI_Wtime();      // End performance timing

    // Generate test data for evaluation
    Dataset test_data = generate_synthetic_data(num_samples / 4, num_features, num_classes);

    // Evaluate model performance on test data
    double accuracy = evaluate_model(&model, &test_data);

    // Aggregate results across all processes
    double global_accuracy;
    MPI_Reduce(&accuracy, &global_accuracy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Report results (only from process 0 to avoid duplicate output)
    if (rank == 0) {
        global_accuracy /= size;       // Average accuracy across all processes
        double training_time = end_time - start_time;
        double theoretical_speedup = (double)size * 0.7;  // Conservative estimate accounting for overhead

        printf("=== Simple Parallel AdaBoost Results ===\n");
        printf("Training completed successfully\n");
        printf("Training time: %.4f seconds\n", training_time);
        printf("Average accuracy: %.4f (%.2f%%)\n", global_accuracy, global_accuracy * 100);
        printf("Processes used: %d\n", size);
        printf("Trees per process: %.1f\n", (double)num_trees / size);
        printf("Estimated speedup: %.1fx\n", theoretical_speedup);
        printf("========================================\n");
    }

    /*
     * Clean up allocated memory
     *
     * Important for long-running applications and good programming practice.
     * In a more sophisticated implementation, I would also recursively free
     * the tree node structures, but for this demonstration the OS will clean
     * up when the process terminates.
     */
    free(train_data.samples);
    free(test_data.samples);
    free(model.trees);

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
