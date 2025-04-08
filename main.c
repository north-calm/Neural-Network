/*
 * Copyright (c) 2025 Satish Singh & Arman Badyal
 * All Rights Reserved.
 *
 * Unauthorized copying, modification, distribution, or use of this software,
 * via any medium, is strictly prohibited without explicit permission from
 * the author.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <time.h>
 #include <math.h>
 
 /* ========== Data Structures ========== */
 
 /**
  * Structure to represent a weight node in the linked list
  * Each weight connects a neuron to all neurons in the previous layer
  */
 typedef struct WeightNode {
     double weight;               // Weight value
     struct WeightNode *next;     // Pointer to next weight
     struct WeightNode *prev;     // Pointer to previous weight
 } Weight;
 
 /**
  * Structure to represent a neuron in the network
  * Each neuron has a bias, an output value, and weights to previous layer
  */
 typedef struct Neuron {
     Weight* weightNode;          // Linked list of weights to previous layer
     double bias;                 // Bias value for the neuron
     double value;                // Output value of the neuron after activation
     struct Neuron *next;         // Pointer to next neuron in the same layer
     struct Neuron *prev;         // Pointer to previous neuron in the same layer
 } Neuron;
 
 /**
  * Structure to represent a layer in the network
  * Each layer contains a linked list of neurons
  */
 typedef struct Layer {
     Neuron* neuron;              // Linked list of neurons in this layer
     struct Layer *next;          // Pointer to next layer
     struct Layer *prev;          // Pointer to previous layer
 } Layer;
 
 /* ========== Function Declarations ========== */
 void initializeNetwork(int [], int);
 int importNetwork(void);
 double relu(double);
 void softmax(double*, double*, int);
 void feedForward(double []);
 void displayFinalOutput(void);
 
 /* ========== Global Variables ========== */
 int n = 3;                          // Number of layers in the network
 int network_structure[] = {784, 128, 10};  // Number of neurons per layer
 Layer *Network = NULL;              // Head of the network linked list (global access point)
 
 /* ========== Main Function ========== */
 int main() {
     // Seed the random number generator for different weights each run
     srand(time(NULL));
     
     // Initialize the network with random weights and biases
     initializeNetwork(network_structure, n);
     
     // Import pre-trained weights and biases from files
     importNetwork();
     
     // Test with first example input (a digit image in flattened format)
     printf("\n===== PROCESSING FIRST INPUT =====\n");
     double input1[784] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,
        0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
     };
     
     // Perform forward pass and display results
     feedForward(input1);
     displayFinalOutput();
     
     // Test with second example input
     printf("\n===== PROCESSING SECOND INPUT =====\n");
     double input2[784] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,
        0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0,
        0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,
        0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,
        0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,
        0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,
        0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,
        0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    };
     
     // Perform forward pass and display results
     feedForward(input2);
     displayFinalOutput();
     
     return 0;
 }
 
 /**
  * Initialize the neural network with the specified structure
  * 
  * @param structure Array containing number of neurons in each layer
  * @param n1 Number of layers in the network
  */
 void initializeNetwork(int structure[], int n1) {
     int num = 0;
     
     printf("\nInitializing Neural Network\n");
     
     Layer* head = NULL;    // Head of the layer linked list
     Layer* tail = NULL;    // Tail pointer to keep track of last layer
     
     // Create each layer in the network
     for(int i = 0; i < n1; i++) {
         Layer* l = (Layer*) malloc(sizeof(Layer));
         if (l == NULL) {
             fprintf(stderr, "Memory allocation failed for layer\n");
             exit(1);
         }
         
         num = structure[i];    // Number of neurons in current layer
         
         Neuron *n = NULL;      // Head of neuron linked list for this layer
         Neuron *n_tail = NULL; // Tail pointer for neurons
         
         // Create neurons for this layer
         for(int j = 0; j < num; j++) {
             Neuron* tempNeuron = (Neuron*) malloc(sizeof(Neuron));
             if (tempNeuron == NULL) {
                 fprintf(stderr, "Memory allocation failed for neuron\n");
                 exit(1);
             }
             
             // Initialize with random bias between -0.5 and 0.5
             tempNeuron->bias = (rand() / (double)RAND_MAX) - 0.5;
             
             Weight *w = NULL;     // Head of weight linked list
             Weight *w_tail = NULL; // Tail pointer for weights
             
             // Create weights for this neuron (except for input layer)
             if (i > 0) {
                 for(int k = 0; k < network_structure[i-1]; k++) {
                     Weight *tw = (Weight*)malloc(sizeof(Weight));
                     if (tw == NULL) {
                         fprintf(stderr, "Memory allocation failed for weight\n");
                         exit(1);
                     }
                     
                     // Random weight initialization between -1 and 1
                     tw->weight = (rand() / (double)RAND_MAX) * 2 - 1;
                     
                     // Add to weight linked list
                     if(w == NULL) {
                         w = tw;
                         w_tail = tw;
                         tw->prev = NULL;
                         tw->next = NULL;
                     } else {
                         tw->prev = w_tail;
                         w_tail->next = tw;
                         w_tail = tw;
                         w_tail->next = NULL;
                     }
                 }
             }
             
             tempNeuron->weightNode = w;  // Attach weights to the neuron
             
             // Add to neuron linked list
             if(n == NULL) {
                 n = tempNeuron;
                 n_tail = tempNeuron;
                 tempNeuron->prev = NULL;
                 tempNeuron->next = NULL;
             } else {
                 tempNeuron->prev = n_tail;
                 n_tail->next = tempNeuron;
                 n_tail = tempNeuron;
                 n_tail->next = NULL;
             }
         }
         
         l->neuron = n;  // Attach neurons to the layer
         
         // Add to layer linked list
         if(head == NULL) {
             head = l;
             tail = l;
             l->prev = NULL;
             l->next = NULL;
         } else {
             l->prev = tail;
             tail->next = l;
             tail = l;
             tail->next = NULL;
         }
     }
     
     Network = head;  // Set the global network pointer
     printf("Network initialization complete\n");
 }
 
 /**
  * ReLU activation function
  * Returns x if x >= 0, otherwise returns 0
  * 
  * @param x Input value
  * @return Activated value
  */
 double relu(double x) {
     return x >= 0 ? x : 0;
 }
 
 /**
  * Softmax activation function for output layer
  * Converts raw outputs to probability distribution
  * 
  * @param input Array of input values
  * @param output Array to store softmax results
  * @param size Size of the arrays
  */
 void softmax(double* input, double* output, int size) {
     // Find the maximum value for numerical stability
     double max_val = input[0];
     for (int i = 1; i < size; i++) {
         if (input[i] > max_val) {
             max_val = input[i];
         }
     }
     
     // Compute exp(x - max) for each element and sum them
     double sum = 0.0;
     for (int i = 0; i < size; i++) {
         output[i] = exp(input[i] - max_val);
         sum += output[i];
     }
     
     // Normalize by dividing each value by the sum
     for (int i = 0; i < size; i++) {
         output[i] /= sum;
     }
 }
 
 /**
  * Forward propagation through the network
  * Takes input array and propagates values through the network
  * 
  * @param input Array of input values (must match input layer size)
  */


 void feedForward(double input[]) {
     Layer* currentLayer = Network;
     Neuron* n = currentLayer->neuron;
     
     // Set input layer values directly from input array
     for(int i = 0; i < network_structure[0]; i++) {
         n->value = (double)input[i];
         n = n->next;
     }
     
     // Process each hidden and output layer
     currentLayer = currentLayer->next;
     int h = 1;  // Layer index counter
     
     while(currentLayer != NULL) {
         Neuron* currentNeuron = currentLayer->neuron;
         
         // Process each neuron in current layer
         for(int i = 0; i < network_structure[h]; i++) {
             double value = 0.0;
             Neuron* previousLayerNeuron = currentLayer->prev->neuron;
             Weight* w = currentNeuron->weightNode;
             
             // Calculate weighted sum from previous layer
             for(int j = 0; j < network_structure[h-1]; j++) {
                 value += (w->weight) * (previousLayerNeuron->value);
                 previousLayerNeuron = previousLayerNeuron->next;
                 w = w->next;
             }
             
             // Apply activation function (ReLU) with bias
             if(h!= 2) {
                 currentNeuron->value = relu(value + (currentNeuron->bias));
             } else { // For output layer, apply softmax later
                 currentNeuron->value = value + (currentNeuron->bias);
             }

             currentNeuron = currentNeuron->next;
         }
         
         currentLayer = currentLayer->next;
         h++;
     }
 }
 
 /**
  * Display the output layer values with softmax applied
  * Shows the final prediction probabilities for each class
  */
 void displayFinalOutput() {
     double output[10] = {0};      // Raw outputs
     double final_output[10] = {0}; // Softmax probabilities
     int i = 0;
     
     // Navigate to the last layer
     Layer* lastLayer = Network;
     while (lastLayer->next != NULL) {
         lastLayer = lastLayer->next;
     }
     
     // Collect raw output values
     printf("Final Output (Class Probabilities):\n");
     Neuron* outputNeuron = lastLayer->neuron;
     while(outputNeuron != NULL) {
         output[i] = outputNeuron->value;
         outputNeuron = outputNeuron->next;
         i++;
     }
     
     // Apply softmax to get probabilities
     softmax(output, final_output, 10);
     
     // Display the probabilities for each class
     for(i = 0; i < 10; i++) {
         printf("Class %d: %lf\n", i, final_output[i]);
     }
     
     // Find and display the predicted class (highest probability)
     int prediction = 0;
     double max_prob = final_output[0];
     for(i = 1; i < 10; i++) {
         if(final_output[i] > max_prob) {
             max_prob = final_output[i];
             prediction = i;
         }
     }
     printf("\nPredicted digit: %d (confidence: %.2f%%)\n", prediction, max_prob*100);
 }
 
 /**
  * Import pre-trained weights and biases from files
  * 
  * @return 0 on success, 1 on failure
  */
 int importNetwork() {
     printf("Importing network parameters from files...\n");
     
     // Open model parameter files
     FILE *b1 = fopen("b1.txt", "r");  // Biases for first hidden layer
     FILE *b2 = fopen("b2.txt", "r");  // Biases for output layer
     FILE *w1 = fopen("W1_transpose.txt", "r");  // Weights for first hidden layer
     FILE *w2 = fopen("W2_transpose.txt", "r");  // Weights for output layer
     
     // Check if files opened successfully
     if (b1 == NULL || b2 == NULL || w1 == NULL || w2 == NULL) {
         perror("Error opening parameter files. Check if files exist in the same directory.");
         return 1;
     }
     
     // Verify network is initialized
     if(Network == NULL) {
         printf("Error: Network is not initialized\n");
         return 1;
     }
     
     // Navigate through network structure and import parameters
     int i = 0;
     Layer* head = Network;
     
     // Process each layer
     while (head != NULL) {
         Neuron* tn = head->neuron;
         
         // Process each neuron in the layer
         for(int j = 0; j < network_structure[i]; j++) {
             // Import biases for hidden and output layers
             if(i == 1) {
                 if(fscanf(b1, "%lf", &tn->bias) != 1) {
                     printf("Error reading bias for hidden layer\n");
                     return 1;
                 }
             } else if(i == 2) {
                 if(fscanf(b2, "%lf", &tn->bias) != 1) {
                     printf("Error reading bias for output layer\n");
                     return 1;
                 }
             }
             
             // Import weights (except for input layer which doesn't have weights)
             if(i != 0) {
                 Weight *tw = tn->weightNode;
                 while(tw != NULL) {
                     if(i == 1) {
                         if(fscanf(w1, "%lf", &tw->weight) != 1) {
                             printf("Error reading weight for hidden layer\n");
                             return 1;
                         }
                     } else if(i == 2) {
                         if(fscanf(w2, "%lf", &tw->weight) != 1) {
                             printf("Error reading weight for output layer\n");
                             return 1;
                         }
                     }
                     tw = tw->next;
                 }
             }
             
             tn = tn->next;
         }
         
         i++;
         head = head->next;
     }
     
     // Close all files
     fclose(b1);
     fclose(b2);
     fclose(w1);
     fclose(w2);
     
     printf("Network parameters imported successfully\n");
     return 0;
 }