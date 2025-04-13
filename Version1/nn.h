#ifndef NN_H
#define NN_H

/* ========== Includes ========== */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* ========== Data Structures ========== */

// Weight linked list node
typedef struct WeightNode {
    double weight;
    struct WeightNode *next;
    struct WeightNode *prev;
} Weight;

// Neuron node containing output, bias, and weight links
typedef struct Neuron {
    Weight *weightNode;
    double bias;
    double value;
    struct Neuron *next;
    struct Neuron *prev;
} Neuron;

// Layer node containing list of neurons
typedef struct Layer {
    Neuron *neuron;
    struct Layer *next;
    struct Layer *prev;
} Layer;

/* ========== Global Variables ========== */

extern int n;
extern int network_structure[];
extern Layer *Network;

/* ========== Function Declarations ========== */

void initializeNetwork(int structure[], int layerCount);
int importNetwork(void);
double relu(double x);
void softmax(double *input, double *output, int length);
void feedForward(double input[]);
void displayFinalOutput(void);
int getPrediction(void);

#endif // NN_H
