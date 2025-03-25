#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Defining Structures

// Structure to represent a weight node in the linked list
struct WeightNode{
    double weight;
    struct WeightNode *next;
    struct WeightNode *prev;
};

typedef struct WeightNode Weight;

// Structure to represent a neuron in the network
struct Neuron{
    Weight* weightNode;
    double bias; // Bias value for the neuron
    double value; // output of each neuron
    struct Neuron *next;
    struct Neuron *prev;
};

typedef struct Neuron Neuron;

// Structure to represent a layer in the network
struct Layer{
    Neuron* neuron;
    struct Layer *next;
    struct Layer *prev;
};

typedef struct Layer Layer;


// Function declarations
void initializeNetwork(int [], int);
double sigmoid(double);
void feedForward(int []);
void displayFinalOutput();



// Global variables for network configuration
int n = 4; // Number of layers
int network_structure[] = {3, 5, 5, 2}; // Number of neurons per layer
Layer *Network = NULL; // Head of the network linked list



int main(){
    srand(time(NULL)); // Seeding the random function to generate different random values each run

    // Initializing the network
    initializeNetwork(network_structure, n);
    // example input
    int input[] = {3, 7, 12};

    //perform feedforward
    feedForward(input);
    //to see final outputs
    displayFinalOutput();


    return 0;
}


// Function to initialize the neural network
void initializeNetwork(int structure[], int n1){
    int num = 0;

    printf("\nInitializing Neural Network\n");

    Layer* head = NULL; // Head of the layer linked list
    Layer* tail = NULL; // Tail pointer to keep track of last layer


    // Adding layers to the network
    for(int i = 0; i < n1; i++){
        Layer* l = (Layer*) malloc(sizeof(Layer)); // Allocate memory for new layer
        num = structure[i]; // Number of neurons in current layer

        Neuron *n = NULL; // Head of neuron linked list
        Neuron *n_tail = NULL; // Tail pointer for neurons


        // Adding neurons to the layer
        for(int j = 0; j < num; j++){

            Neuron* tempNeuron = (Neuron*) malloc(sizeof(Neuron)); // Allocate memory for neuron
            tempNeuron->bias = (rand() / (double)RAND_MAX) - 0.5; // Assign a random bias

            Weight *w = NULL; // Head of weight linked list
            Weight *w_tail = NULL; // Tail pointer for weights


            // Assigning random weights (except for the first layer, which doesn't need weights)
            if (i>0) {
                for(int k = 0; k < network_structure[i-1]; k++){

                    Weight *tw = (Weight*)malloc(sizeof(Weight)); // Allocate memory for weight node
                      // Assign a random weight
                    tw->weight = (rand() / (double)RAND_MAX) - 0.5;  // ✅ Correct scaling

                    // Insert at the end of the weight linked list
                    if(w == NULL){
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

            tempNeuron->weightNode = w; // Attach the weight linked list to neuron

            // Insert neuron at the end of the neuron linked list
            if(n == NULL){
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

        l->neuron = n; // Attach neuron linked list to layer

        // Insert layer at the end of the layer linked list
        if(head == NULL){
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

    Network = head; // Set the global network head pointer
    printf("Done\n\n");
}
// Activation function (Sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
// Feedforward function
void feedForward(int input[]){
    Layer* currentLayer = Network;
    Neuron* n = currentLayer->neuron;
    // giving input values
    for(int i = 0; i < network_structure[0]; i++){
      n->value = (double)input[i];
      n = n->next;
    }
    currentLayer = currentLayer->next;
    int h = 1;
    while(currentLayer != NULL){
      Neuron* currentNeuron = currentLayer->neuron;
      for(int i = 0; i < network_structure[h]; i++){
        double value = 0.0;
        Neuron* previousLayerNeuron = currentLayer->prev->neuron;
        Weight* w = currentNeuron->weightNode;
        for(int j = 0; j < network_structure[h-1]; j++){
          value += (w->weight)*(previousLayerNeuron->value);
          previousLayerNeuron = previousLayerNeuron->next;
          w = w->next;
        }
        currentNeuron->value = sigmoid(value+(currentNeuron->bias));
        currentNeuron = currentNeuron->next;
      }
      currentLayer = currentLayer->next;
      h++;
    }
}
// i want to see the final outputs in the last layer
void displayFinalOutput(){
    Layer* lastLayer = Network;
    while (lastLayer->next != NULL) {
        lastLayer = lastLayer->next;
    }
    printf("Final Output:\n");
    Neuron* outputNeuron = lastLayer->neuron;
    while(outputNeuron != NULL){
      printf("%lf\n", outputNeuron->value);
      outputNeuron = outputNeuron->next;
    }

}
