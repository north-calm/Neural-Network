#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defining Structures

// Structure to represent a weight node in the linked list
struct WeightNode{
    int weight;
    struct WeightNode *next;
    struct WeightNode *prev;
};

typedef struct WeightNode Weight;

// Structure to represent a neuron in the network
struct Neuron{
    Weight* weightNode;
    int bias; // Bias value for the neuron
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
int displayNetwork();


// Global variables for network configuration
int n = 3; // Number of layers
int network_structure[] = {1,5,1}; // Number of neurons per layer
Layer *Network = NULL; // Head of the network linked list



int main(){
    srand(time(NULL)); // Seeding the random function to generate different random values each run

    // Initializing the network
    initializeNetwork(network_structure, n);
    
    // Display the network structure
    displayNetwork();

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
            tempNeuron->bias = rand(); // Assign a random bias
            
            Weight *w = NULL; // Head of weight linked list
            Weight *w_tail = NULL; // Tail pointer for weights
            

            // Assigning random weights (except for the first layer, which doesn't need weights)
            for(int k = 0; k < network_structure[i-1]; k++){

                Weight *tw = (Weight*)malloc(sizeof(Weight)); // Allocate memory for weight node
                tw->weight = rand();  // Assign a random weight
                
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

// Function to display the network structure
int displayNetwork(){
    if(Network == NULL){
        return printf("Network is Empty");
    }
    
    printf("Below is the structure of the network with random initialization.\n\n");
    printf("Format:\n-> Layer Address\n    -> Neurons' Bias\n        -> Weights\n\n");
    printf("\nStructure:\n\n");
    
    int i = 0;
    Layer* head = Network;
    
    // Traverse each layer
    while (head != NULL){
        printf("-> Layer %p\n", head);
        Neuron* tn = head->neuron;
        
        // Traverse each neuron in the layer
        for(int j = 0; j < network_structure[i]; j++){
            printf("    -> Bias: %d", tn->bias);

            // Print weights if not the first layer
            if(i != 0){
                Weight *tw = tn->weightNode;
                printf("\n        Weights: ");
                while(tw != NULL){
                    printf("-> %d", tw->weight);
                    tw = tw->next;
                }
                printf("\n");
            } else {
                printf("\n        No Weights Required\n");
            }
            
            tn = tn->next;
        }
        printf("\n");
        i++;
        head = head->next;
    }
    
    return 0;
}
