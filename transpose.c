// Program to transpost the matrix stored in file W1.txt(784X128) and W2.txt(128X10)

#include <stdio.h>

int main(){
    FILE *f1 = fopen("W1.txt", "r");
    FILE *f2 = fopen("W2.txt", "r");
    FILE *f3 = fopen("W1_transpose.txt", "w");
    FILE *f4 = fopen("W2_transpose.txt", "w");

    if (f1 == NULL || f2 == NULL || f3 == NULL || f4 == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    double matrix[784][128];
    double matrix2[128][10];

    // Read the first matrix from W1.txt
    for (int i = 0; i < 784; i++) {
        for (int j = 0; j < 128; j++) {
            fscanf(f1, "%lf", &matrix[i][j]);
        }
    }

    // Write the transposed matrix to W1_transpose.txt
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 784; j++) {
            fprintf(f3, "%.17lf ", matrix[j][i]);
        }
        fprintf(f3, "\n");
    }

    // Read the second matrix from W2.txt
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 10; j++) {
            fscanf(f2, "%lf", &matrix2[i][j]);
        }
    }

    // Write the transposed matrix to W2_transpose.txt
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 128; j++) {
            fprintf(f4, "%.17lf ", matrix2[j][i]);
        }
        fprintf(f4, "\n");
    }

    fclose(f1);
    fclose(f2);
    fclose(f3);
    fclose(f4);

    return 0;
}