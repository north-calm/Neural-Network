/*
 * Copyright (c) 2025 Satish Singh & Arman Badyal
 * All Rights Reserved.
 *
 * Unauthorized copying, modification, distribution, or use of this software,
 * via any medium, is strictly prohibited without explicit permission from
 * the author.
 */


// Include the Raylib library header for graphics, windowing, input, etc.
#include "raylib.h"
// Include standard input/output library (for printf, etc.)
#include <stdio.h>
// Include standard library (for malloc, free, rand, exit, etc. - potentially used in nn.h or elsewhere)
#include <stdlib.h>
// Include math library (for sqrt, pow, round, ceil, etc.)
#include <math.h>
// Include the custom neural network header file (defines network structures and functions)
#include "nn.h"

// --- Constants ---

// Define the width of the logical grid (used for processing the drawing)
#define GRID_W 28
// Define the height of the logical grid (used for processing the drawing)
#define GRID_H 28
// Define the width of the application window in pixels
#define SCR_W 1600
// Define the height of the application window in pixels
#define SCR_H 900

// Constant for padding around elements in the UI (in pixels)
const int PAD = 50;
// Constant for the radius of the drawing brush (in pixels)
const double BRUSH_R = 35.0;
// Constant defining the background color (Black)
const Color BG_COL = BLACK;
// Constant defining the foreground color (drawing color - White)
const Color FG_COL = WHITE;
// Constant defining a default font size (seems unused in the provided snippet, maybe used elsewhere)
const int FONT_SZ = 12;

// --- Function Definitions ---

/**
 * @brief Flattens a 2D array (grid) into a 1D array.
 * This is typically needed to prepare input for a fully connected neural network layer.
 * @param input2D The 28x28 2D array representing the grid.
 * @param output1D The 1D array (size 784) to store the flattened data.
 */
void flatten2D(double input2D[GRID_H][GRID_W], double output1D[GRID_W * GRID_H]) {
    int index = 0; // Initialize the index for the 1D array
    // Iterate through rows of the 2D array
    for (int i = 0; i < GRID_H; i++) {
        // Iterate through columns of the 2D array
        for (int j = 0; j < GRID_W; j++) {
            // Assign the 2D element to the corresponding position in the 1D array
            output1D[index++] = input2D[i][j];
        }
    }
}

/**
 * @brief Updates the 2D grid array based on the contents of a RenderTexture.
 * It samples the texture, calculating the average "color intensity" or "coverage"
 * for each cell in the logical grid.
 * @param tex The RenderTexture2D containing the user's drawing.
 * @param grid The 28x28 2D array to be updated with values (0.0 to 1.0).
 */
void UpdateGrid(RenderTexture2D tex, double grid[GRID_H][GRID_W]) {
    // Load the texture data into an Image structure in CPU memory
    Image img = LoadImageFromTexture(tex.texture);
    // Check if loading the image failed
    if (img.data == NULL) {
        printf("Error: LoadImageFromTexture failed.\n");
        return; // Exit the function if image loading failed
    }
    // Flip the image vertically. Textures often have origin at top-left,
    // while image processing or grid mapping might assume bottom-left.
    ImageFlipVertical(&img);

    // Calculate the width of one grid cell in pixels within the image
    double cw = (double)img.width / GRID_W;
    // Calculate the height of one grid cell in pixels within the image
    double ch = (double)img.height / GRID_H;

    // Iterate through each cell in the logical grid (rows)
    for (int gy = 0; gy < GRID_H; gy++) {
        // Iterate through each cell in the logical grid (columns)
        for (int gx = 0; gx < GRID_W; gx++) {
            // Calculate the starting x-pixel coordinate for this grid cell's region in the image
            int sx = (int)round(gx * cw);
            // Calculate the starting y-pixel coordinate for this grid cell's region in the image
            int sy = (int)round(gy * ch);
            // Calculate the ending x-pixel coordinate for this grid cell's region in the image
            int ex = (int)round((gx + 1) * cw);
            // Calculate the ending y-pixel coordinate for this grid cell's region in the image
            int ey = (int)round((gy + 1) * ch);

            // Clamp ending coordinates to image boundaries to prevent out-of-bounds access
            if (ex > img.width) ex = img.width;
            if (ey > img.height) ey = img.height;
            // Clamp starting coordinates (safety, though unlikely needed if gx/gy start at 0)
            if (sx < 0) sx = 0;
            if (sy < 0) sy = 0;

            // Counter for pixels within the region that are NOT the background color
            long marked = 0;
            // Counter for the total number of pixels checked within the region
            long total = 0;

            // Iterate through the pixels within the calculated region (y-coordinate)
            for (int py = sy; py < ey; py++) {
                // Iterate through the pixels within the calculated region (x-coordinate)
                for (int px = sx; px < ex; px++) {
                    // Get the color of the pixel at (px, py) in the image
                    Color pix = GetImageColor(img, px, py);
                    // Check if the pixel color is different from the defined background color
                    // (Compares R, G, B, and Alpha components)
                    if (pix.r != BG_COL.r || pix.g != BG_COL.g || pix.b != BG_COL.b || pix.a != BG_COL.a) {
                        // If it's not the background color, increment the 'marked' counter
                        marked++;
                    }
                    // Increment the total pixel counter for this region
                    total++;
                }
            }
            // Calculate the grid cell value: ratio of marked pixels to total pixels.
            // If total is 0 (shouldn't happen with valid sx/sy/ex/ey), value is 0.0.
            // This gives a value between 0.0 (empty) and 1.0 (fully marked).
            grid[gy][gx] = (total > 0) ? ((double)marked / total) : 0.0;
        }
    }
    // Unload the image data from CPU memory now that the grid is updated
    UnloadImage(img);
}

// --- Main Program Entry Point ---
int main(void) {
    // Variable to store the digit predicted by the neural network. Initialized to -1 (no prediction yet).
    int predicted_digit = -1;

    // 1D array to hold the flattened grid data (input for the neural network)
    // Size is GRID_W * GRID_H (28 * 28 = 784)
    double input1D[GRID_W * GRID_H];

    // --- Neural Network Setup ---
    // (Assumed functions from nn.h)
    // TODO: Define network_structure and n before calling initializeNetwork
    //       Example: int network_structure[] = {784, 128, 10}; int n = 3;
    // initializeNetwork(network_structure, n); // Initialize network architecture (layers, neurons)

    // Import pre-trained weights and biases from files (defined in nn.h/nn.c)
    // This loads the "learned" parameters of the network.
    importNetwork(); // Assumes nn.h provides this function

    // --- Raylib Initialization ---
    // Initialize the window with specified dimensions and title
    InitWindow(SCR_W, SCR_H, "Raylib MNIST Digit Recognizer");
    // Set the target frames per second (FPS)
    SetTargetFPS(90);

    // --- UI Layout Calculations ---
    const int marginB = 50;   // Bottom margin
    const int prvWRatio = 250; // Proportional width for the preview area (relative to a base size)
    const int txtWRatio = 400; // Proportional width for a text/output area (relative to a base size) - seems unused later
    const int elemPad = 40;   // Padding between UI elements (drawing area, preview)

    // Calculate the available height for drawing/preview elements, considering top/bottom padding and margin
    const int availH = SCR_H - PAD * 2 - marginB;
    // Set the drawing area size to be the maximum available height (making it square)
    int drawSz = availH;

    // Calculate a scaling factor 'sf' based on a reference height (700 minus padding/margin)
    // This allows elements like preview/text areas to scale proportionally if the window size changes (though window size is fixed here).
    // The 700 seems arbitrary, maybe from original design dimensions.
    double sf = (double)drawSz / (700.0 - PAD * 2 - marginB);
    // Calculate the actual width of the preview area based on its ratio and the scaling factor
    int prvW = (int)(prvWRatio * sf);
    // Calculate the actual width of the text area (unused in drawing loop)
    int txtW = (int)(txtWRatio * sf);
    // Calculate the total width required by the drawing area, preview area, and padding between them
    // Note: txtW is included here but the txtRect area isn't drawn in the loop below.
    int totalW = drawSz + elemPad + prvW + elemPad + txtW;
    // Calculate the horizontal padding needed to center the elements
    int padH = SCR_W - totalW;
    // Determine the left padding (half of the horizontal padding, or PAD if space is insufficient)
    int padL = (padH > 0) ? padH / 2 : PAD;

    // Define the rectangle for the main drawing area
    Rectangle drawRect = {(float)padL, (float)PAD, (float)drawSz, (float)drawSz };
    // Define the rectangle for the 28x28 grid preview area, positioned to the right of the drawing area
    Rectangle prvRect = {drawRect.x + drawRect.width + elemPad, drawRect.y, (float)prvW, (float)prvW };
    // Calculate the width of a single cell in the preview rectangle
    double prvCW = prvRect.width / GRID_W;
    // Calculate the height of a single cell in the preview rectangle
    double prvCH = prvRect.height / GRID_H;
    // Define the rectangle for the text area (unused in drawing loop), positioned to the right of the preview area
    Rectangle txtRect = {prvRect.x + prvRect.width + elemPad, prvRect.y, (float)txtW, (float)prvW }; // Note: Height uses prvW, likely meant prvW or drawSz
    // Calculate cell width for the text area (unused)
    double txtCW = txtRect.width / GRID_W;
    // Calculate cell height for the text area (unused)
    double txtCH = txtRect.height / GRID_H;

    // --- Canvas Setup ---
    // Create a RenderTexture. This is an off-screen buffer we can draw onto.
    // Size matches the drawing area dimensions.
    RenderTexture2D canv = LoadRenderTexture(drawRect.width, drawRect.height);
    // Begin drawing onto the render texture
    BeginTextureMode(canv);
    // Clear the render texture to the background color
    ClearBackground(BG_COL);
    // End drawing onto the render texture
    EndTextureMode();

    // --- State Variables ---
    // Initialize the 28x28 grid with all zeros (representing empty)
    double grid[GRID_H][GRID_W] = {0.0};
    // Flag to track if the user is currently drawing (mouse button held down)
    bool drawing = false;
    // Store the previous mouse position relative to the canvas, used for drawing lines
    // Initialized to an invalid position.
    Vector2 prevMp = { -1.0f, -1.0f };

    // --- Main Game Loop ---
    // Loop continues as long as the user hasn't closed the window (e.g., by clicking X or pressing Alt+F4)
    while (!WindowShouldClose()) {
        // Get the current mouse position in screen coordinates
        Vector2 mp = GetMousePosition();
        // Check if the mouse cursor is currently within the drawing area rectangle
        bool inDrawRect = CheckCollisionPointRec(mp, drawRect);
        // Calculate the mouse position relative to the top-left corner of the drawing canvas
        Vector2 mpCanv = { mp.x - drawRect.x, mp.y - drawRect.y };

        // --- Input Handling and Drawing Logic ---

        // Check if the left mouse button was just pressed AND the cursor is inside the drawing area
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && inDrawRect) {
            drawing = true;              // Set the drawing flag to true
            prevMp = mpCanv;             // Store the current canvas mouse position as the previous position
            // Begin drawing onto the render texture (canvas)
            BeginTextureMode(canv);
            // Draw a circle at the current mouse position to start the stroke
            DrawCircleV(mpCanv, (float)BRUSH_R, FG_COL);
            // End drawing onto the render texture
            EndTextureMode();
        }

        // Check if the user is currently drawing (button held down)
        if (drawing && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            // Check if the mouse is still inside the drawing area
            if (inDrawRect) {
                // Begin drawing onto the render texture
                BeginTextureMode(canv);
                // Draw a circle at the current mouse position
                DrawCircleV(mpCanv, (float)BRUSH_R, FG_COL);
                // Draw a thick line connecting the previous mouse position to the current one
                // This fills gaps when the mouse moves quickly. Thickness is 2 * radius.
                DrawLineEx(prevMp, mpCanv, (float)(BRUSH_R * 2.0), FG_COL);
                // End drawing onto the render texture
                EndTextureMode();
                // Update the previous mouse position for the next frame
                prevMp = mpCanv;
            } else {
                // If the mouse moved outside the drawing area while drawing, stop drawing
                drawing = false;
                prevMp = (Vector2){ -1.0f, -1.0f }; // Reset previous position
            }
        }

        // Check if the left mouse button was just released
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
            drawing = false; // Stop drawing
            prevMp = (Vector2){ -1.0f, -1.0f }; // Reset previous position
        }

        // Check if the 'C' key was just pressed
        if (IsKeyPressed(KEY_C)) {
            // Begin drawing onto the render texture
            BeginTextureMode(canv);
            // Clear the canvas back to the background color
            ClearBackground(BG_COL);
            // End drawing onto the render texture
            EndTextureMode();
            // Reset the logical grid array to all zeros
            for(int y=0; y<GRID_H; ++y) for(int x=0; x<GRID_W; ++x) grid[y][x] = 0.0;
            // Reset drawing state variables
            drawing = false;
            prevMp = (Vector2){ -1.0f, -1.0f };
            predicted_digit = -1; // Clear previous prediction
        }

        // Check if the 'Enter' key was just pressed
        if (IsKeyPressed(KEY_ENTER)) {
            // Update the 'grid' array based on the current state of the 'canv' render texture
            UpdateGrid(canv, grid);
            // Flatten the 2D 'grid' array into the 1D 'input1D' array for the neural network
            flatten2D(grid, input1D);
            // Feed the 'input1D' array forward through the neural network (defined in nn.h/nn.c)
            feedForward(input1D); // Assumes nn.h provides this
            // Display the final output layer activations (e.g., probabilities for each digit) - likely prints to console
            displayFinalOutput(); // Assumes nn.h provides this
            // Get the index of the highest activation, which corresponds to the predicted digit
            predicted_digit = getPrediction(); // Assumes nn.h provides this
        }

        // --- Drawing to the Screen ---
        BeginDrawing(); // Start drawing frame for the main window

        // Clear the main window background to light gray
        ClearBackground(LIGHTGRAY);

        // Draw the contents of the render texture (the canvas) onto the screen
        // The source rectangle flips the texture vertically (negative height) because
        // RenderTextures have origin at top-left, but we want to display it as drawn.
        DrawTextureRec(canv.texture, (Rectangle){ 0, 0, (float)canv.texture.width, -(float)canv.texture.height }, (Vector2){ drawRect.x, drawRect.y }, WHITE);
        // Draw an outline around the drawing area
        DrawRectangleLinesEx(drawRect, 2, DARKGRAY);
        // Draw a label above the drawing area
        DrawText("Drawing Area (28x28 logical)", drawRect.x, drawRect.y - 25, 20, DARKGRAY);

        // Draw a label above the preview area
        DrawText("Processed 28x28 Preview", prvRect.x, prvRect.y - 25, 20, DARKGRAY);
        // Draw an outline around the preview area
        DrawRectangleLinesEx(prvRect, 1, DARKGRAY);
        // Draw the 28x28 preview grid based on the 'grid' array data
        for (int y = 0; y < GRID_H; y++) {
            for (int x = 0; x < GRID_W; x++) {
                // Calculate grayscale value: 1.0 (white) - grid_value (0.0 to 1.0)
                // So, grid value 0.0 (empty) -> gray 255 (white), grid value 1.0 (full) -> gray 0 (black)
                // This inverts the color for preview compared to the drawing area.
                unsigned char gray = (unsigned char)(255.0 * (grid[y][x])); // Changed to show drawing directly
                // Create the color for the cell
                Color cellCol = { gray, gray, gray, 255 }; // Use the calculated gray value
                // Draw the rectangle for the current grid cell in the preview area
                // Use ceil() for width/height to avoid gaps due to floating point inaccuracies
                DrawRectangle((int)(prvRect.x + x * prvCW), (int)(prvRect.y + y * prvCH), (int)ceil(prvCW), (int)ceil(prvCH), cellCol);
            }
        }

        // Draw instruction text at the bottom of the screen
        DrawText("[LMB] Draw | [C] Clear | [Enter] Process Grid & Predict", PAD, SCR_H - 35, 20, DARKGRAY);

        // If a digit has been predicted (i.e., Enter was pressed)
        if (predicted_digit != -1) {
            // Format the prediction text string
            const char *label = TextFormat("Predicted: %d", predicted_digit);
            // Set font size for the prediction text
            int fontSize = 50;
            // Measure the size of the text to help with positioning
            Vector2 size = MeasureTextEx(GetFontDefault(), label, fontSize, 2); // Using default font
             // Calculate position to draw the text (e.g., bottom-right corner area)
            int posX = prvRect.x + prvRect.width + elemPad; // Position near the preview area
            int posY = prvRect.y + prvRect.height + elemPad; // Position below the preview area
             // Ensure text doesn't go off-screen (simple check)
            if (posX + size.x > SCR_W - PAD) posX = SCR_W - PAD - (int)size.x;
            if (posY + size.y > SCR_H - marginB) posY = SCR_H - marginB - (int)size.y;

            // Draw the prediction text on the screen
            DrawText(label, posX, posY, fontSize, MAROON); // Using MAROON color
        }

        EndDrawing(); // Finish drawing the frame for the main window
    }

    // --- Cleanup ---
    // Unload the render texture from memory
    UnloadRenderTexture(canv);
    // Close the Raylib window and release resources
    CloseWindow();

    // Return 0 to indicate successful program execution
    return 0;
}
