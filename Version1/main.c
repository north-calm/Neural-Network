// ==== INCLUDED HEADERS ====

#include "raylib.h"         // Raylib for graphics, input, drawing
#include <stdio.h>          // Standard I/O
#include <stdlib.h>         // General utilities (e.g., memory, rand)
#include <math.h>           // For rounding and math operations
#include "nn.h"             // Your custom neural network header


// ==== DEFINES AND CONSTANTS ====

#define GRID_W 28           // Width of digit input grid (like MNIST)
#define GRID_H 28           // Height of digit input grid
#define SCR_W 2900          // Screen width in pixels
#define SCR_H 1400          // Screen height in pixels

const int PAD = 50;                     // General padding for UI layout
const double BRUSH_R = 12.0;            // Radius of brush for drawing on canvas
const Color BG_COL = SKYBLUE;           // Background color of the drawing canvas
const Color FG_COL = RED;               // Color used to draw (the "ink")
const int FONT_SZ = 12;                 // Font size for displaying grid cell values


// ==== FLATTENING 2D GRID TO 1D ARRAY ====

/*
 * Converts a 2D 28x28 grid into a 1D array of 784 values.
 * This is necessary because neural networks typically use 1D input arrays.
 */
void flatten2D(double input2D[28][28], double output1D[784]) {
    int index = 0;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            output1D[index++] = input2D[i][j];
        }
    }
}


// ==== CONVERT CANVAS DRAWING TO GRID ====

/*
 * Reads pixel data from the drawing canvas and maps it into a 28x28 grid.
 * Each grid cell gets a value between 0.0 (empty) to 1.0 (fully drawn).
 */
void UpdateGrid(RenderTexture2D tex, double grid[GRID_H][GRID_W]) {
    Image img = LoadImageFromTexture(tex.texture);   // Convert texture to image (CPU-readable)
    if (img.data == NULL) { 
        printf("Error: LoadImageFromTexture failed.\n"); 
        return; 
    }

    ImageFlipVertical(&img); // Flip because textures are upside-down in Raylib

    double cw = (double)img.width / GRID_W;  // Cell width in pixels
    double ch = (double)img.height / GRID_H; // Cell height in pixels

    // Go through each 28x28 grid cell
    for (int gy = 0; gy < GRID_H; gy++) {
        for (int gx = 0; gx < GRID_W; gx++) {
            // Determine pixel area this grid cell maps to
            int sx = (int)round(gx * cw);      // Start x
            int sy = (int)round(gy * ch);      // Start y
            int ex = (int)round((gx + 1) * cw);// End x
            int ey = (int)round((gy + 1) * ch);// End y

            // Clamp to image bounds
            if (ex > img.width) ex = img.width;
            if (ey > img.height) ey = img.height;
            if (sx < 0) sx = 0;
            if (sy < 0) sy = 0;

            long marked = 0; // Count of drawn pixels in this cell
            long total = 0;  // Total number of pixels in this cell

            // Count how many pixels are "ink" (i.e., not background)
            for (int py = sy; py < ey; py++) {
                for (int px = sx; px < ex; px++) {
                    Color pix = GetImageColor(img, px, py);
                    if (pix.r != BG_COL.r || pix.g != BG_COL.g || pix.b != BG_COL.b || pix.a != BG_COL.a) {
                        marked++;
                    }
                    total++;
                }
            }

            // Store fraction of cell covered by drawing
            grid[gy][gx] = (total > 0) ? ((double)marked / total) : 0.0;
        }
    }

    UnloadImage(img); // Free memory for image
}


// ==== MAIN APPLICATION ENTRY POINT ====

int main(void) {
    int predicted_digit = -1;        // -1 means no prediction yet

    double input1D[784];             // 1D input for neural net (flattened grid)
    srand(time(NULL));              // Random seed for any stochastic ops

    initializeNetwork(network_structure, n); // Set up NN layers
    importNetwork();                         // Load pre-trained weights from file

    // Create main window
    InitWindow(SCR_W, SCR_H, "Raylib 28x28 Grid");
    SetTargetFPS(60); // Set 60 frames per second


    // ==== LAYOUT CALCULATIONS ====

    const int marginB = 50;
    const int prvWRatio = 250;  // Size ratio of preview area
    const int txtWRatio = 400;  // Size ratio of text grid
    const int elemPad = 40;     // Padding between elements

    int drawSz = SCR_H - PAD * 2 - marginB;  // Size of drawing canvas
    double sf = (double)drawSz / (700.0 - PAD * 2 - marginB); // Scale factor

    // Preview and text areas
    int prvW = (int)(prvWRatio * sf);
    int txtW = (int)(txtWRatio * sf);
    int totalW = drawSz + elemPad + prvW + elemPad + txtW;
    int padH = SCR_W - totalW;
    int padL = (padH > 0) ? padH / 2 : PAD;

    // Define positions and sizes of each area
    Rectangle drawRect = {(float)padL, (float)PAD, (float)drawSz, (float)drawSz};
    Rectangle prvRect = {drawRect.x + drawRect.width + elemPad, drawRect.y, (float)prvW, (float)prvW};
    Rectangle txtRect = {prvRect.x + prvRect.width + elemPad, prvRect.y, (float)txtW, (float)prvW};

    // Preview cell sizes
    double prvCW = prvRect.width / GRID_W;
    double prvCH = prvRect.height / GRID_H;

    // Grid-text cell sizes
    double txtCW = txtRect.width / GRID_W;
    double txtCH = txtRect.height / GRID_H;

    // Canvas for drawing
    RenderTexture2D canv = LoadRenderTexture(drawRect.width, drawRect.height);
    BeginTextureMode(canv); ClearBackground(BG_COL); EndTextureMode();

    // 2D drawing grid (0 to 1 values)
    double grid[GRID_H][GRID_W] = {0.0};

    // Drawing state
    bool drawing = false;
    Vector2 prevMp = { -1.0f, -1.0f }; // Previous mouse position for smooth line drawing


    // ==== MAIN LOOP (RUNS UNTIL WINDOW CLOSED) ====

    while (!WindowShouldClose()) {
        // Get mouse position and adjust for canvas
        Vector2 mp = GetMousePosition();
        bool inDrawRect = CheckCollisionPointRec(mp, drawRect);
        Vector2 mpCanv = { mp.x - drawRect.x, mp.y - drawRect.y };

        // Start drawing
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && inDrawRect) {
            drawing = true;
            prevMp = mpCanv;
            BeginTextureMode(canv);
            DrawCircleV(mpCanv, (float)BRUSH_R, FG_COL);
            EndTextureMode();
        }

        // Draw smooth line when dragging
        if (drawing && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (inDrawRect) {
                BeginTextureMode(canv);
                DrawLineEx(prevMp, mpCanv, (float)(BRUSH_R * 2.0), FG_COL);
                EndTextureMode();
                prevMp = mpCanv;
            } else {
                drawing = false;
                prevMp = (Vector2){ -1.0f, -1.0f };
            }
        }

        // Stop drawing on mouse release
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
            drawing = false;
            prevMp = (Vector2){ -1.0f, -1.0f };
        }

        // Clear canvas and reset everything if user presses 'C'
        if (IsKeyPressed(KEY_C)) {
            BeginTextureMode(canv); ClearBackground(BG_COL); EndTextureMode();
            for(int y=0; y<GRID_H; ++y) for(int x=0; x<GRID_W; ++x) grid[y][x] = 0.0;
            drawing = false;
            prevMp = (Vector2){ -1.0f, -1.0f };
        }

        // Press ENTER to convert image to grid and run through NN
        if (IsKeyPressed(KEY_ENTER)) {
            UpdateGrid(canv, grid);         // Convert drawn image to grid
            flatten2D(grid, input1D);       // Flatten grid for neural net
            feedForward(input1D);           // Run input through network
            displayFinalOutput();           // Debug: print outputs
            predicted_digit = getPrediction(); // Get the predicted digit
        }


        // ==== DRAW EVERYTHING ====
        BeginDrawing();
            ClearBackground(LIGHTGRAY);

            // Show the canvas
            DrawTextureRec(canv.texture, (Rectangle){ 0, 0, (float)canv.texture.width, -(float)canv.texture.height }, (Vector2){ drawRect.x, drawRect.y }, WHITE); 
            DrawRectangleLinesEx(drawRect, 2, DARKGRAY);
            DrawText("Drawing Area", drawRect.x, drawRect.y - 25, 20, DARKGRAY);

            // Show the 28x28 grayscale preview
            DrawText("28x28 Preview", prvRect.x, prvRect.y - 25, 20, DARKGRAY);
            DrawRectangleLinesEx(prvRect, 1, DARKGRAY);
            for (int y = 0; y < GRID_H; y++) {
                for (int x = 0; x < GRID_W; x++) {
                    unsigned char gray = (unsigned char)(255.0 * (1.0 - grid[y][x])); 
                    Color cellCol = { gray, gray, gray, 255 };
                    DrawRectangle((int)(prvRect.x + x * prvCW), (int)(prvRect.y + y * prvCH), (int)ceil(prvCW), (int)ceil(prvCH), cellCol);
                }
            }

            // Show numerical grid values
            DrawText("Grid Values", txtRect.x, txtRect.y - 25, 20, DARKGRAY);
            DrawRectangleLinesEx(txtRect, 1, DARKGRAY);
            for (int y = 0; y < GRID_H; y++) {
                for (int x = 0; x < GRID_W; x++) {
                    const char *txt = TextFormat("%.2f", grid[y][x]);
                    double textX = txtRect.x + x * txtCW;
                    double textY = txtRect.y + y * txtCH;
                    Vector2 txtSz = MeasureTextEx(GetFontDefault(), txt, FONT_SZ, 1); 
                    double tx = textX + (txtCW - txtSz.x) / 2.0;
                    double ty = textY + (txtCH - txtSz.y) / 2.0;
                    Color txtCol = (grid[y][x] > 0.6) ? RAYWHITE : BLACK; 
                    if (grid[y][x] > 0.6) {
                        DrawRectangle((int)textX, (int)textY, (int)ceil(txtCW), (int)ceil(txtCH), BLACK);
                    }
                    DrawText(txt, (int)tx, (int)ty, FONT_SZ, txtCol);
                }
            }

            // Show control instructions
            DrawText("[LMB] Draw | [C] Clear | [Enter] Process Grid", PAD, SCR_H - 35, 20, DARKGRAY); 

            // Display predicted digit from neural network
            if (predicted_digit != -1) {
                const char *label = TextFormat("Predicted: %d", predicted_digit);
                int fontSize = 50;
                Vector2 size = MeasureTextEx(GetFontDefault(), label, fontSize, 2);
                int posX = SCR_W - PAD - (int)size.x - 100;
                int posY = SCR_H - PAD - (int)size.y - 100;
                DrawText(label, posX, posY, fontSize, MAROON);
            }
        EndDrawing();
    }

    // Cleanup and close
    UnloadRenderTexture(canv); 
    CloseWindow();               
    return 0;
}
