#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h> // For DBL_MAX, DBL_MIN
#include "nn.h"   // NN functions are declared here

#define GRID_W 28
#define GRID_H 28
#define SCR_W 2000
#define SCR_H 1200

const int PAD = 50;
const double BRUSH_R = 10.0; 
const Color BG_COL = BLACK;
const Color FG_COL = WHITE;
const int FONT_SZ_INFO = 20;

// --- Function Declarations ---
void flatten2D(double input2D[GRID_H][GRID_W], double output1D[GRID_W * GRID_H]);
Rectangle CalculateBoundingBox(Image img, Color bgCol);
void CenterImage(Image srcImg, RenderTexture2D destTexture, Color bgCol);

// --- Function Definitions --- 

void flatten2D(double input2D[GRID_H][GRID_W], double output1D[GRID_W * GRID_H]) {
    int index = 0;
    for (int i = 0; i < GRID_H; i++) {
        for (int j = 0; j < GRID_W; j++) {
            output1D[index++] = input2D[i][j];
        }
    }
}
Rectangle CalculateBoundingBox(Image img, Color bgCol) {
    if (img.data == NULL) return (Rectangle){0, 0, 0, 0};
    int minX = img.width, minY = img.height, maxX = -1, maxY = -1;
    bool foundPixel = false;

    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            Color pix = GetImageColor(img, x, y);
            if (pix.r != bgCol.r || pix.g != bgCol.g || pix.b != bgCol.b || pix.a != bgCol.a) {
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
                foundPixel = true;
            }
        }
    }

    if (foundPixel) {
        return (Rectangle){(float)minX, (float)minY, (float)(maxX - minX + 1), (float)(maxY - minY + 1)};
    } else {
        return (Rectangle){0, 0, 0, 0};
    }
}

void CenterImage(Image srcImg, RenderTexture2D destTexture, Color bgCol) {
     if (srcImg.data == NULL) return;
    Rectangle bbox = CalculateBoundingBox(srcImg, bgCol);

    if (bbox.width <= 0 || bbox.height <= 0) {
        BeginTextureMode(destTexture);
        ClearBackground(bgCol);
        EndTextureMode();
        return;
    }

    float scaleX = (float)destTexture.texture.width / bbox.width;
    float scaleY = (float)destTexture.texture.height / bbox.height;
    float scale = (scaleX < scaleY) ? scaleX : scaleY;
    scale *= 0.60f; // Padding factor

    float destWidth = bbox.width * scale;
    float destHeight = bbox.height * scale;
    float destX = (destTexture.texture.width - destWidth) / 2.0f;
    float destY = (destTexture.texture.height - destHeight) / 2.0f;
    Rectangle destRect = { destX, destY, destWidth, destHeight };
    Rectangle sourceRect = bbox;

    Texture2D tempTex = LoadTextureFromImage(srcImg);

    BeginTextureMode(destTexture);
    ClearBackground(bgCol);
    DrawTexturePro(tempTex, sourceRect, destRect, (Vector2){0, 0}, 0.0f, WHITE);
    EndTextureMode();

    UnloadTexture(tempTex);
}

// --- Main Function ---
int main(void) {
    int predicted_digit = -1;
    double inputGrid[GRID_H][GRID_W] = {0.0};
    double input1D[GRID_W * GRID_H];

    // --- Initialize NN ---
    initializeNetwork(network_structure, n);
    importNetwork();

    // --- Setup Window & Layout ---
    InitWindow(SCR_W, SCR_H, "Raylib Digit Recognizer (Improved)");
    SetTargetFPS(90);

    const int marginB = 50, prvW = 280, txtW = 200, elemPad = 40;
    const int availH = SCR_H - PAD * 2 - marginB;
    int drawSz = availH;
    int padL = PAD;

    Rectangle drawRect = { (float)padL, (float)PAD, (float)drawSz, (float)drawSz };
    Rectangle prvRect = { drawRect.x + drawRect.width + elemPad, drawRect.y, (float)prvW, (float)prvW };
    double prvCW = prvRect.width / GRID_W, prvCH = prvRect.height / GRID_H;
    Rectangle txtRect = { prvRect.x + prvRect.width + elemPad, prvRect.y, (float)txtW, (float)prvW }; // Area for prediction text


    // --- Render Textures ---
    RenderTexture2D drawingCanvas = LoadRenderTexture(drawRect.width, drawRect.height);
    BeginTextureMode(drawingCanvas); ClearBackground(BG_COL); EndTextureMode();

    RenderTexture2D centeredCanvas = LoadRenderTexture(drawRect.width, drawRect.height);
    BeginTextureMode(centeredCanvas); ClearBackground(BG_COL); EndTextureMode();

    RenderTexture2D finalInputTexture = LoadRenderTexture(GRID_W, GRID_H);
    BeginTextureMode(finalInputTexture); ClearBackground(BG_COL); EndTextureMode();


    bool drawing = false;
    Vector2 prevMp = { -1.0f, -1.0f };

    // --- Main Loop ---
    while (!WindowShouldClose()) {
        Vector2 mp = GetMousePosition();
        bool inDrawRect = CheckCollisionPointRec(mp, drawRect);
        Vector2 mpCanv = { mp.x - drawRect.x, mp.y - drawRect.y };

        // --- Drawing Logic ---
        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && inDrawRect) {
            drawing = true;
            prevMp = mpCanv;
            BeginTextureMode(drawingCanvas); DrawCircleV(mpCanv, BRUSH_R, FG_COL); EndTextureMode();
        }
        if (drawing && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (inDrawRect) {
                BeginTextureMode(drawingCanvas);
                DrawCircleV(mpCanv, BRUSH_R, FG_COL);
                DrawLineEx(prevMp, mpCanv, BRUSH_R * 2.0f, FG_COL);
                EndTextureMode();
                prevMp = mpCanv;
            } else {
                drawing = false;
                prevMp = (Vector2){ -1.0f, -1.0f };
            }
        }
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
            drawing = false;
            prevMp = (Vector2){ -1.0f, -1.0f };
        }

        // --- Clear Logic ---
        if (IsKeyPressed(KEY_C)) {
            BeginTextureMode(drawingCanvas); ClearBackground(BG_COL); EndTextureMode();
            BeginTextureMode(centeredCanvas); ClearBackground(BG_COL); EndTextureMode();
            BeginTextureMode(finalInputTexture); ClearBackground(BG_COL); EndTextureMode();
            for (int y = 0; y < GRID_H; ++y)
                for (int x = 0; x < GRID_W; ++x)
                    inputGrid[y][x] = 0.0;
            predicted_digit = -1; // Reset prediction
            drawing = false;
            prevMp = (Vector2){ -1.0f, -1.0f };
        }

        // --- Process Logic (KEY_ENTER) ---
        if (IsKeyPressed(KEY_ENTER)) {
            Image drawnImage = LoadImageFromTexture(drawingCanvas.texture);

            CenterImage(drawnImage, centeredCanvas, BG_COL);
            Image centeredImg = LoadImageFromTexture(centeredCanvas.texture);
            Image finalImage = ImageCopy(centeredImg); // Work on a copy

            // Ensure format is grayscale and resize to final 28x28
            ImageFormat(&finalImage, PIXELFORMAT_UNCOMPRESSED_GRAYSCALE);
            ImageResize(&finalImage, GRID_W, GRID_H);

            // --- Directly populate inputGrid from the 28x28 finalImage ---
            if (finalImage.data != NULL && finalImage.width == GRID_W && finalImage.height == GRID_H) {
                unsigned char* pixels = (unsigned char*)finalImage.data;
                for (int y = 0; y < GRID_H; y++) {
                    for (int x = 0; x < GRID_W; x++) {
                        // Get the grayscale pixel value (0-255)
                        unsigned char intensity = pixels[y * GRID_W + x];
                        // Normalize to 0.0 (black) - 1.0 (white) for the NN input
                        inputGrid[y][x] = (double)intensity / 255.0;
                    }
                }
            } else {
                 // Handle error case if image processing failed
                 TraceLog(LOG_ERROR, "Failed to process image to 28x28 grayscale");
                 for (int y = 0; y < GRID_H; ++y) for (int x = 0; x < GRID_W; ++x) inputGrid[y][x] = 0.0;
            }

            flatten2D(inputGrid, input1D);
            feedForward(input1D);

            printf("\n--- Network Output Probabilities ---\n");
            displayFinalOutput(); // Prints probabilities to CONSOLE
            printf("------------------------------------\n");

            predicted_digit = getPrediction();

            UnloadImage(drawnImage);
            UnloadImage(centeredImg);
            UnloadImage(finalImage);
        }
        // --- Drawing Section ---
        BeginDrawing();
        ClearBackground(LIGHTGRAY);

        // Draw the raw drawing canvas
        DrawTextureRec(drawingCanvas.texture, (Rectangle){ 0, 0, (float)drawingCanvas.texture.width, (float)-drawingCanvas.texture.height }, (Vector2){ drawRect.x, drawRect.y }, WHITE);
        DrawRectangleLinesEx(drawRect, 2, DARKGRAY);
        DrawText("Drawing Area", drawRect.x, drawRect.y - 25, 20, DARKGRAY);

        // Draw the 28x28 preview
        DrawText("28x28 Input Preview", prvRect.x, prvRect.y - 25, 20, DARKGRAY);
        DrawRectangleLinesEx(prvRect, 1, DARKGRAY);
        for (int y = 0; y < GRID_H; y++) {
            for (int x = 0; x < GRID_W; x++) {
                unsigned char gray = (unsigned char)(255.0 * (1.0 - inputGrid[y][x])); // Invert for display
                Color cellCol = { gray, gray, gray, 255 };
                DrawRectangle((int)(prvRect.x + x * prvCW), (int)(prvRect.y + y * prvCH), (int)ceil(prvCW), (int)ceil(prvCH), cellCol);
            }
        }

        // --- Draw Prediction Text ---
        DrawRectangleRec(txtRect, Fade(SKYBLUE, 0.1f)); // Background for prediction text area
        DrawRectangleLinesEx(txtRect, 1, DARKGRAY);
        DrawText("Prediction Info", txtRect.x + 5, txtRect.y - 25, 20, DARKGRAY);

        if (predicted_digit != -1) {
            // Draw the main prediction
            const char *predLabel = TextFormat("Predicted: %d", predicted_digit);
            int predFontSize = 30;
            Vector2 predSize = MeasureTextEx(GetFontDefault(), predLabel, predFontSize, 1);
            DrawText(predLabel, (int)(txtRect.x + (txtRect.width - predSize.x)/2), (int)txtRect.y + 10, predFontSize, MAROON);

             // Add note about console output
            DrawText("Probabilities printed", txtRect.x + 10, txtRect.y + 60, FONT_SZ_INFO, DARKGRAY);
            DrawText("to console window.", txtRect.x + 10, txtRect.y + 60 + FONT_SZ_INFO + 2, FONT_SZ_INFO, DARKGRAY);

        } else {
             DrawText("Draw a digit", (int)txtRect.x + 10, (int)txtRect.y + 10, FONT_SZ_INFO, DARKGRAY);
             DrawText("and press [Enter]", (int)txtRect.x + 10, (int)txtRect.y + 10 + FONT_SZ_INFO + 2, FONT_SZ_INFO, DARKGRAY);
             DrawText("Check console for probs.", (int)txtRect.x + 10, (int)txtRect.y + 10 + 2*(FONT_SZ_INFO + 2), FONT_SZ_INFO, DARKGRAY);
        }


        DrawText("[LMB] Draw | [C] Clear | [Enter] Process", PAD, SCR_H - 35, 20, DARKGRAY);

        EndDrawing();
    }

    // --- Cleanup ---
    UnloadRenderTexture(drawingCanvas);
    UnloadRenderTexture(centeredCanvas);
    UnloadRenderTexture(finalInputTexture);
    CloseWindow();
    return 0;
}
