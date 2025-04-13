#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nn.h"

#define GRID_W 28
#define GRID_H 28
#define SCR_W 2900 
#define SCR_H 1400 

const int PAD = 50;
const double BRUSH_R = 12.0;   
const Color BG_COL = SKYBLUE; 
const Color FG_COL = RED;   
const int FONT_SZ = 12; 

void flatten2D(double input2D[28][28], double output1D[784]) {
    int index = 0;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            output1D[index++] = input2D[i][j];
        }
    }
}

void UpdateGrid(RenderTexture2D tex, double grid[GRID_H][GRID_W]) {
    Image img = LoadImageFromTexture(tex.texture);
    if (img.data == NULL) { printf("Error: LoadImageFromTexture failed.\n"); return; }
    ImageFlipVertical(&img); 

    double cw = (double)img.width / GRID_W;
    double ch = (double)img.height / GRID_H;

    for (int gy = 0; gy < GRID_H; gy++) {
        for (int gx = 0; gx < GRID_W; gx++) {
            int sx = (int)round(gx * cw);
            int sy = (int)round(gy * ch);
            int ex = (int)round((gx + 1) * cw);
            int ey = (int)round((gy + 1) * ch);
            if (ex > img.width) ex = img.width;
            if (ey > img.height) ey = img.height;
            if (sx < 0) sx = 0;
            if (sy < 0) sy = 0;
            long marked = 0;
            long total = 0;
            for (int py = sy; py < ey; py++) {
                for (int px = sx; px < ex; px++) {
                    Color pix = GetImageColor(img, px, py);
                    if (pix.r != BG_COL.r || pix.g != BG_COL.g || pix.b != BG_COL.b || pix.a != BG_COL.a) {
                        marked++;
                    }
                    total++;
                }
            }
            grid[gy][gx] = (total > 0) ? ((double)marked / total) : 0.0;
        }
    }
    UnloadImage(img); 
}

int main(void) {
    int predicted_digit = -1;  // -1 means not yet predicted


    double input1D[784];
    srand(time(NULL));
     
     // Initialize the network with random weights and biases
     initializeNetwork(network_structure, n);
     
     // Import pre-trained weights and biases from files
     importNetwork();
    InitWindow(SCR_W, SCR_H, "Raylib 28x28 Grid");
    SetTargetFPS(60);

    const int marginB = 50;
    const int prvWRatio = 250; 
    const int txtWRatio = 400; 
    const int elemPad = 40;   
    const int availH = SCR_H - PAD * 2 - marginB;
    int drawSz = availH; 
    double sf = (double)drawSz / (700.0 - PAD*2 - marginB); 
    int prvW = (int)(prvWRatio * sf);
    int txtW = (int)(txtWRatio * sf);
    int totalW = drawSz + elemPad + prvW + elemPad + txtW;
    int padH = SCR_W - totalW;
    int padL = (padH > 0) ? padH / 2 : PAD; 

    Rectangle drawRect = {(float)padL, (float)PAD, (float)drawSz, (float)drawSz };
    Rectangle prvRect = {drawRect.x + drawRect.width + elemPad, drawRect.y, (float)prvW, (float)prvW };
    double prvCW = prvRect.width / GRID_W;
    double prvCH = prvRect.height / GRID_H;
    Rectangle txtRect = {prvRect.x + prvRect.width + elemPad, prvRect.y, (float)txtW, (float)prvW };
    double txtCW = txtRect.width / GRID_W;
    double txtCH = txtRect.height / GRID_H;

    RenderTexture2D canv = LoadRenderTexture(drawRect.width, drawRect.height);
    BeginTextureMode(canv); ClearBackground(BG_COL); EndTextureMode();

    double grid[GRID_H][GRID_W] = {0.0}; 
    bool drawing = false;
    Vector2 prevMp = { -1.0f, -1.0f }; 

    while (!WindowShouldClose()) {
        Vector2 mp = GetMousePosition();
        bool inDrawRect = CheckCollisionPointRec(mp, drawRect);
        Vector2 mpCanv = { mp.x - drawRect.x, mp.y - drawRect.y };

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && inDrawRect) {
            drawing = true;
            prevMp = mpCanv;
            BeginTextureMode(canv); DrawCircleV(mpCanv, (float)BRUSH_R, FG_COL); EndTextureMode();
        }
        if (drawing && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (inDrawRect) {
                BeginTextureMode(canv); DrawLineEx(prevMp, mpCanv, (float)(BRUSH_R * 2.0), FG_COL); EndTextureMode();
                prevMp = mpCanv; 
            } else { drawing = false; prevMp = (Vector2){ -1.0f, -1.0f }; }
        }
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
            drawing = false; prevMp = (Vector2){ -1.0f, -1.0f };
        }
        if (IsKeyPressed(KEY_C)) {
            BeginTextureMode(canv); ClearBackground(BG_COL); EndTextureMode();
            for(int y=0; y<GRID_H; ++y) for(int x=0; x<GRID_W; ++x) grid[y][x] = 0.0;
            drawing = false; prevMp = (Vector2){ -1.0f, -1.0f };
        }
        if (IsKeyPressed(KEY_ENTER)) {
            UpdateGrid(canv, grid);
            flatten2D(grid, input1D);
            feedForward(input1D);
            displayFinalOutput();
            predicted_digit = getPrediction();
        }

        BeginDrawing();
            ClearBackground(LIGHTGRAY); 
            DrawTextureRec(canv.texture, (Rectangle){ 0, 0, (float)canv.texture.width, -(float)canv.texture.height }, (Vector2){ drawRect.x, drawRect.y }, WHITE); 
            DrawRectangleLinesEx(drawRect, 2, DARKGRAY);
            DrawText("Drawing Area", drawRect.x, drawRect.y - 25, 20, DARKGRAY);

            DrawText("28x28 Preview", prvRect.x, prvRect.y - 25, 20, DARKGRAY);
            DrawRectangleLinesEx(prvRect, 1, DARKGRAY);
            for (int y = 0; y < GRID_H; y++) {
                for (int x = 0; x < GRID_W; x++) {
                    unsigned char gray = (unsigned char)(255.0 * (1.0 - grid[y][x])); 
                    Color cellCol = { gray, gray, gray, 255 };
                    DrawRectangle((int)(prvRect.x + x * prvCW), (int)(prvRect.y + y * prvCH), (int)ceil(prvCW), (int)ceil(prvCH), cellCol);
                }
            }

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

            DrawText("[LMB] Draw | [C] Clear | [Enter] Process Grid", PAD, SCR_H - 35, 20, DARKGRAY); 
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

    UnloadRenderTexture(canv); 
    CloseWindow();               
    return 0;
}
