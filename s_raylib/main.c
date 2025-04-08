#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GRID_W 28
#define GRID_H 28
#define SCR_W 2900 
#define SCR_H 1400 

const int PAD = 50;
const float BRUSH_R = 12.0f;   
const Color BG_COL = SKYBLUE; 
const Color FG_COL = RED;   
const int FONT_SZ = 12; 

void UpdateGrid(RenderTexture2D tex, float grid[GRID_H][GRID_W]) {
    Image img = LoadImageFromTexture(tex.texture);
    if (img.data == NULL) { printf("Error: LoadImageFromTexture failed.\n"); return; }
    ImageFlipVertical(&img); 

    float cw = (float)img.width / GRID_W;
    float ch = (float)img.height / GRID_H;

    for (int gy = 0; gy < GRID_H; gy++) {
        for (int gx = 0; gx < GRID_W; gx++) {
            int sx = (int)roundf(gx * cw);
            int sy = (int)roundf(gy * ch);
            int ex = (int)roundf((gx + 1) * cw);
            int ey = (int)roundf((gy + 1) * ch);
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
            grid[gy][gx] = (total > 0) ? ((float)marked / total) : 0.0f;
        }
    }
    UnloadImage(img); 
}

int main(void) {
    InitWindow(SCR_W, SCR_H, "Raylib 28x28 Grid");
    SetTargetFPS(60);

    const int marginB = 50;
    const int prvWRatio = 250; 
    const int txtWRatio = 400; // Slightly wider for 28 columns
    const int elemPad = 40;   
    const int availH = SCR_H - PAD * 2 - marginB;
    int drawSz = availH; 
    float sf = (float)drawSz / (700.0f - PAD*2 - marginB); 
    int prvW = (int)(prvWRatio * sf);
    int txtW = (int)(txtWRatio * sf);
    int totalW = drawSz + elemPad + prvW + elemPad + txtW;
    int padH = SCR_W - totalW;
    int padL = (padH > 0) ? padH / 2 : PAD; 

    Rectangle drawRect = {(float)padL, (float)PAD, (float)drawSz, (float)drawSz };
    Rectangle prvRect = {drawRect.x + drawRect.width + elemPad, drawRect.y, (float)prvW, (float)prvW };
    float prvCW = prvRect.width / GRID_W;
    float prvCH = prvRect.height / GRID_H;
    Rectangle txtRect = {prvRect.x + prvRect.width + elemPad, prvRect.y, (float)txtW, (float)prvW };
    float txtCW = txtRect.width / GRID_W;
    float txtCH = txtRect.height / GRID_H;

    RenderTexture2D canv = LoadRenderTexture(drawRect.width, drawRect.height);
    BeginTextureMode(canv); ClearBackground(BG_COL); EndTextureMode();

    float grid[GRID_H][GRID_W] = {0.0f}; 
    bool drawing = false;
    Vector2 prevMp = { -1.0f, -1.0f }; 

    while (!WindowShouldClose()) {
        Vector2 mp = GetMousePosition();
        bool inDrawRect = CheckCollisionPointRec(mp, drawRect);
        Vector2 mpCanv = { mp.x - drawRect.x, mp.y - drawRect.y };

        if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && inDrawRect) {
            drawing = true;
            prevMp = mpCanv;
            BeginTextureMode(canv); DrawCircleV(mpCanv, BRUSH_R, FG_COL); EndTextureMode();
        }
        if (drawing && IsMouseButtonDown(MOUSE_LEFT_BUTTON)) {
            if (inDrawRect) {
                BeginTextureMode(canv); DrawLineEx(prevMp, mpCanv, BRUSH_R * 2.0f, FG_COL); EndTextureMode();
                prevMp = mpCanv; 
            } else { drawing = false; prevMp = (Vector2){ -1.0f, -1.0f }; }
        }
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
            drawing = false; prevMp = (Vector2){ -1.0f, -1.0f };
        }
        if (IsKeyPressed(KEY_C)) {
            BeginTextureMode(canv); ClearBackground(BG_COL); EndTextureMode();
            for(int y=0; y<GRID_H; ++y) for(int x=0; x<GRID_W; ++x) grid[y][x] = 0.0f;
            drawing = false; prevMp = (Vector2){ -1.0f, -1.0f };
        }
        if (IsKeyPressed(KEY_ENTER)) {
            UpdateGrid(canv, grid);
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
                    unsigned char gray = (unsigned char)(255.0f * (1.0f - grid[y][x])); 
                    Color cellCol = { gray, gray, gray, 255 };
                    DrawRectangle(prvRect.x + x * prvCW, prvRect.y + y * prvCH, ceilf(prvCW), ceilf(prvCH), cellCol);
                }
            }

            DrawText("Grid Values", txtRect.x, txtRect.y - 25, 20, DARKGRAY);
            DrawRectangleLinesEx(txtRect, 1, DARKGRAY);
             for (int y = 0; y < GRID_H; y++) {
                for (int x = 0; x < GRID_W; x++) {
                    const char *txt = TextFormat("%.2f", grid[y][x]);
                    float textX = txtRect.x + x * txtCW;
                    float textY = txtRect.y + y * txtCH;
                    Vector2 txtSz = MeasureTextEx(GetFontDefault(), txt, FONT_SZ, 1); 
                    float tx = textX + (txtCW - txtSz.x) / 2.0f;
                    float ty = textY + (txtCH - txtSz.y) / 2.0f;
                    Color txtCol = (grid[y][x] > 0.6f) ? RAYWHITE : BLACK; 
                    if (grid[y][x] > 0.6f) {
                         DrawRectangle(textX, textY, ceilf(txtCW), ceilf(txtCH), BLACK);
                    }
                    DrawText(txt, (int)tx, (int)ty, FONT_SZ, txtCol);
                }
            }

            DrawText("[LMB] Draw | [C] Clear | [Enter] Process Grid", PAD, SCR_H - 35, 20, DARKGRAY); 
        EndDrawing();
    }

    UnloadRenderTexture(canv); 
    CloseWindow();               
    return 0;
}