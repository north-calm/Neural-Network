@echo off
echo Compiling main.c...

gcc main.c -o main.exe -I "C:\raylib-5.5_win64_mingw-w64\raylib-5.5_win64_mingw-w64\include" -L "C:\raylib-5.5_win64_mingw-w64\raylib-5.5_win64_mingw-w64\lib" -lraylib -lopengl32 -lgdi32 -lwinmm

if %errorlevel% neq 0 (
    echo Compilation failed!
    pause
    exit /b
)

echo Compilation successful! Running main.exe...
.\main.exe
