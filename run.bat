@echo off
REM Runs the engine. Thin wrapper around engine\build\oo_run.exe so you do not
REM have to remember the build path; every argument is passed straight through.
REM
REM   run                                              one logged game, r7 f8, all tactician
REM   run --list-agents
REM   run --radius 5 --factions 4 --agents tactician,greedy,greedy,random
REM   run --games 200 --agents tactician,marshal --rotate
REM   run --games 50 --agent marshal --replay 7        summary, plus game 7's replay
REM
REM A single game (the default) also writes board_state.json, which
REM web_visualizer.html opens. See engine\PLAN.md for the wider picture.

setlocal
set "EXE=%~dp0engine\build\oo_run.exe"
if not exist "%EXE%" (
  echo Engine not built yet. Build it with:
  echo   engine\dev.bat cmake -S engine -B engine\build -G Ninja -DCMAKE_BUILD_TYPE=Release
  echo   engine\dev.bat cmake --build engine\build
  exit /b 1
)
"%EXE%" %*
