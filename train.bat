@echo off
REM Behaviour cloning, end to end (PLAN.md section 12.2, M10b).
REM
REM   train data                generate teacher games -> data\r7f8.bin
REM   train data 5000           a bigger dump (games count)
REM   train run                 train on it -> checkpoints\bc\best.pt
REM   train run --epochs 8      extra flags go straight to train.py
REM
REM Sizing: about 945 samples per game at r7f8 and ~3 KB each, so 1000 games is
REM roughly 1M samples and 3 GB on disk. Start there - the net is ~1M parameters
REM and does not need more to begin with.
REM
REM Uses goto rather than if-blocks on purpose: inside a parenthesised block,
REM %VAR% expands when the block is PARSED, so setting a variable and reading it
REM in the same block silently reads the old value.

setlocal
set "ROOT=%~dp0"
set "DUMP=%ROOT%engine\build\oo_dump_training.exe"

if /I "%~1"=="data" goto :data
if /I "%~1"=="run"  goto :run
goto :usage

:data
if not exist "%DUMP%" (
  echo Engine not built. Build it with:
  echo   engine\dev.bat cmake --build engine\build
  exit /b 1
)
set "GAMES=%~2"
if "%GAMES%"=="" set "GAMES=1000"
if not exist "%ROOT%data" mkdir "%ROOT%data"
"%DUMP%" --out "%ROOT%data\r7f8.bin" --games %GAMES% --agent tactician --radius 7 --factions 8
exit /b %errorlevel%

:run
REM %1..%9 only reaches nine arguments, so collect the rest into one variable.
REM Needs delayed expansion, hence !ARGS! - see the note at the top.
setlocal enabledelayedexpansion
set "ARGS="
:collect
shift
if "%~1"=="" goto :launch
set "ARGS=!ARGS! %1"
goto :collect
:launch
python "%ROOT%train\train.py" --data "%ROOT%data\r7f8.bin"!ARGS!
exit /b %errorlevel%

:usage
echo usage:
echo   train data [games]     generate teacher games into data\r7f8.bin
echo   train run [flags]      train on that file
exit /b 2
