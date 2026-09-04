@echo off
REM Enters the MSVC x64 build environment and runs whatever is passed in.
REM
REM Needed because the cl.exe on PATH is the 32-bit HostX86\x86 one while Python
REM (and everything we build) is 64-bit - see engine/PLAN.md §2. vcvars64 prints a
REM harmless "'vswhere.exe' is not recognized" warning on this machine; ignore it.
REM
REM   engine\dev.bat cmake -S engine -B engine\build -G Ninja
REM   engine\dev.bat cmake --build engine\build
REM   engine\dev.bat ctest --test-dir engine\build --output-on-failure

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
%*
