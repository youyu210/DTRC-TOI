@echo off
setlocal
cd /d "%~dp0.."
call "D:\tools\MicrosoftVisualStudio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /EHsc /I StiffGIPC tests\toi_budget_test.cpp /Fe:tests\toi_budget_test.exe /Fo:tests\toi_budget_test.obj
if errorlevel 1 exit /b %errorlevel%
tests\toi_budget_test.exe
exit /b %errorlevel%
