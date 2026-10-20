@echo off
setlocal
call "D:\tools\MicrosoftVisualStudio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b %errorlevel%
cd /d "%~dp0"
set "PATH=D:\tools\MicrosoftVisualStudio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
"D:\tools\MicrosoftVisualStudio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 "-DCMAKE_CXX_FLAGS=/utf-8" "-DCMAKE_CUDA_FLAGS_RELEASE=-O3 -DNDEBUG -Xcompiler=/utf-8" "-DCMAKE_TOOLCHAIN_FILE=C:/Users/hj0320/AppData/Local/Temp/vcpkg_src/vcpkg-master/scripts/buildsystems/vcpkg.cmake" "-DVCPKG_INSTALLED_DIR=C:/Users/hj0320/AppData/Local/Temp/vcpkg_installed" -DVCPKG_TARGET_TRIPLET=x64-windows
if errorlevel 1 exit /b %errorlevel%
"D:\tools\MicrosoftVisualStudio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Release --parallel 4
if errorlevel 1 exit /b %errorlevel%
copy /y "%~dp0..\build-ninja\cublas64_12.dll" "%~dp0build\cublas64_12.dll" >nul
if errorlevel 1 exit /b %errorlevel%
copy /y "%~dp0..\build-ninja\cublasLt64_12.dll" "%~dp0build\cublasLt64_12.dll" >nul
if errorlevel 1 exit /b %errorlevel%
exit /b %errorlevel%

