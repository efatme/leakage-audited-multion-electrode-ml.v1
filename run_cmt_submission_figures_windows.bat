@echo off
setlocal EnableExtensions

REM Run this file from the ROOT of the CMT repository.
REM It is intentionally headless for Windows Server / Remote Desktop use.

cd /d "%~dp0"
set "MPLBACKEND=Agg"
set "PYTHONUNBUFFERED=1"
set "ENVNAME=cmt-insertion-electrode-workflow"
set "FIGSCRIPT=papers\cmt\publication\make_cmt_submission_figures.py"

if not exist "%FIGSCRIPT%" (
  echo [ERROR] Figure script not found:
  echo         %CD%\%FIGSCRIPT%
  echo Copy make_cmt_submission_figures.py to papers\cmt\publication\ and try again.
  exit /b 2
)

where conda >nul 2>&1
if errorlevel 1 (
  echo [ERROR] conda is not available in this shell.
  echo Open "Anaconda Prompt" on the Windows server and run this BAT file there.
  exit /b 3
)

if /I not "%CONDA_DEFAULT_ENV%"=="%ENVNAME%" (
  echo Activating Conda environment: %ENVNAME%
  call conda activate "%ENVNAME%"
  if errorlevel 1 (
    echo [ERROR] Could not activate %ENVNAME%.
    echo If the environment does not exist, run:
    echo   conda env create -f environment.yml
    exit /b 4
  )
)

echo.
echo === Python / plotting environment ===
python -c "import sys,matplotlib,numpy,pandas; print('Python:',sys.version); print('Executable:',sys.executable); print('Matplotlib:',matplotlib.__version__, 'backend=', matplotlib.get_backend()); print('NumPy:',numpy.__version__); print('Pandas:',pandas.__version__)"
if errorlevel 1 exit /b 5

echo.
echo === Input preflight ===
python -u "%FIGSCRIPT%" --repo-root "%CD%" --check-only
if errorlevel 1 (
  echo [ERROR] Input preflight failed. Read the MISSING path printed above.
  exit /b 6
)

echo.
echo === Generating submission artwork ===
python -u "%FIGSCRIPT%" --repo-root "%CD%"
if errorlevel 1 (
  echo [ERROR] Figure generation failed.
  exit /b 7
)

echo.
echo === Finished ===
echo Output directory:
echo   %CD%\papers\cmt\publication\submission_artwork
explorer "%CD%\papers\cmt\publication\submission_artwork"
exit /b 0
