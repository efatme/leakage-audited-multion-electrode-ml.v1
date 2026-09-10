CMT submission figure generation on Windows Server / Anaconda
==============================================================

1. Put make_cmt_submission_figures.py at:
   papers\cmt\publication\make_cmt_submission_figures.py

2. Put run_cmt_submission_figures_windows.bat in the repository root,
   next to environment.yml.

3. Open Anaconda Prompt. Do NOT run from a plain PowerShell/CMD window unless
   Conda is already initialized there.

4. From the repository root, if the environment does not already exist:
   conda env create -f environment.yml

5. Double-click the BAT file from Explorer OR run in Anaconda Prompt:
   run_cmt_submission_figures_windows.bat

The runner forces Matplotlib's Agg backend so no graphical display is needed.
It first performs a --check-only preflight and prints every required input path.
Only if that check passes does it generate the six figures.

Expected output:
   papers\cmt\publication\submission_artwork\

Expected formats for each figure:
   .pdf, .svg, .tif (600 dpi), and _preview.png (300 dpi)

If TIFF-LZW compression is unsupported by the local Pillow build, the renderer
falls back automatically to uncompressed TIFF rather than failing the run.

If the batch fails, copy the complete console output from the first [ERROR] line
to the end. The diagnostics include the Python executable, package versions,
Matplotlib backend, repository root, and missing input path(s).
