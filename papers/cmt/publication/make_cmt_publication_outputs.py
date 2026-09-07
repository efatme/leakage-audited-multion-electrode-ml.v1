"""Safe publication-output entrypoint for the frozen CMT release.

The publication figures bundled with this release are reference-rendered artifacts
whose SHA256 hashes are frozen. On systems with a broken Matplotlib font bundle,
this entrypoint verifies those artifacts instead of re-rendering them.

The original Matplotlib rendering source is preserved at:
  papers/cmt/publication/reference_rendering/
  make_cmt_publication_outputs_matplotlib_reference.py
"""
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE / 'verify_cmt_publication_outputs.py'), run_name='__main__')
