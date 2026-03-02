# To be added to ".venv\Lib\site-packages"

import os, sys
from pathlib import Path

venv = Path(sys.prefix)
torch_lib = venv / "Lib" / "site-packages" / "torch" / "lib"
if torch_lib.exists():
    os.add_dll_directory(str(torch_lib))

# also add ffmpeg bin if you want:
ffmpeg_bin = Path(r"C:\ffmpeg-n7.1-latest-win64-gpl-shared-7.1\bin")
if ffmpeg_bin.exists():
    os.add_dll_directory(str(ffmpeg_bin))
    