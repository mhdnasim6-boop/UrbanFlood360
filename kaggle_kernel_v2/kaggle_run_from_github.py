#!/usr/bin/env python3
"""
UrbanFlood360 - Kaggle entry point: clone the code from GitHub and run the UrbanSARFloods training.

The training code lives in the GitHub repo (kaggle_kernel_v2/kaggle_train_urbansarfloods_v2.py), so a new run
always uses the latest pushed version. Outputs go to /kaggle/working/output as before.

Set GIT_REF to a branch, tag or commit to pin the exact code version used for a run.
"""
import os
import runpy
import subprocess
import sys

REPO = "https://github.com/mhdnasim6-boop/UrbanFlood360.git"
GIT_REF = "main"
DEST = "/kaggle/temp/UrbanFlood360"   # not /kaggle/working, so the code is not downloaded with the outputs

if not os.path.exists(DEST):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", GIT_REF, REPO, DEST], check=True)
commit = subprocess.run(["git", "-C", DEST, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"UrbanFlood360 code: {REPO} @ {GIT_REF} ({commit})", flush=True)
os.makedirs("/kaggle/working/output", exist_ok=True)
with open("/kaggle/working/output/code_version.txt", "w") as f:
    f.write(f"{REPO}\n{GIT_REF}\n{commit}\n")

sys.argv = ["kaggle_train_urbansarfloods_v2.py"]
runpy.run_path(os.path.join(DEST, "kaggle_kernel_v2/kaggle_train_urbansarfloods_v2.py"), run_name="__main__")
