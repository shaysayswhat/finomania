#!/usr/bin/env sh
# Start Finomania on macOS / Linux:  ./finomania.sh  [file]   (after: python3 -m pip install -r requirements.txt)
cd "$(dirname "$0")" && exec python3 -m finomania "$@"
