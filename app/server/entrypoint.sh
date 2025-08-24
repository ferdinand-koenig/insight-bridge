#!/bin/bash
set -e  # stop on errors

# Set pip cache
export PIP_CACHE_DIR=/root/.cache/pip

# Install requirements
cd /insight-bridge
pip3 install -r requirements.txt

# Finally, start the main app
exec "$@"
