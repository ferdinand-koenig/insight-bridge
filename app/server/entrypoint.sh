#!/bin/bash
set -e  # stop on errors

# Install requirements
cd /insight-bridge
pip3 install --no-cache-dir -r requirements.txt

# Finally, start the main app
exec "$@"
