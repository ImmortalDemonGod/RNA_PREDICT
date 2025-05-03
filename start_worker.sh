#!/bin/bash
# Start a single Cosmic Ray worker on the specified port

if [ -z "$1" ]; then
    echo "Usage: $0 <port>"
    exit 1
fi

PORT=$1
cosmic-ray http-worker --port $PORT 