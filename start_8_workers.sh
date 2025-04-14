#!/bin/bash
# Start 8 Cosmic Ray workers

# Kill any existing workers
pkill -f "cosmic-ray http-worker"

# Start workers on ports 9876-9883
for port in {9876..9883}
do
    echo "Starting worker on port $port"
    ./start_worker.sh $port &
    # Small delay to avoid overwhelming the system
    sleep 0.5
done

echo "All workers started. Use 'ps -ef | grep cosmic-ray' to verify."
