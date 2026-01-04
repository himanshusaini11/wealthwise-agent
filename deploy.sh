#!/bin/bash

# 1. Update Code
echo "Pulling the updated code."
git pull origin main

# 2. Stop Containers
echo "Stopping containers."
docker-compose down

# 3. Clean Up (Aggressive)
echo "Pruning unused images."
docker system prune -a -f

# 4. Check Disk Space
# Get available space in KB. 3GB approx 3,145,728 KB
REQUIRED_KB=3145728
AVAILABLE_KB=$(df / --output=avail | tail -1)

echo "Checking disk space."
if [ "$AVAILABLE_KB" -lt "$REQUIRED_KB" ]; then
    echo "CRITICAL: Not enough disk space! Available: $AVAILABLE_KB KB. Required: $REQUIRED_KB KB."
    # We exit with error code 1. This tells GitHub "JOB FAILED"
    exit 1
else
    echo "Disk space is sufficient. Proceeding to build."
fi

# 5. Deploy
echo "Building and Starting..."
docker-compose up --build -d