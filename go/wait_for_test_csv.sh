#!/bin/sh

FILE="/data/test.csv"
TIMEOUT=30  # seconds
WAIT_INTERVAL=1
ELAPSED=0

echo "Waiting for $FILE to be created..."

while [ ! -f "$FILE" ]; do
    sleep $WAIT_INTERVAL
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "Timeout waiting for $FILE"
        exit 1
    fi
done

echo "$FILE found. Starting Go application..."
exec ./main  # replace with your actual Go binary name
