#!/bin/bash

# Rand-o-mania Server Startup Script

set -e

# Load environment variables from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check for required environment variable
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    echo "Please set it in .env file or export it directly"
    exit 1
fi

# Default port
PORT=${PORT:-8000}

echo "Starting Rand-o-mania server on port $PORT..."
echo "API documentation available at http://localhost:$PORT/docs"
echo ""

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload

