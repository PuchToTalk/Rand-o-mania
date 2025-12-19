# Rand-o-mania

A FastAPI server that executes mathematical operations on random numbers via natural language prompts.

## Overview

Converts natural language instructions into safe mathematical operations on random numbers using OpenAI for parsing and a custom interpreter for execution.

## Project Structure

```
app/
├── main.py           # FastAPI server
├── llm_processor.py  # OpenAI integration
├── interpreter.py    # Safe execution engine
└── schema.py         # Pydantic models
tests/
├── test_interpreter.py  # 31 unit tests
└── test_api_errors.py   # 4 API tests
```

### Key Features
- Natural language to structured execution plans via OpenAI API
- Safe execution with no eval/exec
- Decimal arithmetic with 10 decimal place precision
- Automatic retry on LLM validation failures
- Structured error responses with error codes
- Maximum 200 steps and 5 nesting levels for safety

## Quick Start

### Installation

```bash
# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment (replace with your OpenAI API key)
cat > .env << 'EOF'
OPENAI_API_KEY=sk-proj-your-key-here
PORT=8000
EOF
```

### Run Server

```bash
# Start the server
./run.sh

# Server runs on http://localhost:8000
# API docs available at http://localhost:8000/docs
```

### Run Tests

```bash
# Run all unit tests (no OpenAI API key needed)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Simple test
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Generate a random number and multiply it by 10"}'

# Complex test (from spec)
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Generate a random number. If the number is less than 0.5, multiply it by 0.1234567, otherwise divide it by 1.1234567. Generate another random number, get the square root of it, and then multiply it by the previous result."}'
```

## API Reference

### Endpoints

**POST /execute** - Execute mathematical operations from natural language
- Request: `{"prompt": "string", "seed": int|null, "debug": bool}`
- Response: `{"result": float, "random_integers": [float], "debug": object|null}`

**POST /execute_plan** - Execute a plan directly (bypass LLM)
- Request: `{"plan": object, "seed": int|null, "debug": bool}`
- Response: `{"result": float, "random_integers": [float], "debug": object|null}`

**GET /health** - Server health check
- Response: `{"status": "healthy", "llm_ready": bool}`

## Supported Operations

`rand`, `mul(a, b)`, `div(a, b)`, `sqrt(x)`, `if(condition, then, else)`, `return(value)`

## Advanced Features

### Deterministic Execution with Seed

Use `seed` for reproducible results:

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Generate a random number", "seed": 42}'
```

Same seed always produces identical results.

### Debug Mode

Add `debug: true` for detailed execution trace:

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Generate a random number and multiply by 2", "seed": 42, "debug": true}'
```

Returns:
- Full execution plan
- Step-by-step trace with inputs/outputs
- Final variable states
- Execution statistics (steps, random count, depth)
- Limits (max steps, max depth)

### Direct Plan Execution

Bypass LLM and execute structured plans directly:

```bash
curl -X POST http://localhost:8000/execute_plan \
  -H "Content-Type: application/json" \
  -d '{
    "plan": {
      "steps": [
        {"op": "rand", "var": "r1"},
        {"op": "mul", "a": "r1", "b": 10, "var": "result"},
        {"op": "return", "value": "result"}
      ]
    },
    "seed": 42,
    "debug": true
  }'
```

Useful for testing, integrations, or when you already have a structured plan.

## Technical Details

### LLM Integration
- Model: gpt-4o with 30s timeout
- Uses OpenAI function calling for structured output
- Automatic retry on validation failure
- Converts natural language to JSON execution plan

### Execution Plan Operations
- `rand` - Generate random number [0,1]
- `mul(a, b)` - Multiplication
- `div(a, b)` - Division
- `sqrt(x)` - Square root
- `if(condition, then, else)` - Conditional logic
- `return(value)` - Return final result

### Safety & Precision
- No eval/exec/ast - only whitelisted operations
- Decimal arithmetic with 10 decimal place precision
- Maximum 200 steps, 5 nesting levels
- All plans validated with Pydantic schemas
- Execution errors isolated from server crashes

## Error Handling

All errors return structured responses:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message"
  }
}
```

### Status Codes
- 200: Success
- 400: Invalid prompt or LLM failure
- 422: Execution error (division by zero, sqrt negative, etc.)
- 500: Internal server error

### Error Codes
`LLM_NOT_INITIALIZED`, `LLM_PROCESSING_ERROR`, `DIVISION_BY_ZERO`, `NEGATIVE_SQRT`, `UNDEFINED_VARIABLE`, `INVALID_OPERATION`, `EXECUTION_ERROR`, `INTERNAL_ERROR`

## Testing

35 unit tests covering all operations. Tests do not require OpenAI API key.

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app
```

Tests include: basic operations, conditionals, multi-step execution, precision, error handling, schema validation, and edge cases. All use mock RNG for deterministic results.


## Public Exposure

```bash
# Using ngrok
ngrok http 8000
```

## License

MIT

