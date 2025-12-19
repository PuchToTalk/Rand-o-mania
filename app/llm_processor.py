"""
OpenAI integration for converting natural language to execution plans.

Uses function/tool calling for structured output.
"""

import json
import os
from typing import Any, Optional

from openai import OpenAI
from pydantic import ValidationError

from app.schema import ExecutionPlan


class LLMProcessingError(Exception):
    """Raised when LLM processing fails."""
    pass


EXECUTION_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "description": "List of operations to execute in order",
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["rand"]},
                            "var": {"type": "string", "description": "Variable name to store the random number"}
                        },
                        "required": ["op", "var"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["mul"]},
                            "a": {"oneOf": [{"type": "string"}, {"type": "number"}], "description": "First operand (variable name or number)"},
                            "b": {"oneOf": [{"type": "string"}, {"type": "number"}], "description": "Second operand (variable name or number)"},
                            "var": {"type": "string", "description": "Variable name to store the result"}
                        },
                        "required": ["op", "a", "b", "var"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["div"]},
                            "a": {"oneOf": [{"type": "string"}, {"type": "number"}], "description": "Dividend (variable name or number)"},
                            "b": {"oneOf": [{"type": "string"}, {"type": "number"}], "description": "Divisor (variable name or number)"},
                            "var": {"type": "string", "description": "Variable name to store the result"}
                        },
                        "required": ["op", "a", "b", "var"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["sqrt"]},
                            "x": {"oneOf": [{"type": "string"}, {"type": "number"}], "description": "Value to take square root of (variable name or number)"},
                            "var": {"type": "string", "description": "Variable name to store the result"}
                        },
                        "required": ["op", "x", "var"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["if"]},
                            "condition": {
                                "type": "object",
                                "properties": {
                                    "left": {"oneOf": [{"type": "string"}, {"type": "number"}], "description": "Left side of comparison"},
                                    "operator": {"type": "string", "enum": ["<", ">", "<=", ">=", "==", "!="]},
                                    "right": {"oneOf": [{"type": "string"}, {"type": "number"}], "description": "Right side of comparison"}
                                },
                                "required": ["left", "operator", "right"],
                                "additionalProperties": False
                            },
                            "then_steps": {
                                "type": "array",
                                "description": "Steps to execute if condition is true",
                                "items": {"type": "object"}
                            },
                            "else_steps": {
                                "type": "array",
                                "description": "Steps to execute if condition is false",
                                "items": {"type": "object"}
                            }
                        },
                        "required": ["op", "condition", "then_steps"],
                        "additionalProperties": False
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["return"]},
                            "value": {"oneOf": [{"type": "string"}, {"type": "number"}], "description": "Value to return (variable name or number)"}
                        },
                        "required": ["op", "value"],
                        "additionalProperties": False
                    }
                ]
            }
        }
    },
    "required": ["steps"],
    "additionalProperties": False
}


SYSTEM_PROMPT = """You are a mathematical operation planner. Your job is to convert natural language instructions about random numbers and mathematical operations into a structured execution plan.

IMPORTANT: Return ONLY valid JSON matching the schema. No markdown formatting, no code blocks, no extra text.

Available operations:
1. rand: Generate a random number between 0 and 1, store in a variable
2. mul: Multiply two values (variables or literals), store result in a variable
3. div: Divide first value by second, store result in a variable
4. sqrt: Take square root, store result in a variable
5. if: Conditional execution based on comparison
6. return: Return the final result

Rules:
- Every plan MUST end with a return operation
- Use descriptive variable names (e.g., "random1", "result", "product")
- For conditionals, condition has: left, operator (<, >, <=, >=, ==, !=), right
- then_steps are executed if condition is true, else_steps if false (can be empty array)
- Reference previous results by their variable names
- Numbers in prompts should be used as literals (e.g., 0.1234567)
- All variable names must be valid identifiers (letters, numbers, underscores only)

Example: "Generate a random number and multiply it by 2"
{
  "steps": [
    {"op": "rand", "var": "r1"},
    {"op": "mul", "a": "r1", "b": 2, "var": "result"},
    {"op": "return", "value": "result"}
  ]
}

Example: "Generate a random number. If it's less than 0.5, return it, otherwise return 0.5"
{
  "steps": [
    {"op": "rand", "var": "r1"},
    {"op": "if", "condition": {"left": "r1", "operator": "<", "right": 0.5}, 
     "then_steps": [{"op": "return", "value": "r1"}],
     "else_steps": [{"op": "return", "value": 0.5}]}
  ]
}

Convert the user's prompt into an execution plan."""


class LLMProcessor:
    """Processes natural language prompts into execution plans using OpenAI."""
    
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_RETRIES = 1  # retry once on validation failure
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gpt-4o",
        timeout: int = DEFAULT_TIMEOUT
    ):
        """
        Initialize LLM processor.
        
        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            model: Model to use. Defaults to gpt-4o.
            timeout: Request timeout in seconds. Defaults to 30.
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
        
        self._client = OpenAI(api_key=self._api_key, timeout=timeout)
        self._model = model
        self._timeout = timeout
    
    def process(self, prompt: str) -> ExecutionPlan:
        """
        Convert natural language prompt to validated execution plan.
        
        Uses retry logic: if validation fails, retry once with a repair prompt.
        
        Args:
            prompt: Natural language instructions
            
        Returns:
            Validated ExecutionPlan
            
        Raises:
            LLMProcessingError: If processing fails after retries
        """
        last_error = None
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                if attempt == 0:
                    # First attempt - use original prompt
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                else:
                    # Retry attempt - add repair instruction
                    repair_msg = (
                        f"Your previous response had a validation error: {last_error}\n\n"
                        "Please fix the execution plan. Return ONLY valid JSON matching the schema, "
                        "with no markdown formatting or extra text."
                    )
                    messages.append({"role": "user", "content": repair_msg})
                
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=[{
                        "type": "function",
                        "function": {
                            "name": "create_execution_plan",
                            "description": "Create a structured execution plan from the user's mathematical instructions",
                            "parameters": EXECUTION_PLAN_SCHEMA
                        }
                    }],
                    tool_choice={"type": "function", "function": {"name": "create_execution_plan"}}
                )
                
                # Extract the function call arguments
                message = response.choices[0].message
                if not message.tool_calls:
                    last_error = "LLM did not return a function call"
                    continue
                
                tool_call = message.tool_calls[0]
                if tool_call.function.name != "create_execution_plan":
                    last_error = f"Unexpected function call: {tool_call.function.name}"
                    continue
                
                # Parse the JSON arguments
                try:
                    plan_data = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    last_error = f"Invalid JSON from LLM: {e}"
                    continue
                
                # Validate with Pydantic
                try:
                    plan = ExecutionPlan.model_validate(plan_data)
                    return plan  # Success!
                except ValidationError as e:
                    last_error = f"Invalid execution plan: {e}"
                    # Store messages for potential retry
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                    ]
                    continue
                    
            except LLMProcessingError:
                raise
            except Exception as e:
                raise LLMProcessingError(f"LLM processing failed: {e}")
        
        # All retries exhausted
        raise LLMProcessingError(f"Failed after {self.MAX_RETRIES + 1} attempts. Last error: {last_error}")

