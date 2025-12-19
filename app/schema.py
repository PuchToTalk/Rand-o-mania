"""
Pydantic models for the execution plan schema.

Defines all supported operations and their validation rules.
"""

from typing import Literal, Union, Annotated, Optional, List, Dict
from pydantic import BaseModel, Field, field_validator, model_validator


class RandOp(BaseModel):
    """Generate a random number between 0 and 1, store in variable."""
    op: Literal["rand"]
    var: str = Field(..., pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$", max_length=50)


class MulOp(BaseModel):
    """Multiply two values (variables or literals)."""
    op: Literal["mul"]
    a: Union[str, float] = Field(...)
    b: Union[str, float] = Field(...)
    var: str = Field(..., pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$", max_length=50)


class DivOp(BaseModel):
    """Divide a by b (variables or literals)."""
    op: Literal["div"]
    a: Union[str, float] = Field(...)
    b: Union[str, float] = Field(...)
    var: str = Field(..., pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$", max_length=50)


class SqrtOp(BaseModel):
    """Square root of a value (variable or literal)."""
    op: Literal["sqrt"]
    x: Union[str, float] = Field(...)
    var: str = Field(..., pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$", max_length=50)


class Condition(BaseModel):
    """Condition for if statements."""
    left: Union[str, float] = Field(...)
    operator: Literal["<", ">", "<=", ">=", "==", "!="]
    right: Union[str, float] = Field(...)


class ReturnOp(BaseModel):
    """Return the final result."""
    op: Literal["return"]
    value: Union[str, float] = Field(...)


# Forward reference for recursive IfOp
class IfOp(BaseModel):
    """Conditional execution with then/else branches."""
    op: Literal["if"]
    condition: Condition
    then_steps: List["Operation"] = Field(..., max_length=50)
    else_steps: List["Operation"] = Field(default_factory=list, max_length=50)


# Union of all operation types
Operation = Annotated[
    Union[RandOp, MulOp, DivOp, SqrtOp, IfOp, ReturnOp],
    Field(discriminator="op")
]

# Update forward references
IfOp.model_rebuild()


class ExecutionPlan(BaseModel):
    """
    Complete execution plan validated from LLM output.
    
    Constraints:
    - Maximum 200 steps total
    - Maximum 5 nesting levels for conditionals
    """
    steps: List[Operation] = Field(..., min_length=1, max_length=200)
    
    @model_validator(mode="after")
    def validate_plan(self) -> "ExecutionPlan":
        """Validate the entire execution plan."""
        total_steps = self._count_steps(self.steps)
        if total_steps > 200:
            raise ValueError(f"Execution plan exceeds maximum of 200 steps (has {total_steps})")
        
        max_depth = self._max_nesting_depth(self.steps)
        if max_depth > 5:
            raise ValueError(f"Nesting depth exceeds maximum of 5 (has {max_depth})")
        
        return self
    
    def _count_steps(self, steps: List[Operation], count: int = 0) -> int:
        """Recursively count all steps including nested ones."""
        for step in steps:
            count += 1
            if isinstance(step, IfOp):
                count = self._count_steps(step.then_steps, count)
                count = self._count_steps(step.else_steps, count)
        return count
    
    def _max_nesting_depth(self, steps: List[Operation], current_depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = current_depth
        for step in steps:
            if isinstance(step, IfOp):
                then_depth = self._max_nesting_depth(step.then_steps, current_depth + 1)
                else_depth = self._max_nesting_depth(step.else_steps, current_depth + 1)
                max_depth = max(max_depth, then_depth, else_depth)
        return max_depth


class ExecuteRequest(BaseModel):
    """Request body for /execute endpoint."""
    prompt: str = Field(..., min_length=1, max_length=5000)
    seed: Optional[int] = None
    debug: bool = False


class ExecutionStep(BaseModel):
    """Single execution step for debug trace."""
    step: int
    op: str
    inputs: Dict[str, str]
    output: Dict[str, str]
    random_generated: Optional[str] = None


class DebugStats(BaseModel):
    """Execution statistics."""
    steps_executed: int
    random_count: int
    max_depth_reached: int


class DebugLimits(BaseModel):
    """Execution limits."""
    max_steps: int
    max_depth: int


class DebugInfo(BaseModel):
    """Debug information for execution."""
    plan: Dict
    execution_trace: List[ExecutionStep]
    final_vars: Dict[str, str]
    result_str: str
    seed_used: Optional[int]
    stats: DebugStats
    limits: DebugLimits


class ExecuteResponse(BaseModel):
    """Response body for /execute endpoint."""
    result: float
    random_integers: List[float]
    debug: Optional[DebugInfo] = None


class ExecutePlanRequest(BaseModel):
    """Request body for /execute_plan endpoint."""
    plan: ExecutionPlan
    seed: Optional[int] = None
    debug: bool = False


class ErrorDetail(BaseModel):
    """Error detail structure."""
    code: str
    message: str


class ErrorResponse(BaseModel):
    """Error response body."""
    error: ErrorDetail

