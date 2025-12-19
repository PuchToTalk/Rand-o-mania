"""
Safe execution engine for mathematical operations.

Uses Decimal for precise arithmetic and tracks all random numbers generated.
NEVER uses eval/exec/ast.literal_eval.
"""

import random
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation, getcontext
from typing import Callable, Optional, List, Dict

from app.schema import (
    ExecutionPlan,
    Operation,
    RandOp,
    MulOp,
    DivOp,
    SqrtOp,
    IfOp,
    ReturnOp,
    Condition,
)


class ExecutionError(Exception):
    """Raised when execution fails."""
    pass


class DivisionByZeroError(ExecutionError):
    """Raised when dividing by zero."""
    pass


class NegativeSqrtError(ExecutionError):
    """Raised when taking sqrt of negative number."""
    pass


class InvalidOperationError(ExecutionError):
    """Raised for invalid operations."""
    pass


class UndefinedVariableError(ExecutionError):
    """Raised when referencing undefined variable."""
    pass


class Interpreter:
    """
    Safe interpreter for execution plans.
    
    Features:
    - Uses Decimal for all arithmetic with 10 decimal place precision
    - Tracks all random numbers generated
    - Stores variables in a dict
    - No eval/exec - only whitelisted operations
    """
    
    PRECISION = 10
    MAX_STEPS = 200
    MAX_DEPTH = 5
    
    def __init__(self, rng: Optional[Callable[[], float]] = None, debug: bool = False):
        """
        Initialize interpreter.
        
        Args:
            rng: Optional random number generator function for testing.
                 Should return floats between 0 and 1.
                 Defaults to random.random().
            debug: If True, collect execution trace
        """
        self._rng = rng or random.random
        self._variables: Dict[str, Decimal] = {}
        self._random_numbers: List[float] = []
        self._result: Optional[Decimal] = None
        self._debug = debug
        self._trace: List[Dict] = []
        self._step_counter = 0
        self._current_depth = 0
        self._max_depth_reached = 0
        
        # Set decimal precision high enough for intermediate calculations
        getcontext().prec = 50
    
    def execute(self, plan: ExecutionPlan):
        """
        Execute an execution plan.
        
        Args:
            plan: Validated execution plan
            
        Returns:
            Tuple of (result, random_numbers_generated, debug_info)
            debug_info is None if debug=False
            
        Raises:
            ExecutionError: If execution fails
        """
        self._variables = {}
        self._random_numbers = []
        self._result = None
        self._trace = []
        self._step_counter = 0
        self._current_depth = 0
        self._max_depth_reached = 0
        
        self._execute_steps(plan.steps)
        
        if self._result is None:
            raise ExecutionError("Execution plan did not return a result")
        
        # Round to 10 decimal places
        result_rounded = float(self._round_decimal(self._result))
        
        # Prepare debug info if needed
        debug_info = None
        if self._debug:
            debug_info = {
                "execution_trace": self._trace,
                "final_vars": {k: str(v) for k, v in self._variables.items()},
                "result_str": f"{result_rounded:.10f}",
                "steps_executed": self._step_counter,
                "random_count": len(self._random_numbers),
                "max_depth_reached": self._max_depth_reached
            }
        
        return result_rounded, self._random_numbers, debug_info
    
    def _execute_steps(self, steps: List[Operation]) -> None:
        """Execute a list of steps."""
        for step in steps:
            if self._result is not None:
                # Already returned, skip remaining steps
                break
            self._execute_step(step)
    
    def _execute_step(self, step: Operation) -> None:
        """Execute a single step based on operation type."""
        if isinstance(step, RandOp):
            self._execute_rand(step)
        elif isinstance(step, MulOp):
            self._execute_mul(step)
        elif isinstance(step, DivOp):
            self._execute_div(step)
        elif isinstance(step, SqrtOp):
            self._execute_sqrt(step)
        elif isinstance(step, IfOp):
            self._execute_if(step)
        elif isinstance(step, ReturnOp):
            self._execute_return(step)
        else:
            raise InvalidOperationError(f"Unknown operation type: {type(step)}")
    
    def _execute_rand(self, op: RandOp) -> None:
        """Generate random number and store in variable."""
        self._step_counter += 1
        
        if self._debug:
            inputs = {}
        
        value = self._rng()
        # Ensure value is between 0 and 1
        value = max(0.0, min(1.0, value))
        self._random_numbers.append(value)
        self._variables[op.var] = Decimal(str(value))
        
        if self._debug:
            self._trace.append({
                "step": self._step_counter,
                "op": "rand",
                "inputs": inputs,
                "output": {op.var: str(self._variables[op.var])},
                "random_generated": str(self._variables[op.var])
            })
    
    def _execute_mul(self, op: MulOp) -> None:
        """Multiply two values and store result."""
        self._step_counter += 1
        
        if self._debug:
            inputs = {"a": str(op.a), "b": str(op.b)}
        
        a = self._resolve_value(op.a)
        b = self._resolve_value(op.b)
        self._variables[op.var] = a * b
        
        if self._debug:
            self._trace.append({
                "step": self._step_counter,
                "op": "mul",
                "inputs": inputs,
                "output": {op.var: str(self._variables[op.var])},
                "random_generated": None
            })
    
    def _execute_div(self, op: DivOp) -> None:
        """Divide a by b and store result."""
        self._step_counter += 1
        
        if self._debug:
            inputs = {"a": str(op.a), "b": str(op.b)}
        
        a = self._resolve_value(op.a)
        b = self._resolve_value(op.b)
        
        if b == Decimal("0"):
            raise DivisionByZeroError("Cannot divide by zero")
        
        self._variables[op.var] = a / b
        
        if self._debug:
            self._trace.append({
                "step": self._step_counter,
                "op": "div",
                "inputs": inputs,
                "output": {op.var: str(self._variables[op.var])},
                "random_generated": None
            })
    
    def _execute_sqrt(self, op: SqrtOp) -> None:
        """Calculate square root and store result."""
        self._step_counter += 1
        
        if self._debug:
            inputs = {"x": str(op.x)}
        
        x = self._resolve_value(op.x)
        
        if x < Decimal("0"):
            raise NegativeSqrtError(f"Cannot take square root of negative number: {x}")
        
        if x == Decimal("0"):
            self._variables[op.var] = Decimal("0")
        else:
            # Use Newton-Raphson method for precise square root
            self._variables[op.var] = self._decimal_sqrt(x)
        
        if self._debug:
            self._trace.append({
                "step": self._step_counter,
                "op": "sqrt",
                "inputs": inputs,
                "output": {op.var: str(self._variables[op.var])},
                "random_generated": None
            })
    
    def _execute_if(self, op: IfOp) -> None:
        """Execute conditional logic."""
        self._step_counter += 1
        self._current_depth += 1
        self._max_depth_reached = max(self._max_depth_reached, self._current_depth)
        
        if self._debug:
            inputs = {
                "left": str(op.condition.left),
                "operator": op.condition.operator,
                "right": str(op.condition.right)
            }
        
        condition_result = self._evaluate_condition(op.condition)
        
        if self._debug:
            self._trace.append({
                "step": self._step_counter,
                "op": "if",
                "inputs": inputs,
                "output": {"condition_result": str(condition_result)},
                "random_generated": None
            })
        
        if condition_result:
            self._execute_steps(op.then_steps)
        else:
            self._execute_steps(op.else_steps)
        
        self._current_depth -= 1
    
    def _execute_return(self, op: ReturnOp) -> None:
        """Set the final result."""
        self._step_counter += 1
        
        if self._debug:
            inputs = {"value": str(op.value)}
        
        self._result = self._resolve_value(op.value)
        
        if self._debug:
            self._trace.append({
                "step": self._step_counter,
                "op": "return",
                "inputs": inputs,
                "output": {"result": str(self._result)},
                "random_generated": None
            })
    
    def _resolve_value(self, value) -> Decimal:
        """
        Resolve a value to a Decimal.
        
        If string, look up variable. If number, convert to Decimal.
        """
        if isinstance(value, str):
            if value not in self._variables:
                raise UndefinedVariableError(f"Undefined variable: {value}")
            return self._variables[value]
        else:
            return Decimal(str(value))
    
    def _evaluate_condition(self, condition: Condition) -> bool:
        """Evaluate a condition and return boolean result."""
        left = self._resolve_value(condition.left)
        right = self._resolve_value(condition.right)
        
        operators = {
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        
        op_func = operators.get(condition.operator)
        if op_func is None:
            raise InvalidOperationError(f"Unknown operator: {condition.operator}")
        
        return op_func(left, right)
    
    def _decimal_sqrt(self, n: Decimal) -> Decimal:
        """
        Calculate square root using Newton-Raphson method.
        
        Provides high precision without using float conversion.
        """
        if n == Decimal("0"):
            return Decimal("0")
        
        # Initial guess
        x = n
        
        # Newton-Raphson iterations
        for _ in range(100):
            x_new = (x + n / x) / Decimal("2")
            if abs(x_new - x) < Decimal("1E-50"):
                break
            x = x_new
        
        return x_new
    
    def _round_decimal(self, value: Decimal) -> Decimal:
        """Round to PRECISION decimal places."""
        quantize_str = "1." + "0" * self.PRECISION
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

