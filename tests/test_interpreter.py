"""
Unit tests for the interpreter module.

Uses deterministic RNG values for predictable testing.
"""

import pytest
from decimal import Decimal
from typing import Iterator

from pydantic import ValidationError

from app.schema import (
    ExecutionPlan,
    RandOp,
    MulOp,
    DivOp,
    SqrtOp,
    IfOp,
    ReturnOp,
    Condition,
)
from app.interpreter import (
    Interpreter,
    ExecutionError,
    DivisionByZeroError,
    NegativeSqrtError,
    UndefinedVariableError,
)


def make_rng(values: list[float]):
    """Create a deterministic RNG that returns values in sequence."""
    iterator = iter(values)
    def rng():
        return next(iterator)
    return rng


class TestBasicOperations:
    """Test basic mathematical operations."""
    
    def test_rand_operation(self):
        """Test random number generation."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            ReturnOp(op="return", value="x")
        ])
        
        interpreter = Interpreter(rng=make_rng([0.5]))
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 0.5
        assert randoms == [0.5]
    
    def test_mul_operation(self):
        """Test multiplication."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            MulOp(op="mul", a="x", b=2.0, var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        interpreter = Interpreter(rng=make_rng([0.25]))
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 0.5
        assert randoms == [0.25]
    
    def test_mul_two_variables(self):
        """Test multiplication of two variables."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="a"),
            RandOp(op="rand", var="b"),
            MulOp(op="mul", a="a", b="b", var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        interpreter = Interpreter(rng=make_rng([0.5, 0.4]))
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 0.2
        assert randoms == [0.5, 0.4]
    
    def test_div_operation(self):
        """Test division."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            DivOp(op="div", a="x", b=2.0, var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        interpreter = Interpreter(rng=make_rng([0.8]))
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 0.4
        assert randoms == [0.8]
    
    def test_sqrt_operation(self):
        """Test square root."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            SqrtOp(op="sqrt", x="x", var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        interpreter = Interpreter(rng=make_rng([0.25]))
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 0.5
        assert randoms == [0.25]
    
    def test_sqrt_zero(self):
        """Test square root of zero."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            SqrtOp(op="sqrt", x="x", var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        interpreter = Interpreter(rng=make_rng([0.0]))
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 0.0
        assert randoms == [0.0]
    
    def test_literal_values(self):
        """Test using literal values in operations."""
        plan = ExecutionPlan(steps=[
            MulOp(op="mul", a=3.0, b=4.0, var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        interpreter = Interpreter()
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 12.0
        assert randoms == []
    
    def test_return_literal(self):
        """Test returning a literal value."""
        plan = ExecutionPlan(steps=[
            ReturnOp(op="return", value=42.5)
        ])
        
        interpreter = Interpreter()
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 42.5
        assert randoms == []


class TestConditionalLogic:
    """Test conditional (if/else) operations."""
    
    def test_if_then_branch(self):
        """Test if statement taking then branch."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            IfOp(
                op="if",
                condition=Condition(left="x", operator="<", right=0.5),
                then_steps=[
                    MulOp(op="mul", a="x", b=2.0, var="result"),
                    ReturnOp(op="return", value="result")
                ],
                else_steps=[
                    DivOp(op="div", a="x", b=2.0, var="result"),
                    ReturnOp(op="return", value="result")
                ]
            )
        ])
        
        # x = 0.3 < 0.5, so then branch: 0.3 * 2 = 0.6
        interpreter = Interpreter(rng=make_rng([0.3]))
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 0.6
        assert randoms == [0.3]
    
    def test_if_else_branch(self):
        """Test if statement taking else branch."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            IfOp(
                op="if",
                condition=Condition(left="x", operator="<", right=0.5),
                then_steps=[
                    MulOp(op="mul", a="x", b=2.0, var="result"),
                    ReturnOp(op="return", value="result")
                ],
                else_steps=[
                    DivOp(op="div", a="x", b=2.0, var="result"),
                    ReturnOp(op="return", value="result")
                ]
            )
        ])
        
        # x = 0.8 >= 0.5, so else branch: 0.8 / 2 = 0.4
        interpreter = Interpreter(rng=make_rng([0.8]))
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 0.4
        assert randoms == [0.8]
    
    def test_if_empty_else(self):
        """Test if statement with no else branch."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            IfOp(
                op="if",
                condition=Condition(left="x", operator=">", right=0.5),
                then_steps=[
                    MulOp(op="mul", a="x", b=10.0, var="x")
                ],
                else_steps=[]
            ),
            ReturnOp(op="return", value="x")
        ])
        
        # x = 0.7 > 0.5, so then branch: x = 0.7 * 10 = 7
        interpreter = Interpreter(rng=make_rng([0.7]))
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 7.0
        assert randoms == [0.7]
    
    def test_nested_conditionals(self):
        """Test nested if statements."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            IfOp(
                op="if",
                condition=Condition(left="x", operator="<", right=0.5),
                then_steps=[
                    IfOp(
                        op="if",
                        condition=Condition(left="x", operator="<", right=0.25),
                        then_steps=[ReturnOp(op="return", value=1.0)],
                        else_steps=[ReturnOp(op="return", value=2.0)]
                    )
                ],
                else_steps=[ReturnOp(op="return", value=3.0)]
            )
        ])
        
        # x = 0.1 < 0.5 and < 0.25, so return 1
        interpreter = Interpreter(rng=make_rng([0.1]))
        result, _, _ = interpreter.execute(plan)
        assert result == 1.0
        
        # x = 0.3 < 0.5 but >= 0.25, so return 2
        interpreter = Interpreter(rng=make_rng([0.3]))
        result, _, _ = interpreter.execute(plan)
        assert result == 2.0
        
        # x = 0.6 >= 0.5, so return 3
        interpreter = Interpreter(rng=make_rng([0.6]))
        result, _, _ = interpreter.execute(plan)
        assert result == 3.0
    
    def test_all_comparison_operators(self):
        """Test all comparison operators."""
        def test_operator(op: str, left: float, right: float, expected: bool):
            plan = ExecutionPlan(steps=[
                IfOp(
                    op="if",
                    condition=Condition(left=left, operator=op, right=right),
                    then_steps=[ReturnOp(op="return", value=1.0)],
                    else_steps=[ReturnOp(op="return", value=0.0)]
                )
            ])
            interpreter = Interpreter()
            result, _, _ = interpreter.execute(plan)
            assert result == (1.0 if expected else 0.0), f"{left} {op} {right} should be {expected}"
        
        # Test <
        test_operator("<", 0.3, 0.5, True)
        test_operator("<", 0.5, 0.5, False)
        test_operator("<", 0.7, 0.5, False)
        
        # Test >
        test_operator(">", 0.7, 0.5, True)
        test_operator(">", 0.5, 0.5, False)
        test_operator(">", 0.3, 0.5, False)
        
        # Test <=
        test_operator("<=", 0.3, 0.5, True)
        test_operator("<=", 0.5, 0.5, True)
        test_operator("<=", 0.7, 0.5, False)
        
        # Test >=
        test_operator(">=", 0.7, 0.5, True)
        test_operator(">=", 0.5, 0.5, True)
        test_operator(">=", 0.3, 0.5, False)
        
        # Test ==
        test_operator("==", 0.5, 0.5, True)
        test_operator("==", 0.3, 0.5, False)
        
        # Test !=
        test_operator("!=", 0.3, 0.5, True)
        test_operator("!=", 0.5, 0.5, False)


class TestMultiStepExecution:
    """Test multi-step execution plans."""
    
    def test_chain_of_operations(self):
        """Test a chain of operations."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="a"),
            RandOp(op="rand", var="b"),
            MulOp(op="mul", a="a", b="b", var="c"),
            SqrtOp(op="sqrt", x="c", var="d"),
            DivOp(op="div", a="d", b=2.0, var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        # a=0.64, b=0.25, c=0.16, d=0.4, result=0.2
        interpreter = Interpreter(rng=make_rng([0.64, 0.25]))
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 0.2
        assert randoms == [0.64, 0.25]
    
    def test_example_from_spec(self):
        """Test the example from the specification."""
        # "Generate a random number. If the number is less than 0.5, 
        # multiply it by 0.1234567, otherwise divide it by 1.1234567. 
        # Generate another random number, get the square root of it, 
        # and then multiply it by the previous result."
        
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="r1"),
            IfOp(
                op="if",
                condition=Condition(left="r1", operator="<", right=0.5),
                then_steps=[
                    MulOp(op="mul", a="r1", b=0.1234567, var="intermediate")
                ],
                else_steps=[
                    DivOp(op="div", a="r1", b=1.1234567, var="intermediate")
                ]
            ),
            RandOp(op="rand", var="r2"),
            SqrtOp(op="sqrt", x="r2", var="sqrt_r2"),
            MulOp(op="mul", a="sqrt_r2", b="intermediate", var="final"),
            ReturnOp(op="return", value="final")
        ])
        
        # r1 = 0.61234 >= 0.5, so divide: 0.61234 / 1.1234567 = 0.5450648...
        # r2 = 0.232343, sqrt = 0.4820197...
        # result = 0.4820197... * 0.5450648... = 0.2627248...
        interpreter = Interpreter(rng=make_rng([0.61234, 0.232343]))
        result, randoms, _ = interpreter.execute(plan)
        
        assert randoms == [0.61234, 0.232343]
        # Verify it's a reasonable value (approximate check)
        assert 0.26 < result < 0.27
    
    def test_variable_reuse(self):
        """Test reusing the same variable name."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            MulOp(op="mul", a="x", b=2.0, var="x"),
            MulOp(op="mul", a="x", b=2.0, var="x"),
            ReturnOp(op="return", value="x")
        ])
        
        # x = 0.1 -> 0.2 -> 0.4
        interpreter = Interpreter(rng=make_rng([0.1]))
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 0.4
        assert randoms == [0.1]


class TestPrecision:
    """Test decimal precision handling."""
    
    def test_ten_decimal_places(self):
        """Test that results have exactly 10 decimal places."""
        plan = ExecutionPlan(steps=[
            DivOp(op="div", a=1.0, b=3.0, var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        interpreter = Interpreter()
        result, _, _ = interpreter.execute(plan)
        
        # 1/3 = 0.3333333333...
        assert result == 0.3333333333
    
    def test_precision_in_chain(self):
        """Test precision is maintained through operations."""
        plan = ExecutionPlan(steps=[
            DivOp(op="div", a=1.0, b=7.0, var="x"),
            MulOp(op="mul", a="x", b=7.0, var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        interpreter = Interpreter()
        result, _, _ = interpreter.execute(plan)
        
        # Due to rounding, 1/7 * 7 should be very close to 1
        assert abs(result - 1.0) < 1e-9
    
    def test_small_numbers(self):
        """Test handling of small numbers."""
        plan = ExecutionPlan(steps=[
            MulOp(op="mul", a=0.0000000001, b=0.0000000001, var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        interpreter = Interpreter()
        result, _, _ = interpreter.execute(plan)
        
        # 1e-10 * 1e-10 = 1e-20, rounded to 10 decimal places = 0
        assert result == 0.0


class TestErrorHandling:
    """Test error handling."""
    
    def test_division_by_zero(self):
        """Test division by zero raises error."""
        plan = ExecutionPlan(steps=[
            DivOp(op="div", a=1.0, b=0.0, var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        interpreter = Interpreter()
        
        with pytest.raises(DivisionByZeroError):
            interpreter.execute(plan)
    
    def test_division_by_zero_variable(self):
        """Test division by zero with variable."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            DivOp(op="div", a=1.0, b="x", var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        interpreter = Interpreter(rng=make_rng([0.0]))
        
        with pytest.raises(DivisionByZeroError):
            interpreter.execute(plan)
    
    def test_sqrt_negative(self):
        """Test square root of negative number raises error."""
        plan = ExecutionPlan(steps=[
            SqrtOp(op="sqrt", x=-1.0, var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        interpreter = Interpreter()
        
        with pytest.raises(NegativeSqrtError):
            interpreter.execute(plan)
    
    def test_undefined_variable(self):
        """Test referencing undefined variable raises error."""
        plan = ExecutionPlan(steps=[
            MulOp(op="mul", a="undefined", b=2.0, var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        interpreter = Interpreter()
        
        with pytest.raises(UndefinedVariableError):
            interpreter.execute(plan)
    
    def test_no_return(self):
        """Test plan without return raises error."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x")
        ])
        
        interpreter = Interpreter()
        
        with pytest.raises(ExecutionError, match="did not return a result"):
            interpreter.execute(plan)


class TestSchemaValidation:
    """Test schema validation."""
    
    def test_max_steps_validation(self):
        """Test that too many steps is rejected."""
        steps = [RandOp(op="rand", var=f"x{i}") for i in range(201)]
        steps.append(ReturnOp(op="return", value="x0"))
        
        with pytest.raises((ValueError, ValidationError)):
            ExecutionPlan(steps=steps)
    
    def test_max_nesting_depth(self):
        """Test that excessive nesting is rejected."""
        # Create 6 levels of nesting
        inner = [ReturnOp(op="return", value=1.0)]
        for _ in range(6):
            inner = [IfOp(
                op="if",
                condition=Condition(left=0.5, operator="<", right=1.0),
                then_steps=inner,
                else_steps=[]
            )]
        
        with pytest.raises(ValueError, match="Nesting depth exceeds maximum of 5"):
            ExecutionPlan(steps=inner)
    
    def test_invalid_variable_name(self):
        """Test that invalid variable names are rejected."""
        with pytest.raises(ValueError):
            RandOp(op="rand", var="123invalid")
        
        with pytest.raises(ValueError):
            RandOp(op="rand", var="has space")
    
    def test_empty_steps(self):
        """Test that empty steps list is rejected."""
        with pytest.raises(ValueError):
            ExecutionPlan(steps=[])


class TestEdgeCases:
    """Test edge cases."""
    
    def test_rng_boundary_values(self):
        """Test RNG boundary values are handled."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            ReturnOp(op="return", value="x")
        ])
        
        # Test 0
        interpreter = Interpreter(rng=make_rng([0.0]))
        result, randoms, _ = interpreter.execute(plan)
        assert result == 0.0
        assert randoms == [0.0]
        
        # Test 1
        interpreter = Interpreter(rng=make_rng([1.0]))
        result, randoms, _ = interpreter.execute(plan)
        assert result == 1.0
        assert randoms == [1.0]
    
    def test_return_stops_execution(self):
        """Test that return stops further execution."""
        call_count = [0]
        def counting_rng():
            call_count[0] += 1
            return 0.5
        
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            ReturnOp(op="return", value="x"),
            RandOp(op="rand", var="y"),  # Should not execute
        ])
        
        interpreter = Interpreter(rng=counting_rng)
        result, randoms, _ = interpreter.execute(plan)
        
        assert result == 0.5
        assert call_count[0] == 1  # Only one random call
        assert len(randoms) == 1
    
    def test_multiple_executions_independent(self):
        """Test that multiple executions don't share state."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="x"),
            ReturnOp(op="return", value="x")
        ])
        
        values = iter([0.1, 0.2, 0.3])
        interpreter = Interpreter(rng=lambda: next(values))
        
        result1, _, _ = interpreter.execute(plan)
        result2, _, _ = interpreter.execute(plan)
        result3, _, _ = interpreter.execute(plan)
        
        assert result1 == 0.1
        assert result2 == 0.2
        assert result3 == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

