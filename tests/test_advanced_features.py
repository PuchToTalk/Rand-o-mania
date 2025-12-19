"""
Tests for advanced features: seed support, debug mode, and direct plan execution.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schema import ExecutionPlan, RandOp, MulOp, DivOp, SqrtOp, ReturnOp, Condition, IfOp
from app.interpreter import Interpreter
import random


client = TestClient(app)


class TestSeedSupport:
    """Test deterministic seed support."""
    
    def test_deterministic_seed(self):
        """Same seed produces identical results."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="r1"),
            ReturnOp(op="return", value="r1")
        ])
        
        # Create interpreter with seed
        rng1 = random.Random(42)
        interpreter1 = Interpreter(rng=rng1.random)
        result1, randoms1, _ = interpreter1.execute(plan)
        
        # Same seed should give same results
        rng2 = random.Random(42)
        interpreter2 = Interpreter(rng=rng2.random)
        result2, randoms2, _ = interpreter2.execute(plan)
        
        assert result1 == result2
        assert randoms1 == randoms2
    
    def test_different_seeds_different_results(self):
        """Different seeds produce different results."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="r1"),
            ReturnOp(op="return", value="r1")
        ])
        
        # Different seeds
        rng1 = random.Random(42)
        interpreter1 = Interpreter(rng=rng1.random)
        result1, randoms1, _ = interpreter1.execute(plan)
        
        rng2 = random.Random(123)
        interpreter2 = Interpreter(rng=rng2.random)
        result2, randoms2, _ = interpreter2.execute(plan)
        
        assert result1 != result2
        assert randoms1 != randoms2
    
    def test_api_seed_support(self):
        """Test seed support via API."""
        plan_dict = {
            "steps": [
                {"op": "rand", "var": "r1"},
                {"op": "return", "value": "r1"}
            ]
        }
        
        # First request with seed
        response1 = client.post("/execute_plan", json={
            "plan": plan_dict,
            "seed": 42
        })
        assert response1.status_code == 200
        data1 = response1.json()
        
        # Second request with same seed
        response2 = client.post("/execute_plan", json={
            "plan": plan_dict,
            "seed": 42
        })
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Should be identical
        assert data1["result"] == data2["result"]
        assert data1["random_integers"] == data2["random_integers"]


class TestDebugMode:
    """Test debug mode with execution trace."""
    
    def test_debug_mode_returns_trace(self):
        """Debug mode includes execution trace."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="r1"),
            MulOp(op="mul", a="r1", b=2, var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        rng = random.Random(42)
        interpreter = Interpreter(rng=rng.random, debug=True)
        result, randoms, debug_info = interpreter.execute(plan)
        
        assert debug_info is not None
        assert "execution_trace" in debug_info
        assert "final_vars" in debug_info
        assert len(debug_info["execution_trace"]) == 3  # rand, mul, return
    
    def test_result_str_has_10_decimals(self):
        """result_str is formatted with exactly 10 decimal places."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="r1"),
            ReturnOp(op="return", value="r1")
        ])
        
        rng = random.Random(42)
        interpreter = Interpreter(rng=rng.random, debug=True)
        result, randoms, debug_info = interpreter.execute(plan)
        
        result_str = debug_info["result_str"]
        decimal_part = result_str.split(".")[1]
        assert len(decimal_part) == 10
    
    def test_debug_trace_structure(self):
        """Debug trace has correct structure."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="r1"),
            MulOp(op="mul", a="r1", b=5, var="x"),
            ReturnOp(op="return", value="x")
        ])
        
        rng = random.Random(42)
        interpreter = Interpreter(rng=rng.random, debug=True)
        result, randoms, debug_info = interpreter.execute(plan)
        
        trace = debug_info["execution_trace"]
        
        # Check rand step
        assert trace[0]["op"] == "rand"
        assert trace[0]["step"] == 1
        assert "random_generated" in trace[0]
        assert trace[0]["random_generated"] is not None
        
        # Check mul step
        assert trace[1]["op"] == "mul"
        assert trace[1]["step"] == 2
        assert "inputs" in trace[1]
        assert "output" in trace[1]
        
        # Check return step
        assert trace[2]["op"] == "return"
        assert trace[2]["step"] == 3
    
    def test_debug_stats(self):
        """Debug stats are calculated correctly."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="r1"),
            RandOp(op="rand", var="r2"),
            MulOp(op="mul", a="r1", b="r2", var="result"),
            ReturnOp(op="return", value="result")
        ])
        
        rng = random.Random(42)
        interpreter = Interpreter(rng=rng.random, debug=True)
        result, randoms, debug_info = interpreter.execute(plan)
        
        assert debug_info["steps_executed"] == 4
        assert debug_info["random_count"] == 2
        assert debug_info["max_depth_reached"] == 0  # No conditionals
    
    def test_debug_with_conditionals(self):
        """Debug mode tracks nesting depth with conditionals."""
        plan = ExecutionPlan(steps=[
            RandOp(op="rand", var="r1"),
            IfOp(
                op="if",
                condition=Condition(left="r1", operator="<", right=0.5),
                then_steps=[
                    MulOp(op="mul", a="r1", b=2, var="result"),
                    ReturnOp(op="return", value="result")
                ],
                else_steps=[
                    DivOp(op="div", a="r1", b=2, var="result"),
                    ReturnOp(op="return", value="result")
                ]
            )
        ])
        
        rng = random.Random(42)
        interpreter = Interpreter(rng=rng.random, debug=True)
        result, randoms, debug_info = interpreter.execute(plan)
        
        assert debug_info["max_depth_reached"] == 1  # One level of nesting
    
    def test_api_debug_mode(self):
        """Test debug mode via API."""
        plan_dict = {
            "steps": [
                {"op": "rand", "var": "r1"},
                {"op": "mul", "a": "r1", "b": 10, "var": "result"},
                {"op": "return", "value": "result"}
            ]
        }
        
        response = client.post("/execute_plan", json={
            "plan": plan_dict,
            "seed": 42,
            "debug": True
        })
        
        assert response.status_code == 200
        data = response.json()
        
        # Check basic response
        assert "result" in data
        assert "random_integers" in data
        
        # Check debug info
        assert "debug" in data
        debug = data["debug"]
        assert "plan" in debug
        assert "execution_trace" in debug
        assert "final_vars" in debug
        assert "result_str" in debug
        assert "seed_used" in debug
        assert debug["seed_used"] == 42
        assert "stats" in debug
        assert "limits" in debug
        
        # Check stats
        stats = debug["stats"]
        assert "steps_executed" in stats
        assert "random_count" in stats
        assert "max_depth_reached" in stats
        
        # Check limits
        limits = debug["limits"]
        assert limits["max_steps"] == 200
        assert limits["max_depth"] == 5


class TestExecutePlanEndpoint:
    """Test direct plan execution endpoint."""
    
    def test_execute_plan_basic(self):
        """Test basic plan execution."""
        plan_dict = {
            "steps": [
                {"op": "rand", "var": "r1"},
                {"op": "return", "value": "r1"}
            ]
        }
        
        response = client.post("/execute_plan", json={"plan": plan_dict})
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "random_integers" in data
        assert len(data["random_integers"]) == 1
    
    def test_execute_plan_with_operations(self):
        """Test plan with multiple operations."""
        plan_dict = {
            "steps": [
                {"op": "rand", "var": "r1"},
                {"op": "rand", "var": "r2"},
                {"op": "mul", "a": "r1", "b": "r2", "var": "product"},
                {"op": "sqrt", "x": "product", "var": "result"},
                {"op": "return", "value": "result"}
            ]
        }
        
        response = client.post("/execute_plan", json={
            "plan": plan_dict,
            "seed": 42
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert len(data["random_integers"]) == 2
    
    def test_execute_plan_error_handling(self):
        """Test error handling in direct execution."""
        plan_dict = {
            "steps": [
                {"op": "div", "a": 1, "b": 0, "var": "result"},
                {"op": "return", "value": "result"}
            ]
        }
        
        response = client.post("/execute_plan", json={"plan": plan_dict})
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "DIVISION_BY_ZERO"
    
    def test_execute_plan_invalid_schema(self):
        """Test invalid plan schema."""
        plan_dict = {
            "steps": [
                {"op": "invalid_op", "var": "r1"}
            ]
        }
        
        response = client.post("/execute_plan", json={"plan": plan_dict})
        
        # Should fail validation
        assert response.status_code == 422


class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_seed_and_debug_together(self):
        """Test using seed and debug mode together."""
        plan_dict = {
            "steps": [
                {"op": "rand", "var": "r1"},
                {"op": "mul", "a": "r1", "b": 3, "var": "result"},
                {"op": "return", "value": "result"}
            ]
        }
        
        # First execution
        response1 = client.post("/execute_plan", json={
            "plan": plan_dict,
            "seed": 100,
            "debug": True
        })
        
        # Second execution with same seed
        response2 = client.post("/execute_plan", json={
            "plan": plan_dict,
            "seed": 100,
            "debug": True
        })
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Results should be identical
        assert data1["result"] == data2["result"]
        assert data1["random_integers"] == data2["random_integers"]
        
        # Debug info should also match
        assert data1["debug"]["result_str"] == data2["debug"]["result_str"]
        assert data1["debug"]["final_vars"] == data2["debug"]["final_vars"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

