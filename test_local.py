# test_local.py
from app.interpreter import Interpreter
from app.schema import (
    ExecutionPlan, RandOp, MulOp, DivOp, SqrtOp, IfOp, ReturnOp, Condition
)

# Simuler l'exemple de la spec
def test_example():
    plan = ExecutionPlan(steps=[
        # Generate random number
        RandOp(op="rand", var="r1"),
        
        # If r1 < 0.5: multiply by 0.1234567, else: divide by 1.1234567
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
        
        # Generate another random number
        RandOp(op="rand", var="r2"),
        
        # Square root of r2
        SqrtOp(op="sqrt", x="r2", var="sqrt_r2"),
        
        # Multiply sqrt_r2 by intermediate result
        MulOp(op="mul", a="sqrt_r2", b="intermediate", var="final"),
        
        # Return final result
        ReturnOp(op="return", value="final")
    ])
    
    # Execute
    interpreter = Interpreter()
    result, randoms = interpreter.execute(plan)
    
    print("success !")
    print(f"Result: {result}")
    print(f"Random numbers: {randoms}")
    print(f"\nJSON Response format:")
    print(f'{{"result": {result}, "random_integers": {randoms}}}')

if __name__ == "__main__":
    test_example()