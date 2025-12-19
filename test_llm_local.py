# test_llm_local.py
import os
from app.llm_processor import LLMProcessor, LLMProcessingError

def test_llm_simple():
    try:
        # Initialiser le processeur
        processor = LLMProcessor()
        
        # Prompt simple
        prompt = "Generate a random number and multiply it by 2"
        
        print(f"Envoi du prompt: {prompt}")
        print("Attente réponse OpenAI...")
        
        # Conversion en plan d'exécution
        plan = processor.process(prompt)
        
        print("Plan d'exécution reçu !")
        print(f"Nombre d'étapes: {len(plan.steps)}")
        print(f"\nPlan détaillé:")
        for i, step in enumerate(plan.steps, 1):
            print(f"  {i}. {step}")
        
        return plan
        
    except LLMProcessingError as e:
        print(f"Erreur LLM: {e}")
        return None
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        return None

def test_llm_complex():
    try:
        processor = LLMProcessor()
        
        prompt = """Generate a random number. If the number is less than 0.5, 
        multiply it by 0.1234567, otherwise divide it by 1.1234567. 
        Generate another random number, get the square root of it, 
        and then multiply it by the previous result."""
        
        print(f"Envoi du prompt complexe...")
        print("Attente réponse OpenAI...")
        
        plan = processor.process(prompt)
        
        print("Plan complexe reçu !")
        print(f"Nombre d'étapes: {len(plan.steps)}")
        
        # Exécuter le plan
        from app.interpreter import Interpreter
        interpreter = Interpreter()
        result, randoms = interpreter.execute(plan)
        
        print(f"\nfinal result:")
        print(f"  Result: {result}")
        print(f"  Random numbers: {randoms}")
        
        return plan
        
    except Exception as e:
        print(f"error: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Simple prompt")
    print("=" * 60)
    test_llm_simple()
    
    print("\n" + "=" * 60)
    print("TEST 2: Complexe prompt")
    print("=" * 60)
    test_llm_complex()