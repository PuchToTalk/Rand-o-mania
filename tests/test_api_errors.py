"""
Integration tests for API error responses.

Tests that error responses follow the structured format.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schema import ExecutionPlan, RandOp, DivOp, ReturnOp


# Create test client
client = TestClient(app)


class TestErrorFormat:
    """Test that all errors follow the structured format."""
    
    def test_error_has_correct_structure(self):
        """Test that error responses have the correct structure."""
        # This will fail because LLM is not initialized in tests
        response = client.post(
            "/execute",
            json={"prompt": "Generate a random number"}
        )
        
        assert response.status_code in [400, 422, 500]
        data = response.json()
        
        # Verify structure
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert isinstance(data["error"]["code"], str)
        assert isinstance(data["error"]["message"], str)
    
    def test_llm_not_initialized_error(self):
        """Test error when LLM processor is not initialized."""
        response = client.post(
            "/execute",
            json={"prompt": "Test prompt"}
        )
        
        # Should return 500 with LLM_NOT_INITIALIZED
        data = response.json()
        assert response.status_code == 500
        assert data["error"]["code"] == "LLM_NOT_INITIALIZED"
        assert "OPENAI_API_KEY" in data["error"]["message"]
    
    def test_invalid_prompt_format(self):
        """Test error with invalid prompt (empty string)."""
        response = client.post(
            "/execute",
            json={"prompt": ""}
        )
        
        # Should fail validation
        assert response.status_code == 422


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test that health endpoint returns correct structure."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "llm_ready" in data
        assert data["status"] == "healthy"
        assert isinstance(data["llm_ready"], bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

