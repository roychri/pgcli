"""Simple test to verify pydantic-ai works correctly."""
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

# Create a simple agent with TestModel (no API keys needed)
agent = Agent(
    TestModel(),
    system_prompt='You are a helpful assistant.',
)

# Test sync run
result = agent.run_sync('Say hello!')
print(f"✓ pydantic-ai works! Response: {result.output}")
