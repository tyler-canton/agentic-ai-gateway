"""
Agentic AI Gateway - AWS Bedrock Example

Shows how to set up fallback chains and canary deployments with Bedrock.
"""

from agentic_ai_gateway import create_bedrock_gateway

# Basic usage with fallback
gateway = create_bedrock_gateway(
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    fallback_models=[
        "anthropic.claude-3-haiku-20240307-v1:0",
        "meta.llama3-70b-instruct-v1:0",
    ],
    region="us-east-1"
)

# Make a request
response = gateway.invoke(
    prompt="What is the capital of France?",
    max_tokens=100,
    temperature=0.7
)

print(f"Answer: {response.content}")
print(f"Model used: {response.model_used}")
print(f"Latency: {response.latency_ms}ms")
print(f"Fallback used: {response.fallback_used}")


# With canary deployment
canary_gateway = create_bedrock_gateway(
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    canary_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    canary_percentage=10,  # 10% traffic to new model
    fallback_models=["anthropic.claude-3-haiku-20240307-v1:0"],
    region="us-east-1"
)

# Run multiple requests to see canary distribution
for i in range(10):
    response = canary_gateway.invoke("Hello!")
    canary_tag = " (CANARY)" if response.canary_used else ""
    print(f"Request {i+1}: {response.model_used}{canary_tag}")

# Check metrics
print("\nMetrics:")
print(canary_gateway.get_metrics())


# Runtime configuration update
print("\nUpdating canary to 50%...")
canary_gateway.update_config(canary_percentage=50)

for i in range(10):
    response = canary_gateway.invoke("Hello!")
    canary_tag = " (CANARY)" if response.canary_used else ""
    print(f"Request {i+1}: {response.model_used}{canary_tag}")
