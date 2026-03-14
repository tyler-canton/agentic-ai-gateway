"""
Agentic AI Gateway - Multi-Agent Example with Tool Calling

Shows how to use the converse() method for multi-agent workflows
with tool calling and automatic fallbacks.
"""

from agentic_ai_gateway import create_bedrock_gateway

# Create gateway with fallbacks
gateway = create_bedrock_gateway(
    primary_model="anthropic.claude-3-sonnet-20240229-v1:0",
    fallback_models=[
        "anthropic.claude-3-haiku-20240307-v1:0",
        "meta.llama3-70b-instruct-v1:0",
    ],
    region="us-east-1"
)

# Define tools for the agent
tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_patient_data",
                "description": "Retrieve patient data from the medical records system",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "The patient ID to look up"
                            }
                        },
                        "required": ["patient_id"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "schedule_appointment",
                "description": "Schedule an appointment for a patient",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "string",
                                "description": "The patient ID"
                            },
                            "date": {
                                "type": "string",
                                "description": "Appointment date (YYYY-MM-DD)"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for the appointment"
                            }
                        },
                        "required": ["patient_id", "date", "reason"]
                    }
                }
            }
        }
    ]
}

# System prompt for the agent
system_prompt = [
    {
        "text": """You are a healthcare assistant agent. You help with patient data
retrieval and appointment scheduling. Use the available tools to complete tasks.
Always verify patient information before making changes."""
    }
]

# Conversation messages
messages = [
    {
        "role": "user",
        "content": [
            {
                "text": "Can you look up patient P001 and schedule a follow-up for next Monday?"
            }
        ]
    }
]

# Use converse() with tool calling - includes automatic fallback
result = gateway.converse(
    messages=messages,
    system=system_prompt,
    tool_config=tool_config,
    inference_config={
        "maxTokens": 4096,
        "temperature": 0.1
    }
)

print(f"Model used: {result['model_used']}")
print(f"Latency: {result['latency_ms']}ms")
print(f"Fallback used: {result['fallback_used']}")
print(f"Input tokens: {result['input_tokens']}")
print(f"Output tokens: {result['output_tokens']}")

# Access the raw Bedrock response
response = result["response"]
stop_reason = response.get("stopReason")
print(f"Stop reason: {stop_reason}")

# Check if the model wants to use a tool
output = response.get("output", {})
message = output.get("message", {})
content = message.get("content", [])

for block in content:
    if "text" in block:
        print(f"\nAssistant: {block['text']}")
    elif "toolUse" in block:
        tool_use = block["toolUse"]
        print(f"\nTool call: {tool_use['name']}")
        print(f"Input: {tool_use['input']}")


# Example: Agentic loop with tool execution
def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Mock tool execution - replace with real implementations."""
    if tool_name == "get_patient_data":
        return f"Patient {tool_input['patient_id']}: John Doe, Age 45, Last visit: 2024-01-15"
    elif tool_name == "schedule_appointment":
        return f"Appointment scheduled for {tool_input['patient_id']} on {tool_input['date']}"
    return "Unknown tool"


def run_agent_loop(gateway, initial_message: str, max_turns: int = 5):
    """Run an agentic loop with tool calling."""
    messages = [
        {
            "role": "user",
            "content": [{"text": initial_message}]
        }
    ]

    for turn in range(max_turns):
        result = gateway.converse(
            messages=messages,
            system=system_prompt,
            tool_config=tool_config,
            inference_config={"maxTokens": 4096, "temperature": 0.1}
        )

        response = result["response"]
        stop_reason = response.get("stopReason")
        assistant_message = response.get("output", {}).get("message", {})

        # Add assistant response to messages
        messages.append(assistant_message)

        if stop_reason == "end_turn":
            # Agent is done
            for block in assistant_message.get("content", []):
                if "text" in block:
                    print(f"\nFinal response: {block['text']}")
            break

        elif stop_reason == "tool_use":
            # Process tool calls
            tool_results = []
            for block in assistant_message.get("content", []):
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    print(f"\nExecuting tool: {tool_use['name']}")

                    result_text = execute_tool(tool_use["name"], tool_use["input"])
                    tool_results.append({
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "content": [{"text": result_text}]
                        }
                    })

            # Add tool results to messages
            messages.append({
                "role": "user",
                "content": tool_results
            })

    return messages


# Run the agent loop
print("\n" + "=" * 50)
print("Running agent loop...")
print("=" * 50)

final_messages = run_agent_loop(
    gateway,
    "Look up patient P001 and tell me their last visit date"
)

# Check metrics
print("\n" + "=" * 50)
print("Gateway Metrics:")
print(gateway.get_metrics())
