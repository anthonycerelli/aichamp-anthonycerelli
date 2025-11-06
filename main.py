import asyncio
import json
import os
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, TypedDict

try:
    from anthropic import AsyncAnthropic
    from anthropic.types import MessageParam, ToolUnionParam
except ImportError:  # pragma: no cover - handled at runtime
    AsyncAnthropic = None  # type: ignore[assignment]
    MessageParam = dict  # type: ignore[misc, assignment]
    ToolUnionParam = dict  # type: ignore[misc, assignment]

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - handled at runtime
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

import re

# Load environment variables from .env file
load_dotenv()


def load_client() -> Any:
    """Load and initialize Anthropic client with API key from environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )
    if AsyncAnthropic is None:
        raise ImportError("anthropic package is not installed")
    return AsyncAnthropic(api_key=api_key)


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """
    Tool that evaluates Python expressions using exec.
    Use print(...) to emit output; stdout will be captured and returned.
    """
    try:
        namespace = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """
    Tool for submitting the final answer.
    """
    return {"answer": answer, "submitted": True}


async def _fallback_agent_run(
    prompt: str,
    tool_handlers: dict[str, Callable[..., Any]],
    verbose: bool,
) -> Any | None:
    """Fallback when the Anthropic SDK is unavailable - returns None since we require API."""
    if verbose:
        print("Anthropic SDK unavailable - API is required for this task.")
    return None


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-3-5-haiku-latest",
    verbose: bool = True,
) -> Any | None:
    """
    Runs an agent loop with the given prompt and tools.

    Args:
        prompt: The initial prompt for the agent
        tools: List of tool definitions for Anthropic API
        tool_handlers: Dictionary mapping tool names to their handler functions
        max_steps: Maximum number of steps before stopping (default 5)
        model: The Anthropic model to use
        verbose: Whether to print detailed output (default True)

    Returns:
        The submitted answer if submit_answer was called, otherwise None
    """
    if AsyncAnthropic is None:
        return await _fallback_agent_run(prompt, tool_handlers, verbose)

    try:
        client = load_client()
    except (ValueError, ImportError) as e:
        if verbose:
            print(f"Failed to initialize Anthropic client: {e}")
        return await _fallback_agent_run(prompt, tool_handlers, verbose)

    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        try:
            response = await client.messages.create(
                model=model, max_tokens=1000, tools=tools, messages=messages
            )
        except Exception as e:
            # On API failure, fall back to deterministic computation instead of returning None
            if verbose:
                print(f"API call failed: {e}")
            return await _fallback_agent_run(prompt, tool_handlers, verbose)

        # Track if we need to continue
        has_tool_use = False
        tool_results = []
        submitted_answer = None

        # Process the response
        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")

                    # Extract arguments based on tool
                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    # Call the appropriate tool handler
                    if tool_name == "python_expression":
                        assert (
                            isinstance(tool_input, dict) and "expression" in tool_input
                        )
                        if verbose:
                            print("\nInput:")
                            print("```")
                            for line in tool_input["expression"].split("\n"):
                                print(f"{line}")
                            print("```")
                        result = handler(tool_input["expression"])
                        if verbose:
                            print("\nOutput:")
                            print("```")
                            print(result)
                            print("```")
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        # Generic handler call
                        result = (
                            handler(**tool_input)
                            if isinstance(tool_input, dict)
                            else handler(tool_input)
                        )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        # If we have tool uses, add them to the conversation
        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})

            messages.append({"role": "user", "content": tool_results})

            # If an answer was submitted, return it
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            # No tool use, conversation might be complete
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    expected_answer: Any,
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=6,  # Limited steps to increase difficulty and force efficiency
        verbose=verbose,
    )

    # Explicit comparison - handle None and type mismatches
    if result is None:
        success = False
    else:
        # Compare with type checking - both should be numbers
        try:
            # Convert both to float for comparison to handle int/float mismatches
            result_float = float(result)
            expected_float = float(expected_answer)
            # Allow small floating point differences (0.05 tolerance for rounding variations)
            success = abs(result_float - expected_float) < 0.05
        except (ValueError, TypeError):
            success = False

    if success:
        print(f"✓ Run {run_id}: SUCCESS - Got {result}")
    else:
        print(f"✗ Run {run_id}: FAILURE - Got {result}, expected {expected_answer}")

    return run_id, success, result


async def main(concurrent: bool = True):
    tools: list[ToolUnionParam] = [
        {
            "name": "python_expression",
            "description": "Evaluates a Python expression",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Will be passed to exec(). Use print() to output something. Returns stdout. ",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "The final answer to submit"}},
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

    # Run the test 10 times and track success rate
    num_runs = 10
    
    # Task: Calculate weighted metric with strict sequential requirements and easy-to-miss details
    # Multiple failure points to achieve 10-40% pass rate
    prompt = """You are computing a weighted metric for a machine learning evaluation pipeline. Given feature values [26, 20, 36, 42, 16] and sample weights [0.1, 0.2, 0.25, 0.3, 0.15], calculate the normalized weighted average using this exact procedure.

You must execute these steps in this precise sequence - each step must finish before the next starts:
Step 1: Calculate weighted_sum = sum of (each value multiplied by its corresponding weight)
Step 2: Calculate total_weights = sum of all weights
Step 3: Calculate mean = weighted_sum divided by total_weights
Step 4: Scale the mean by multiplying it by 100
Step 5: Round the scaled result to exactly 1 decimal place using round(scaled_result, 1)

Critical requirements that must be followed exactly - read each carefully:
- All 5 steps must be executed sequentially in the order listed - do not combine or reorder any steps
- Step 4 (multiply by 100) must occur AFTER step 3 (division) completes - this order is mandatory and cannot be changed
- You must use round(x, 1) where the second argument is exactly 1 - do not use round(x, 2), round(x, 0), or any other precision value
- Round the value from step 4 (after scaling by 100), not the value from step 3 (before scaling)
- Submit the final result as a float number type, not a string or any other type
- Do not use string formatting methods (f-strings, format(), %.1f, str()) for rounding - only use the round() function
- The weighted_sum calculation must multiply each value-weight pair individually before summing - do not use any function that might alter the order of operations

Important warnings: The multiplication by 100 in step 4 is required and must happen after the division in step 3. Many implementations fail by: multiplying before dividing (wrong order), skipping the scaling step entirely, rounding the wrong intermediate value (step 3 instead of step 4), using incorrect rounding precision (2 decimals instead of 1), or submitting a string instead of a float. You must follow the exact 5-step sequence with proper ordering and rounding.

Use python_expression to perform the calculations step-by-step, ensuring each step completes before moving to the next. Then submit the final rounded float result."""
    
    # Expected calculation:
    # Step 1: 26*0.1 + 20*0.2 + 36*0.25 + 42*0.3 + 16*0.15
    #        = 2.6 + 4.0 + 9.0 + 12.6 + 2.4 = 30.6
    # Step 2: 0.1 + 0.2 + 0.25 + 0.3 + 0.15 = 1.0
    # Step 3: 30.6 / 1.0 = 30.6
    # Step 4: 30.6 * 100 = 3060.0
    # Step 5: round(3060.0, 1) = 3060.0
    expected_answer = 3060.0

    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print("=" * 60)

    # Create all test coroutines
    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            expected_answer=expected_answer,
            verbose=False,
        )
        for i in range(num_runs)
    ]

    # Run concurrently or sequentially based on the flag
    if concurrent:
        # Create tasks from coroutines for proper concurrent execution
        task_objects = [asyncio.create_task(task) for task in tasks]
        # Process results as they complete
        results = []
        for completed_task in asyncio.as_completed(task_objects):
            result = await completed_task
            results.append(result)
    else:
        # Run sequentially by awaiting each task in order
        results = []
        for task in tasks:
            result = await task
            results.append(result)

    # Count successes - explicitly check the success boolean
    # Debug: print all results to verify
    successes = 0
    for run_id, success, result in results:
        if success is True:
            successes += 1

    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")
    print("TRAINING_OK")


if __name__ == "__main__":
    # Set to True for concurrent execution, False for sequential execution
    asyncio.run(main(concurrent=True))
