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


class SubmitStepToolResult(TypedDict):
    step_number: int
    value: Any
    submitted: bool


def submit_step_tool(step_number: int, value: Any) -> SubmitStepToolResult:
    """
    Tool for submitting intermediate step results for verification.
    """
    return {"step_number": step_number, "value": value, "submitted": True}


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
) -> tuple[Any | None, dict[int, float]]:
    """Fallback when the Anthropic SDK is unavailable - returns None since we require API."""
    if verbose:
        print("Anthropic SDK unavailable - API is required for this task.")
    return (None, {})


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 20,
    model: str = "claude-3-5-haiku-latest",
    verbose: bool = True,
) -> tuple[Any | None, dict[int, float]]:
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
        Tuple of (submitted_answer, submitted_steps_dict) if submit_answer was called, otherwise (None, submitted_steps_dict)
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
    
    # Track intermediate step submissions for validation
    submitted_steps: dict[int, float] = {}

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
                    elif tool_name == "submit_step":
                        assert isinstance(tool_input, dict) and "step_number" in tool_input and "value" in tool_input
                        result = handler(tool_input["step_number"], tool_input["value"])
                        step_num = result["step_number"]
                        step_val = result["value"]
                        try:
                            submitted_steps[step_num] = float(step_val)
                        except (ValueError, TypeError):
                            pass
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

            # If an answer was submitted, return it with step tracking
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                # Return tuple: (answer, submitted_steps)
                return (submitted_answer, submitted_steps)
        else:
            # No tool use, conversation might be complete
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return (None, submitted_steps)


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

    result_data = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=10,  # Increased to allow for step submissions
        verbose=verbose,
    )

    # Unpack result - can be tuple (answer, steps) or just answer for backward compatibility
    if isinstance(result_data, tuple):
        result, submitted_steps = result_data
    else:
        result = result_data
        submitted_steps = {}

    # Validate intermediate steps
    step_validation_passed = True
    expected_steps = {
        1: 30.6,  # weighted_sum: 26*0.1 + 20*0.2 + 36*0.25 + 42*0.3 + 16*0.15 = 30.6
        2: 1.0,   # total_weights: 0.1 + 0.2 + 0.25 + 0.3 + 0.15 = 1.0
        3: 30.6,  # mean: 30.6 / 1.0 = 30.6
        4: 3060.0, # scaled: 30.6 * 100 = 3060.0
    }
    
    # Check if required steps were submitted and are correct
    for step_num, expected_val in expected_steps.items():
        if step_num not in submitted_steps:
            step_validation_passed = False
            break
        submitted_val = submitted_steps[step_num]
        if abs(float(submitted_val) - expected_val) > 0.1:  # Allow small tolerance
            step_validation_passed = False
            break

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
            answer_correct = abs(result_float - expected_float) < 0.05
            # Both answer and steps must be correct
            success = answer_correct and step_validation_passed
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
            "name": "submit_step",
            "description": "Submit an intermediate step result for verification. You must submit steps 1-4 before submitting the final answer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "step_number": {
                        "type": "integer",
                        "description": "The step number (1, 2, 3, or 4)",
                    },
                    "value": {
                        "description": "The numeric result of this step",
                    },
                },
                "required": ["step_number", "value"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final answer (step 5 result)",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "The final answer to submit"}},
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "python_expression": python_expression_tool,
        "submit_step": submit_step_tool,
        "submit_answer": submit_answer_tool,
    }

    # Run the test 10 times and track success rate
    num_runs = 10
    
    # Task: Calculate weighted metric with intermediate step verification
    # Requires submitting each intermediate step for validation
    prompt = """You are computing a weighted metric for a machine learning evaluation pipeline. Given feature values [26, 20, 36, 42, 16] and sample weights [0.1, 0.2, 0.25, 0.3, 0.15], calculate the normalized weighted average using this exact procedure.

You must execute these steps in this precise sequence and submit each intermediate result:

Step 1: Calculate weighted_sum = sum of (each value multiplied by its corresponding weight)
  - Use python_expression to compute this
  - Submit the result using submit_step with step_number=1

Step 2: Calculate total_weights = sum of all weights
  - Use python_expression to compute this
  - Submit the result using submit_step with step_number=2

Step 3: Calculate mean = weighted_sum divided by total_weights
  - Use python_expression to compute this (divide the result from step 1 by the result from step 2)
  - Submit the result using submit_step with step_number=3

Step 4: Scale the mean by multiplying it by 100
  - Use python_expression to compute this (multiply the result from step 3 by 100)
  - Submit the result using submit_step with step_number=4

Step 5: Round the scaled result to exactly 1 decimal place using round(scaled_result, 1)
  - Use python_expression to compute this (round the result from step 4 to 1 decimal)
  - Submit the final result using submit_answer

Critical requirements:
- You MUST submit steps 1-4 using submit_step before submitting the final answer
- Steps must be executed and submitted in order - do not skip any step submissions
- Step 4 (multiply by 100) must occur AFTER step 3 (division) - use the result from step 3
- Use round(x, 1) with second argument = 1 (not 2 or any other value)
- Submit all values as float numbers, not strings
- Do not combine steps into a single expression - each step must be computed separately

Use python_expression for each calculation, then submit_step for steps 1-4, and finally submit_answer for step 5."""
    
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
