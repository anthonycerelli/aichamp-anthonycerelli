import asyncio
import json
import os
import time
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, TypedDict

try:
    from anthropic import AsyncAnthropic
    from anthropic.types import MessageParam, ToolUnionParam
except ImportError:
    AsyncAnthropic = None
    MessageParam = dict
    ToolUnionParam = dict

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

import re

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
    submitted_steps: dict[int, float] = {}

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        max_retries = 3
        retry_delay = 1.0
        for retry in range(max_retries):
            try:
                response = await client.messages.create(
                    model=model, max_tokens=800, tools=tools, messages=messages
                )
                break
            except Exception as e:
                error_str = str(e)
                is_rate_limit = (
                    "429" in error_str or 
                    "rate_limit" in error_str.lower() or
                    "rate limit" in error_str.lower() or
                    "exceed the rate limit" in error_str.lower()
                )
                
                if is_rate_limit:
                    if retry < max_retries - 1:
                        wait_time = retry_delay * (2 ** retry)
                        if verbose:
                            print(f"Rate limit hit, waiting {wait_time:.1f}s before retry {retry + 1}/{max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        if verbose:
                            print(f"Rate limit exceeded after {max_retries} retries, falling back")
                        return await _fallback_agent_run(prompt, tool_handlers, verbose)
                else:
                    if verbose:
                        print(f"API call failed: {e}")
                    return await _fallback_agent_run(prompt, tool_handlers, verbose)

        has_tool_use = False
        tool_results = []
        submitted_answer = None

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

                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    if tool_name == "python_expression":
                        if not isinstance(tool_input, dict) or "expression" not in tool_input:
                            if verbose:
                                print(f"Error: Invalid tool input for python_expression: {tool_input}")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": json.dumps({"error": "Invalid input format. Expected dict with 'expression' key."}),
                            })
                            continue
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
                        if not isinstance(tool_input, dict) or "step_number" not in tool_input or "value" not in tool_input:
                            if verbose:
                                print(f"Error: Invalid tool input for submit_step: {tool_input}")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": json.dumps({"error": "Invalid input format. Expected dict with 'step_number' and 'value' keys."}),
                            })
                            continue
                        result = handler(tool_input["step_number"], tool_input["value"])
                        step_num = result["step_number"]
                        step_val = result["value"]
                        try:
                            submitted_steps[step_num] = float(step_val)
                        except (ValueError, TypeError):
                            pass
                    elif tool_name == "submit_answer":
                        if not isinstance(tool_input, dict) or "answer" not in tool_input:
                            if verbose:
                                print(f"Error: Invalid tool input for submit_answer: {tool_input}")
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": json.dumps({"error": "Invalid input format. Expected dict with 'answer' key."}),
                            })
                            continue
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
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

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})

            messages.append({"role": "user", "content": tool_results})

            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return (submitted_answer, submitted_steps)
        else:
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
        max_steps=100,
        verbose=True,
    )

    if isinstance(result_data, tuple):
        result, submitted_steps = result_data
    else:
        result = result_data
        submitted_steps = {}

    step_validation_passed = True
    expected_steps = {
        1: 26.7925543,
        2: 1.0,
        3: 26.7925543,
        4: 19.2393817,
        5: 0.9999,
        6: 19.24130583058306,
        7: 7.55124846941694,
        9: 31.218,
        10: 18.7308,
        11: 8.0617543,
        12: 685.2491155,
        13: 17.1312278875,
        14: 668.1178876125,
    }

    for step_num, expected_val in expected_steps.items():
        if step_num not in submitted_steps:
            step_validation_passed = False
            break
        submitted_val = submitted_steps[step_num]
        if abs(float(submitted_val) - expected_val) > 0.01:
            step_validation_passed = False
            break

    if result is None:
        success = False
    else:
        try:
            result_float = float(result)
            expected_float = float(expected_answer)
            answer_correct = abs(result_float - expected_float) < 0.001
            success = answer_correct and step_validation_passed
        except (ValueError, TypeError):
            success = False

    if success:
        print(f"✓ Run {run_id}: SUCCESS - Got {result}")
    else:
        print(f"✗ Run {run_id}: FAILURE - Got {result}, expected {expected_answer}")
        if not step_validation_passed:
            print(f"  Step validation failed. Submitted steps: {submitted_steps}")
            print(f"  Expected steps: {expected_steps}")
        if result is None:
            print(f"  Model did not submit an answer (returned None)")
            if submitted_steps:
                print(f"  Model did submit {len(submitted_steps)} intermediate step(s): {submitted_steps}")
            else:
                print(f"  Model did not submit any intermediate steps")

    return run_id, success, result


async def main(concurrent: bool = True):
    csv_files = ['dataset_a.csv', 'dataset_b.csv']
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            raise FileNotFoundError(
                f"Required CSV file '{csv_file}' not found in current working directory: {os.getcwd()}. "
                f"Please ensure the CSV files are in the same directory as main.py."
            )
    
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

    num_runs = 10

    prompt = """
        Audit a multi-dataset metric normalization routine. Read 'dataset_a.csv' and 'dataset_b.csv' (feature_value, sample_weight columns). Process sequentially, submit each intermediate result.

        Steps:
        1. Dataset A weighted sum, submit_step(1)
        2. Dataset A total weights, submit_step(2)
        3. Dataset A baseline mean from steps 1-2, submit_step(3)
        4. Dataset B weighted sum, submit_step(4)
        5. Dataset B total weights, submit_step(5)
        6. Dataset B baseline mean from steps 4-5, submit_step(6)
        7. Absolute difference between step 3 and step 6, submit_step(7)
        8. Decision: if step 7 > 5.0, use Dataset A for remaining steps; otherwise use Dataset B (no submission)
        9. Range for the selected dataset's feature values, submit_step(9)
        10. Penalty: step 9 divided by selected dataset's total weights, multiplied by 0.6, submit_step(10)
        11. Adjusted mean: selected dataset's baseline mean minus step 10, submit_step(11)
        12. Scaled score: step 11 multiplied by 85, submit_step(12)
        13. Variance adjustment: step 12 divided by 100, then multiplied by 2.5, submit_step(13)
        14. Final adjusted: step 12 minus step 13, submit_step(14)
        15. Normalize step 14 to 3 decimal places using Decimal quantize with ROUND_HALF_UP, convert to float, submit_answer

        Critical: Submit steps 1-7 and 9-14 in sequence. Use previously submitted step values where referenced. All submissions must be numeric floats.
        """
    
    expected_answer = 668.118

    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print("=" * 60)

    tasks = [
        run_single_test(
            run_id=i + 1,
            num_runs=num_runs,
            prompt=prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            expected_answer=expected_answer,
            verbose=True,
        )
        for i in range(num_runs)
    ]

    if concurrent:
        task_objects = []
        for i, task in enumerate(tasks):
            task_objects.append(asyncio.create_task(task))
            if i < len(tasks) - 1:
                await asyncio.sleep(0.1)

        results = []
        for completed_task in asyncio.as_completed(task_objects):
            result = await completed_task
            results.append(result)
    else:
        results = []
        for task in tasks:
            result = await task
            results.append(result)
            await asyncio.sleep(0.2)

    successes = 0
    for run_id, success, result in results:
        if success is True:
            successes += 1

    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")
    print("TRAINING_OK")


if __name__ == "__main__":
    asyncio.run(main(concurrent=True))