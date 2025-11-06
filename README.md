# Multi-Dataset Metric Normalization RL Task

An RL task for LLM training that tests a model's ability to perform complex multi-step calculations with conditional logic, CSV data processing, and precise intermediate step verification.

## Implementation Overview

This task implements a multi-dataset metric normalization routine that requires the model to:

1. **Read and parse CSV files** - Extract feature values and sample weights from two datasets
2. **Perform sequential calculations** - Execute 15 steps in strict order with dependencies
3. **Apply conditional logic** - Choose which dataset to use based on intermediate calculations
4. **Submit intermediate results** - Verify each step before proceeding (steps 1-7, 9-14)
5. **Handle precision** - Use Decimal rounding with specific quantization rules

The task is designed to achieve a **10-40% pass rate** by introducing multiple failure modes:
- CSV parsing errors
- Floating-point precision issues
- Conditional logic mistakes
- Missing intermediate step submissions
- Incorrect value reuse from previous steps
- Rounding errors

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd hello-py
   ```

2. Create a virtual environment:
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install anthropic python-dotenv
   ```

4. Set up your Anthropic API key:
   Create a `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

5. Ensure CSV datasets exist:
   The task requires `dataset_a.csv` and `dataset_b.csv` in the same directory as `main.py`.

## Running the Task

Run the task with:
```bash
python main.py
```

The script will:
- Run 10 test iterations
- Execute concurrently by default (can be changed to sequential)
- Display verbose output showing model interactions
- Print pass/fail results for each run
- Calculate and display the final pass rate

### Execution Modes

Edit the `concurrent` parameter at the bottom of `main.py`:

```python
# Concurrent execution (faster, but may hit rate limits)
asyncio.run(main(concurrent=True))

# Sequential execution (slower, safer for rate limits)
asyncio.run(main(concurrent=False))
```

## Understanding Results

### Output Format

The task outputs:
- **Per-run results**: `✓ Run X: SUCCESS` or `✗ Run X: FAILURE`
- **Final statistics**: Pass rate percentage
- **Verbose logs**: Model tool calls, calculations, and submissions

### Example Output

```
============================================================
Test Results:
  Passed: 6/10
  Failed: 4/10
  Pass Rate: 60.0%
============================================================
TRAINING_OK
```

### Success Criteria

A run succeeds only if:
1. **All intermediate steps** (1-7, 9-14) are submitted correctly within tolerance
2. **Final answer** matches expected value (668.118) within 0.001
3. **Steps submitted in order** without skipping
4. **Correct dataset selected** based on conditional logic (step 8)

### Common Failure Modes

- **CSV parsing errors**: Incorrectly reading or converting CSV values
- **Wrong dataset selection**: Using Dataset A when B should be used (or vice versa)
- **Missing steps**: Skipping required intermediate step submissions
- **Precision errors**: Floating-point calculation mistakes
- **Value reuse errors**: Recomputing instead of using submitted step values
- **Rounding errors**: Incorrect Decimal quantization

## Adjusting Success Rate

The current implementation achieves approximately **60% pass rate**. To reduce it to **40% or less**, consider:

### 1. Increase Prompt Ambiguity

**Current approach**: Prompt specifies what to do but not always how.

**Ways to increase ambiguity:**
- Remove explicit references to "max - min" for range calculation
- Use more abstract terminology (e.g., "spread" instead of "range")
- Remove explicit formula hints
- Make conditional logic less explicit
- Use domain-specific jargon that requires inference

**Example modification:**
```python
# Instead of: "Range (max - min) for the selected dataset's feature values"
# Use: "Dispersion metric for the selected dataset's feature values"
```

### 2. Increase Dataset Complexity

**Current datasets**: 18 records per dataset with 3-4 decimal places.

**Ways to increase complexity:**
- **More records**: Increase to 25-30 records per dataset
- **More decimal places**: Use 5-6 decimal places for values and weights
- **Non-normalized weights**: Ensure weights don't sum to exactly 1.0 (already done for Dataset B: 0.9999)
- **Wider value ranges**: Increase spread between min/max values
- **Mixed precision**: Some values with 2 decimals, others with 6
- **Edge cases**: Include values very close to zero or very large numbers

**Example modification:**
```python
# Generate datasets with 30 records, 6 decimal places
values_a = [round(random.uniform(10.0, 50.0), 6) for _ in range(30)]
weights_a = [round(random.uniform(0.02, 0.15), 6) for _ in range(30)]
```

### 3. Add More Ambiguous Steps

**Current**: 15 steps with clear dependencies.

**Ways to add ambiguity:**
- Add steps that require interpretation (e.g., "normalize the variance")
- Include steps with multiple valid interpretations
- Add conditional steps that depend on non-obvious conditions
- Require the model to infer calculation order for some steps

### 4. Increase Calculation Complexity

**Ways to make calculations harder:**
- Add more intermediate transformations
- Require chained calculations with more dependencies
- Introduce steps that require careful precision management
- Add steps that are easy to compute incorrectly (e.g., division by small numbers)

### 5. Stricter Validation

**Ways to increase failure rate:**
- Reduce tolerance for floating-point comparisons (currently 0.01 for steps, 0.001 for final)
- Require exact matches for more steps
- Add validation for calculation methods, not just results
- Check that models use submitted values rather than recomputing

### 6. Add Red Herrings

**Ways to mislead models:**
- Include unnecessary data in CSV files
- Add steps that seem important but aren't
- Use terminology that could be misinterpreted
- Include calculations that look correct but use wrong data

### 7. Reduce Max Steps

**Current**: 100 max steps.

**Ways to increase pressure:**
- Reduce to 60-70 steps to create time pressure
- This forces models to be more efficient and increases chance of skipping steps

## Implementation Details

### Task Structure

- **15 sequential steps** with strict ordering requirements
- **Conditional logic** at step 8 (dataset selection)
- **14 intermediate step submissions** for verification
- **1 final answer submission** with Decimal rounding

### Validation Logic

- **Step validation**: Checks all 14 intermediate steps within 0.01 tolerance
- **Final answer validation**: Checks within 0.001 tolerance
- **Order validation**: Steps must be submitted sequentially
- **Dataset validation**: Correct dataset must be used based on step 7 result

### Error Handling

- **Rate limiting**: Exponential backoff for 429 errors
- **Invalid tool input**: Graceful error messages to model
- **API failures**: Falls back to deterministic computation (returns None)
- **CSV file validation**: Checks for required files before execution

## Files

- `main.py`: Core task implementation and agent loop
- `dataset_a.csv`: First dataset with feature values and weights
- `dataset_b.csv`: Second dataset with feature values and weights
- `expected_answer.txt`: Expected calculation breakdown for reference
- `.env`: Environment variables (not committed, add your API key here)

## Requirements

- Python 3.13+
- Anthropic API key
- `anthropic` package
- `python-dotenv` package

## Credits

- **Codex**: Initial discovery of codebase and initial setup/context
- **Cursor**: Generated README documentation and created datasets A and B
