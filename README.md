# Assignment 2 - Stolen Model Detection

Detect whether given suspect models are stolen versions of a target model.

## Overview

This assignment involves analyzing 360 suspect models to determine if they are stolen versions of the target model. Each suspect model needs to be evaluated and assigned a confidence score indicating the likelihood that it's a stolen model.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# or
source .venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:

```bash
pip install torch torchvision pandas numpy scikit-learn requests python-dotenv safetensors
```

3. Create a `.env` file in the project root and add your API key:

```env
API_KEY=your_api_key_here
```

You can copy `env.example` and replace the placeholder value.

## Project Structure

- **`target_model/`**: The target model to compare against
- **`suspect_models/`**: 360 suspect model files (suspect_000.safetensors to suspect_359.safetensors)
- **`task_template.py`**: Example code showing how to load models and format submissions
- **`submission.py`**: Your implementation for the model stealing detection algorithm
- **`env.example`**: Template for environment variables

## Model Loading

Use the provided example in `task_template.py` to load models from safetensors format:

```python
from safetensors.torch import load_file
state_dict = load_file("path/to/model.safetensors", device="cpu")
model = make_model()
model.load_state_dict(state_dict, strict=True)
model.eval()
```

## Submission Format

Your submission must be a CSV file with the following specifications:

**Required Format:**
- File extension: `.csv`
- Exactly two columns: `id`, `score` (column names must match exactly)
- Exactly 360 rows (one per suspect model, ids 0-359)
- Each id must appear exactly once

**Example:**
```csv
id,score
0,0.95
1,0.23
...
359,0.78
```

**Score Values:**
- Numeric values representing confidence that the model is stolen
- Can be probabilities (0.0-1.0) or raw model scores
- Must be finite numeric values (no strings, NaN, or infinity)
- Higher scores indicate higher likelihood of being stolen

## Implementation

Implement your model stealing detection algorithm in `submission.py`. Consider analyzing:
- Model architecture similarities
- Weight distributions and statistical properties
- Output behavior on test samples
- Other features that might indicate model theft

## Evaluation

Models are ranked by their confidence scores, evaluated using:
- **Metric**: True Positive Rate (TPR) at False Positive Rate (FPR) = 0.05
- The evaluation system will rank submissions by score values

## Resources

- See `task_template.py` for model loading examples
- See `submission.py` for submission format requirements and validation rules
- CIFAR-100 dataset can be automatically downloaded during model evaluation

