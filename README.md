# California Housing Price Prediction — Few-Shot Prompt Engineering

A prompt-engineering assignment that predicts California housing prices using **few-shot learning** with **Gemini 3.5 Flash**, instead of a trained ML model.

## What This Project Does

Rather than training a regression model, this project teaches Gemini the relationship between housing attributes and price entirely through the prompt:

1. **60 to 70 random examples** (the exact count is randomized on each run) are sampled from the California Housing dataset, each containing the house's attributes *and* its actual `median_house_value`.
2. Those examples are formatted into a single **few-shot prompt**.
3. A **new house** (with unknown price) is formatted into its own prompt asking for a prediction.
4. The few-shot prompt and the new-house prompt are concatenated into one **final prompt**.
5. The final prompt is sent to **Gemini 2.5 Flash** via **LangChain's Google GenAI integration**.
6. Gemini responds with its predicted `median_house_value`, based purely on the pattern it inferred from the examples.

This is the core idea of few-shot prompting: instead of fine-tuning a model, you demonstrate the task with examples directly inside the prompt.

## Dataset

`california_housing_train.csv` — the classic California Housing dataset, with columns:

`longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value`

## Files

| File | Purpose |
|---|---|
| `generate_few_shot_prompt.py` | Loads the CSV, samples 60 to 70 rows (`df.sample(n=random.randint(60, 70), random_state=42)`), and builds `few_shot_prompt` via `create_few_shot_prompt()` |
| `predict_house_value.py` | Loads the Gemini API key, builds the new-house prompt via `create_new_house_prompt()`, combines it with `few_shot_prompt` into `final_prompt`, and sends it to Gemini 2.5 Flash for a prediction |
| `california_housing_train.csv` | The dataset both scripts read from |
| `.env.local.example` | Template for your API key — copy to `.env.local` |

## Installation

```bash
pip install pandas langchain-google-genai python-dotenv
```

## API Key Setup

1. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Copy the example env file:
   ```bash
   cp .env.local.example .env.local
   ```
3. Open `.env.local` and paste your key:
   ```
   GOOGLE_API_KEY=your_actual_key_here
   ```

`.env.local` is git-ignored, so your key is never committed.

## How to Run

**1. Build and preview the few-shot prompt on its own:**

```bash
python generate_few_shot_prompt.py
```

This prints how many examples were selected (a random count between 60 and 70) and a preview of the generated prompt.

**2. Run a prediction:**

```bash
python predict_house_value.py
```

This uses a built-in example house and prints Gemini's predicted price.

To instead type in your own house attributes interactively:

```bash
python predict_house_value.py --interactive
```

## Example Prediction Output

```
Using 64 few-shot examples from california_housing_train.csv

Predicted Home Price: $342,150.00
```

(Exact numbers vary between runs, since this is a language model's prediction, not a deterministic formula.)

## How the Prompt Engineering Works

```python
few_shot_prompt = create_few_shot_prompt(few_shot_df)      # 60-70 labeled examples
new_house_prompt = create_new_house_prompt(new_house)       # the house to predict
final_prompt = few_shot_prompt + new_house_prompt            # combined prompt sent to Gemini

response = model.invoke(final_prompt)
price = float(response.content[0]["text"])
```

`few_shot_df` contains a randomly chosen number of rows between 60 and 70 on each run, via `df.sample(n=random.randint(60, 70), random_state=42)`.
