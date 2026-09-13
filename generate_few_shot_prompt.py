"""
generate_few_shot_prompt.py

Builds the few-shot prompt used to teach Gemini the relationship between
California housing attributes and median_house_value.

Per the original assignment: randomly selects 60 to 70 rows to use as
few-shot examples (the count itself is randomized on each run; which
specific rows are drawn is reproducible via random_state).
  - few_shot_df = df.sample(n=<random 60-70>, random_state=42)
  - create_few_shot_prompt(examples)
  - few_shot_prompt = create_few_shot_prompt(few_shot_df)
"""

import random

import pandas as pd

CSV_PATH = "california_housing_train.csv"

df = pd.read_csv(CSV_PATH)

# Randomly select 60 to 70 rows from the dataset
NUM_FEW_SHOT_EXAMPLES = random.randint(60, 70)

few_shot_df = df.sample(
    n=NUM_FEW_SHOT_EXAMPLES,
    random_state=42
)


def create_few_shot_prompt(examples):

    prompt = """
You are a home price prediction assistant.

Your task is to predict the median house value
using California housing data.

Below are examples containing housing attributes
and their actual median house values.

Use these examples as few-shot examples to learn
the relationship between the house attributes and price.

FEW-SHOT EXAMPLES:
"""

    for _, row in examples.iterrows():

        prompt += f"""

Example:
longitude: {row['longitude']}
latitude: {row['latitude']}
housing_median_age: {row['housing_median_age']}
total_rooms: {row['total_rooms']}
total_bedrooms: {row['total_bedrooms']}
population: {row['population']}
households: {row['households']}
median_income: {row['median_income']}

Actual median_house_value: {row['median_house_value']}
"""

    return prompt


few_shot_prompt = create_few_shot_prompt(few_shot_df)


if __name__ == "__main__":
    print(f"Selected {len(few_shot_df)} few-shot examples from {CSV_PATH}")
    print(few_shot_prompt[:5000])
