"""
predict_house_value.py

Sends the few-shot prompt (built in generate_few_shot_prompt.py) plus a
new-house prediction request to Gemini 2.5 Flash via LangChain, and prints
the predicted median_house_value.

Preserves the original Colab notebook's prompt-engineering flow:
  few_shot_prompt = create_few_shot_prompt(few_shot_df)
  new_house_prompt = create_new_house_prompt(new_house)
  final_prompt = few_shot_prompt + new_house_prompt
"""

import os
import sys

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from generate_few_shot_prompt import few_shot_prompt, few_shot_df

# Load GOOGLE_API_KEY from .env.local (never hard-code the key)
load_dotenv(".env.local")

if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Copy .env.local.example to .env.local "
        "and add your Google API key."
    )

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# The new house we want a price prediction for
new_house = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 35,
    "total_rooms": 2500,
    "total_bedrooms": 500,
    "population": 900,
    "households": 450,
    "median_income": 5.5
}


def create_new_house_prompt(new_house):

    prompt = f"""

NEW HOUSE TO PREDICT:

longitude: {new_house['longitude']}
latitude: {new_house['latitude']}
housing_median_age: {new_house['housing_median_age']}
total_rooms: {new_house['total_rooms']}
total_bedrooms: {new_house['total_bedrooms']}
population: {new_house['population']}
households: {new_house['households']}
median_income: {new_house['median_income']}

Based on the few-shot examples above, predict the
median house value for this new house.

Return only the predicted price as a number.
"""

    return prompt


def predict(new_house):
    new_house_prompt = create_new_house_prompt(new_house)
    final_prompt = few_shot_prompt + new_house_prompt

    response = model.invoke(final_prompt)
    price = float(response.content[0]["text"])

    print("Predicted Home Price: ${:,.2f}".format(price))
    return price


def predict_from_user_input():
    print("Enter the new house information:")

    house = {
        "longitude": float(input("Longitude: ")),
        "latitude": float(input("Latitude: ")),
        "housing_median_age": float(input("Housing median age: ")),
        "total_rooms": float(input("Total rooms: ")),
        "total_bedrooms": float(input("Total bedrooms: ")),
        "population": float(input("Population: ")),
        "households": float(input("Households: ")),
        "median_income": float(input("Median income: ")),
    }

    return predict(house)


if __name__ == "__main__":
    print(f"Using {len(few_shot_df)} few-shot examples from california_housing_train.csv\n")

    if "--interactive" in sys.argv:
        predict_from_user_input()
    else:
        predict(new_house)
