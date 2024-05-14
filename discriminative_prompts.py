ZERO_PROMPT = """Given this reasoning path, do you think this is a valid path to answer the question? If yes please answer "YES", otherwise please answer "NO".

Question:
{question}

Reasoning path:
{path}
"""

ZERO_COT_PROMPT = """Given this reasoning path, do you think this is a valid path to answer the question? If yes please answer "YES", otherwise please answer "NO". Let's think it step by step.

Question:
{question}

Reasoning path:
{path}
"""


FEWSHOT_PROMPT = """
Given this reasoning path, do you think this is a valid path to answer the question? If yes please answer "YES", otherwise please answer "NO". Let's think it step by step. Here are some examples:

## Input:
Question: 
What type of government is used in the country with Northern District?

Reasoning Paths: 
Northern District -> location.administrative_division.first_level_division_of -> Israel -> government.form_of_government.countries -> Parliamentary system

## Output:
YES

## Input:
Question:
Where is the home stadium of the team who won the 1946 World Series championship?

Reasoning Paths:
1946 World Series -> sports.sports_team.championships -> St. Louis Cardinals -> sports.sports_team.arena_stadium -> Roger Dean Stadium

## Output:
NO

## Input:
Question:
Who stared in the TV program that has the theme song Barney Is a Dinosaur?

Reasoning Paths:
Barney Is a Dinosaur -> tv.tv_program.theme_song -> Barney & Friends -> tv.regular_tv_appearance.character -> Barney

## Output:
NO

## Input:
Question:
{question}

Reasoning path:
{path}

## Output:
"""

FEWSHOT_COT_PROMPT = """
Given this reasoning path, do you think this is a valid path to answer the question? If yes please answer "YES", otherwise please answer "NO". Let's think it step by step. Here are some examples:

## Input:
Question: 
What type of government is used in the country with Northern District?

Reasoning Paths: 
Northern District -> location.administrative_division.first_level_division_of -> Israel -> government.form_of_government.countries -> Parliamentary system

## Output:
This reasoning path indicates that:
1. "Northern District" is a location within some country.
2. The reasoning path mentions "Northern District -> location.administrative_division.first_level_division_of -> Israel," indicating that the Northern District is part of Israel.
3. It further states "Israel -> government.form_of_government.countries," suggesting that Israel's form of government is being discussed.
4. The last part of the reasoning path indicates that Israel has a "Parliamentary system."

Therefore, based on the provided reasoning paths, it can be concluded that the type of government used in the country with the Northern District (Israel) is a Parliamentary system. The answer is "YES"

## Input:
Question:
Where is the home stadium of the team who won the 1946 World Series championship?

Reasoning Paths:
1946 World Series -> sports.sports_team.championships -> St. Louis Cardinals -> sports.sports_team.arena_stadium -> Roger Dean Stadium

## Output:
This reasoning path indicates that:
1. St. Louis Cardinals as the team that won the 1946 World Series
2. Roger Dean Stadium is the stadium associated with the St. Louis Cardinals.

However, the home stadium of St. Louis Cardinals is Busch Stadium, not Roger Dean Stadium. Therefore, the answer is "NO"

## Input:
Question:
Who stared in the TV program that has the theme song Barney Is a Dinosaur?

Reasoning Paths:
Barney Is a Dinosaur -> tv.tv_program.theme_song -> Barney & Friends -> tv.regular_tv_appearance.character -> Barney

## Output:
This reasoning path indicates that:
1. Barney Is a Dinosaur is the theme song of Barney & Friends.
2. The main character and star of "Barney & Friends" is Barney the purple dinosaur.

The main character is not the actual actor who plays the character. Therefore, the answer is "NO"

## Input:
Question:
{question}

Reasoning path:
{path}

## Output:
"""
