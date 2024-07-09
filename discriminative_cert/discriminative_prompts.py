ZERO_PROMPT = """A reasoning path is a sequence of triples that can be used to derive the answer of given question. A valid reasoning path should follow these rules:
1. No factual errors: Each triple in the reasoning path should adhere to real-world factual knowledge.
2. Coherence: The tail entity of the previous triple should be the head entity of the next triple.
3. Correctness: The reasoning path should lead to the correct answer at the last tail entity.

Given this reasoning path, do you think this is a valid path to derive the answer of given question? If yes please answer "YES", otherwise please answer "NO".

Question:
{question}

Answer:
{answer}

Reasoning path:
{path}
"""

ZERO_COT_PROMPT = """A reasoning path is a sequence of triples that can be used to derive the answer of given question. A valid reasoning path should follow these rules:
1. No factual errors: Each triple in the reasoning path should adhere to real-world factual knowledge.
2. Coherence: The tail entity of the previous triple should be the head entity of the next triple.
3. Correctness: The reasoning path should lead to the correct answer at the last tail entity.

Given this reasoning path, do you think this is a valid path to derive the answer of given question? If yes please answer "YES", otherwise please answer "NO". Let's think it step by step.

Question:
{question}

Answer:
{answer}

Reasoning path:
{path}
"""


FEWSHOT_PROMPT = """A reasoning path is a sequence of triples that can be used to derive the answer of given question. A valid reasoning path should follow these rules:
1. No factual errors: Each triple in the reasoning path should adhere to real-world factual knowledge.
2. Coherence: The tail entity of the previous triple should be the head entity of the next triple.
3. Correctness: The reasoning path should lead to the correct answer at the last tail entity.

Given this reasoning path, do you think this is a valid path to derive the answer of given question? If yes please answer "YES", otherwise please answer "NO". Here are some examples:

## Input:
Question: 
What type of government is used in the country with Northern District?

Answer:
Parliamentary system

Reasoning Paths: 
Step 1: Northern District -> location.administrative_division.first_level_division_of -> Israel
Step 2: Israel -> government.form_of_government.countries -> Parliamentary system

## Output:
YES

## Input:
Question:
Where is the home stadium of the team who won the 1946 World Series championship?

Answer:
Busch Stadium

Reasoning Paths:
Step 1: 1946 World Series -> sports.sports_team.championships -> St. Louis Cardinals
Step 2: St. Louis Cardinals -> sports.sports_team.arena_stadium -> Roger Dean Stadium

## Output:
NO

## Input:
Question:
In which American Southern City did the ""Downs"" composer die?

Answer:
New Orleans

Reasoning Paths:
Step 1: Alex Chilton -> people.deceased_person.place_of_death -> New Orleans
Step 2: Downs -> music.composer.compositions -> Alex Chilton

## Output:
NO

## Input:
Question:
Where was the main artist featured in the Rihanna: Live in Concert Tour raised?

Answer:
Saint Michael Parish

Reasoning Paths:
Step 1: Rihanna: Live in Concert Tour -> music.concert_tour.artist -> Rihanna
Step 2: Rihanna -> music.track_contribution.role -> Vocals

## Output:
NO

## Input:
Question:
{question}

Answer:
{answer}

Reasoning path:
{path}

## Output:
"""

FEWSHOT_COT_PROMPT = """A reasoning path is a sequence of triples that can be used to derive the answer of given question. A valid reasoning path should follow these rules:
1. No factual errors: Each triple in the reasoning path should adhere to real-world factual knowledge.
2. Coherence: The tail entity of the previous triple should be the head entity of the next triple.
3. Correctness: The reasoning path should lead to the correct answer at the last tail entity.

Given this reasoning path, do you think this is a valid path to derive the answer of given question? If yes please answer "YES", otherwise please answer "NO". Here are some examples:

## Input:
Question: 
What type of government is used in the country with Northern District?
Answer:
Parliamentary system

Reasoning Paths: 
Step 1: Northern District -> location.administrative_division.first_level_division_of -> Israel
Step 2: Israel -> government.form_of_government.countries -> Parliamentary system

## Output:
This reasoning path indicates that:
1. "Northern District" is a location within some country.
2. The reasoning path mentions "Northern District -> location.administrative_division.first_level_division_of -> Israel," indicating that the Northern District is part of Israel.
3. It further states "Israel -> government.form_of_government.countries," suggesting that Israel's form of government is being discussed.
4. The last part of the reasoning path indicates that Israel has a "Parliamentary system."

Based on the provided reasoning paths, it can be concluded that the type of government used in the country with the Northern District (Israel) is a Parliamentary system. Therefore, the path adhaer the definition of valid reasoning path. The answer is "YES"

## Input:
Question:
Where is the home stadium of the team who won the 1946 World Series championship?

Answer:
Busch Stadium

Reasoning Paths:
Step 1: 1946 World Series -> sports.sports_team.championships -> St. Louis Cardinals
Step 2: St. Louis Cardinals -> sports.sports_team.arena_stadium -> Roger Dean Stadium

## Output:
This reasoning path indicates that:
1. St. Louis Cardinals as the team that won the 1946 World Series
2. Roger Dean Stadium is the stadium associated with the St. Louis Cardinals.

However, the home stadium of St. Louis Cardinals is Busch Stadium, not Roger Dean Stadium. Therefore, this path contains factual errors. The answer is "NO".

## Input:
Question:
In which American Southern City did the ""Downs"" composer die?

Answer:
New Orleans

Reasoning Paths:
Step 1: Alex Chilton -> people.deceased_person.place_of_death -> New Orleans
Step 2: Downs -> music.composer.compositions -> Alex Chilton

## Output:
This reasoning path indicates that:
1. Alex Chilton was dead in New Orleans
2. The composition of Downs is Alex Chilton

Even through the first step of reasoning path leads to the correct answer, which is New Orleans, the reasoning path is not coherent. To answer the question, we need to first the composition of Downs which is Alex Chilton, then we find the death place of Alex Chilton which is New Orleans. Therefore, the answer is "NO".

## Input:
Question:
Where was the main artist featured in the Rihanna: Live in Concert Tour raised?

Answer:
Saint Michael Parish

Reasoning Paths:
Step 1: Rihanna: Live in Concert Tour -> music.concert_tour.artist -> Rihanna
Step 2: Rihanna -> music.track_contribution.role -> Vocals

## Output:
This reasoning path indicates that:
1. The artist of Rihanna: Live in Concert Tour is Rihanna
2. Rihanna is a vocal artist

Even through there are no factual erros and the reasoning path is coherent, the reasoning path does not lead to the correct answer. The question asks for the birth place of the main artist, not the role of the artist. Therefore, the answer is "NO".

## Input:
Question:
{question}

Answer:
{answer}

Reasoning path:
{path}

## Output:
"""

NEG_FEWSHOT_COT_PROMPT = """
Given this reasoning path, do you think this is a valid path to answer the question? If yes please answer "YES", otherwise please answer "NO". Let's think it step by step. Here are some examples:

## Input:
Question: 
What type of government is used in the country with Northern District?

Reasoning Paths: 
Step 1: Northern District -> location.administrative_division.first_level_division_of -> Israel
Step 2: Israel -> government.form_of_government.countries -> Parliamentary system

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
In which American Southern City did the ""Downs"" composer die?

Reasoning Paths:
Downs -> music.composer.compositions -> Judæo-Persian dialects -> people.deceased_person.place_of_death -> New Orleans

## Output:
This reasoning path indicates that:
1. The composer of "Downs" is Judæo-Persian dialects.
2. Judæo-Persian dialects was deaded in New Orleans 

However, the composer of Downs is Alex Chilton, not Judæo-Persian dialects. Therefore, the answer is "NO"

## Input:
Question:
{question}

Reasoning path:
{path}

## Output:
"""

NEG_FEWSHOT_PROMPT = """
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
In which American Southern City did the ""Downs"" composer die?

Reasoning Paths:
Downs -> music.composer.compositions -> Judæo-Persian dialects -> people.deceased_person.place_of_death -> New Orleans

## Output:
NO

## Input:
Question:
{question}

Reasoning path:
{path}

## Output:
"""


NEG_REORDER_FEWSHOT_PROMPT = """
Given this reasoning path, do you think this is a valid path to answer the question? If yes please answer "YES", otherwise please answer "NO". Let's think it step by step. Here are some examples:

## Input:
Question: 
What type of government is used in the country with Northern District?

Reasoning Paths: 
Step 1: Northern District -> location.administrative_division.first_level_division_of -> Israel 
Step 2: Israel -> government.form_of_government.countries -> Parliamentary system

## Output:
YES

## Input:
Question:
Where is the home stadium of the team who won the 1946 World Series championship?

Reasoning Paths:
Step 1: St. Louis Cardinals -> sports.sports_team.arena_stadium -> Busch Stadium
Step 2: 1946 World Series -> sports.sports_team.championships -> St. Louis Cardinals

## Output:
NO

## Input:
Question:
In which American Southern City did the ""Downs"" composer die?

Reasoning Paths:
Step 1: Alex Chilton -> people.deceased_person.place_of_death -> New Orleans
Step 2: Downs -> music.composer.compositions -> Alex Chilton

## Output:
NO

## Input:
Question:
{question}

Reasoning path:
{path}

## Output:
"""
