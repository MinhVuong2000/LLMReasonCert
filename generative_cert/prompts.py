FEWSHOT_COT_ONLY = """1. <step1>
2. <step2>
...
So the answer is (<answer>).
Make sure that the answer uses the above format and answers the question step by step.

Q: when Lou Seal is the mascot for the team that last won the World Series?
A: Let's work this out in a step by step way to be sure we have the right answer.
1. Lou Seal is the mascot for the San Francisco Giants.
2. The San Francisco Giants are associated with the sports championship event, the 2014 World Series.
So the answer is (2014 World Series).

Q: What nation has an army or more than 713480 people and borders the country of Bolivia?
A: Let's work this out in a step by step way to be sure we have the right answer.
1. Bolivia is a landlocked country located in South America.
2. Bolivia shares its borders with several countries, including Argentina, Brazil, Chile, Paraguay, and Peru.
So the answer is (Brazil).

Q: What movie was displayed at the 2012 Refugee Film Festival and had Angelia Jolie directing it?
A: Let's work this out in a step by step way to be sure we have the right answer.
1. Angelia Jolie whose first major film as a director which named “In the Land of Blood and Honey”.
2. “In the Land of Blood and Honey” was shown at the 2012 Refugee Film Festival.
So the answer is (In the Land of Blood and Honey).

Q: How many Mary Mary sisters?
A: Let's work this out in a step by step way to be sure we have the right answer.
1. Mary Mary is a group which has a member named Tina Campbell
2. Mary Mary is a group which has a member named Erica Campbell
So the answer is (Erica Campbell, Tina Campbell).

Q: Which languages are used in the location that the breed Egyptian Mau started in?
A: Let's work this out in a step by step way to be sure we have the right answer.
1. The Egyptian Mau is a breed of domestic cat that is believed to have originated in Egypt.
2. In Egypt, the primary language spoken is Arabic, besides Domari or Nobiin.
So the answer is (Arabic, Domari, Nobiin).

Q: {question}
A: Let's work this out in a step by step way to be sure we have the right answer."""


FEWSHOT_COT_HINT = """Relation path is a sequence relation that describes each step of the reasoning process. You first give a relation path as a HINT, then reason the answer step-by-step based on it.
HINT:
1. <step1>
2. <step2>
...
So the answer is (<answer>).
Make sure that the answer uses the above format and answers the question step by step.

Q: when Lou Seal is the mascot for the team that last won the World Series?
A: Let's work this out in a step by step way to be sure we have the right answer.
HINT: sports.sports_team.team_mascot -> sports.sports_team.championships
1. Lou Seal is the mascot for the San Francisco Giants.
2. The San Francisco Giants are associated with the sports championship event, the 2014 World Series.
So the answer is (2014 World Series).

Q: What nation has an army or more than 713480 people and borders the country of Bolivia?
A: Let's work this out in a step by step way to be sure we have the right answer.
HINT: geography.river.basin_countries -> location.location.partially_contains
1. Bolivia is a landlocked country located in South America.
2. Bolivia shares its borders with several countries, including Argentina, Brazil, Chile, Paraguay, and Peru.
So the answer is (Brazil).

Q: What movie was displayed at the 2012 Refugee Film Festival and had Angelia Jolie directing it?
A: Let's work this out in a step by step way to be sure we have the right answer.
HINT: film.director.film -> film.film_regional_release_date.film_regional_debut_venue
1. Angelia Jolie whose first major film as a director which named “In the Land of Blood and Honey”.
2. “In the Land of Blood and Honey” was shown at the 2012 Refugee Film Festival.
So the answer is (In the Land of Blood and Honey).

Q: How many Mary Mary sisters?
A: Let's work this out in a step by step way to be sure we have the right answer.
HINT: music.group_membership.member -> music.group_membership.member
1. Mary Mary is a group which has a member named Tina Campbell
2. Mary Mary is a group which has a member named Erica Campbell
So the answer is (Erica Campbell, Tina Campbell).

Q: Which languages are used in the location that the breed Egyptian Mau started in?
A: Let's work this out in a step by step way to be sure we have the right answer.
HINT: biology.breed_origin.breeds_originating_here -> location.country.languages_spoken
1. The Egyptian Mau is a breed of domestic cat that is believed to have originated in Egypt.
2. In Egypt, the primary language spoken is Arabic, besides Domari or Nobiin.
So the answer is (Arabic, Domari, Nobiin).

Q: {question}
A: Let's work this out in a step by step way to be sure we have the right answer."""


FEWSHOT_COT_HINT_GROUND = """Relation path is a sequence relation that describes each step of the reasoning process. I will give you a relation path as a hint. Please reason the answer step-by-step based on it.
1. <step1>
2. <step2>
...
So the answer is (<answer>).
Make sure that the answer uses the above format and answers the question step by step.

Q: when Lou Seal is the mascot for the team that last won the World Series?
A: Let's work this out in a step by step way to be sure we have the right answer.
HINT: sports.sports_team.team_mascot -> sports.sports_team.championships
1. Lou Seal is the mascot for the San Francisco Giants.
2. The San Francisco Giants are associated with the sports championship event, the 2014 World Series.
So the answer is (2014 World Series).

Q: What nation has an army or more than 713480 people and borders the country of Bolivia?
A: Let's work this out in a step by step way to be sure we have the right answer.
HINT: geography.river.basin_countries -> location.location.partially_contains
1. Bolivia is a landlocked country located in South America.
2. Bolivia shares its borders with several countries, including Argentina, Brazil, Chile, Paraguay, and Peru.
So the answer is (Brazil).

Q: What movie was displayed at the 2012 Refugee Film Festival and had Angelia Jolie directing it?
A: Let's work this out in a step by step way to be sure we have the right answer.
HINT: film.director.film -> film.film_regional_release_date.film_regional_debut_venue
1. Angelia Jolie whose first major film as a director which named “In the Land of Blood and Honey”.
2. “In the Land of Blood and Honey” was shown at the 2012 Refugee Film Festival.
So the answer is (In the Land of Blood and Honey).

Q: How many Mary Mary sisters?
A: Let's work this out in a step by step way to be sure we have the right answer.
HINT: music.group_membership.member -> music.group_membership.member
1. Mary Mary is a group which has a member named Tina Campbell
2. Mary Mary is a group which has a member named Erica Campbell
So the answer is (Erica Campbell, Tina Campbell).

Q: Which languages are used in the location that the breed Egyptian Mau started in?
A: Let's work this out in a step by step way to be sure we have the right answer.
HINT: biology.breed_origin.breeds_originating_here -> location.country.languages_spoken
1. The Egyptian Mau is a breed of domestic cat that is believed to have originated in Egypt.
2. In Egypt, the primary language spoken is Arabic, besides Domari or Nobiin.
So the answer is (Arabic, Domari, Nobiin).

Q: {question}
A: Let's work this out in a step by step way to be sure we have the right answer.
HINT: {hint}."""
