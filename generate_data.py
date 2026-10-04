"""
Generates a synthetic training dataset by combining templates with
word pools per emotion. This is clearly synthetic data, built for
this rebuild - not scraped or collected from real users.
"""
import random
random.seed(42)

happiness_templates = [
    "I am so {adj} about {thing}",
    "This {thing} makes me feel {adj}",
    "I feel {adj} and grateful for {thing}",
    "{thing} was absolutely wonderful, I feel {adj}",
    "I can't stop smiling because of {thing}",
    "What a {adj} day, {thing} went so well",
    "I am {adj} that {thing} finally happened",
]
happiness_adj = ["happy", "thrilled", "overjoyed", "delighted", "excited", "ecstatic", "proud", "joyful"]
happiness_thing = ["the party", "my exam results", "the trip", "this news", "the promotion",
                    "the surprise", "my new job", "the concert", "the weekend plans", "this gift",
                    "the win", "spending time with family", "this achievement", "the celebration"]

sadness_templates = [
    "I feel so {adj} about {thing}",
    "{thing} left me feeling {adj}",
    "I am {adj} because of {thing}",
    "It hurts to think about {thing}, I feel {adj}",
    "I can't stop feeling {adj} after {thing}",
    "{thing} made me really {adj}",
]
sadness_adj = ["sad", "heartbroken", "lonely", "depressed", "devastated", "down", "miserable", "hopeless"]
sadness_thing = ["the breakup", "failing the exam", "losing my pet", "moving away", "the argument",
                 "missing my friends", "the rejection", "the loss", "being ignored", "the bad news",
                 "the disappointment", "the farewell", "this situation", "being alone"]

anger_templates = [
    "I am so {adj} about {thing}",
    "{thing} makes me absolutely {adj}",
    "I feel {adj} because of {thing}",
    "This is {adj}, {thing} is unacceptable",
    "I can't believe {thing}, I am {adj}",
    "{thing} really made me {adj}",
]
anger_adj = ["angry", "furious", "livid", "outraged", "frustrated", "irritated", "enraged", "mad"]
anger_thing = ["the delay", "being lied to", "the rude comment", "the broken promise", "the unfair decision",
               "the disrespect", "being ignored again", "the constant mistakes", "the bad service",
               "the last-minute cancellation", "being blamed unfairly", "the dishonesty"]

neutral_templates = [
    "The {thing} is scheduled for {time}",
    "I need to {action} before {time}",
    "Please send me the {thing} by {time}",
    "The {thing} is located near the {place}",
    "We will discuss the {thing} during {time}",
    "The {thing} contains information about {place}",
]
neutral_thing = ["meeting", "report", "document", "class", "train", "package", "presentation", "assignment", "invoice", "schedule"]
neutral_time = ["tomorrow morning", "next week", "3 PM today", "Monday", "the end of the month", "this evening"]
neutral_action = ["submit the form", "buy groceries", "check the email", "update the file", "call the office", "review the notes"]
neutral_place = ["main office", "library", "second floor", "city center", "campus", "station"]

def fill(template, **pools):
    result = template
    for key, options in pools.items():
        if "{" + key + "}" in result:
            result = result.replace("{" + key + "}", random.choice(options))
    return result

def generate(templates, n, **pools):
    sentences = set()
    while len(sentences) < n:
        t = random.choice(templates)
        s = fill(t, **pools)
        sentences.add(s)
    return list(sentences)

happiness = [(s, "happiness") for s in generate(happiness_templates, 40, adj=happiness_adj, thing=happiness_thing)]
sadness = [(s, "sadness") for s in generate(sadness_templates, 40, adj=sadness_adj, thing=sadness_thing)]
anger = [(s, "anger") for s in generate(anger_templates, 40, adj=anger_adj, thing=anger_thing)]
neutral = [(s, "neutral") for s in generate(neutral_templates, 40, thing=neutral_thing, time=neutral_time, action=neutral_action, place=neutral_place)]

all_data = happiness + sadness + anger + neutral
random.shuffle(all_data)

with open("data.py", "w") as f:
    f.write('"""\n')
    f.write("Training data for the Emotion Detection Chatbot.\n\n")
    f.write("Synthetically generated for this rebuild using templates + word\n")
    f.write("pools per emotion, since the original project's data no longer\n")
    f.write("exists.\n")
    f.write('"""\n\n')
    f.write("TRAINING_DATA = [\n")
    for text, label in all_data:
        f.write(f'    ({text!r}, {label!r}),\n')
    f.write("]\n")

print(f"Generated {len(all_data)} examples: "
      f"{len(happiness)} happiness, {len(sadness)} sadness, "
      f"{len(anger)} anger, {len(neutral)} neutral")