import csv
import numpy as np
import openai
from neox_tokenizer import *

bad_word_dict = \
    {"0": -1000, "50256": -1000, "3353": -1000}
# add all line breaks to the bad word dict
for idx in line_break_token_ids:
    bad_word_dict[str(idx)] = -1000

key = ""
with open("../../pytorch/data/utils/api_key.txt") as f:
    key = f.read()

# nice try ;)
openai.api_key = key
openai.api_base = "https://api.goose.ai/v1"

prompt = "You are an editor of stories. Below is a set of stories and the the criticisms you have written for each " \
         "manuscript.\n\n "


def read_dataset_component(filepath):
    data = list()
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            data.append(row[1])
    return data


val_stories = read_dataset_component("../../pytorch/data/utils/val_stories.csv")
val_crits = read_dataset_component("../../pytorch/data/utils/val_crits.csv")

# Filter out stories that are below seven words long.
stories_crits = list(list(filter(lambda x: (len(x[0].split()) > 7), zip(val_stories, val_crits))))
stories_crits = [[story for story, _ in stories_crits],
                 [crit for _, crit in stories_crits]]
val_stories = stories_crits[0]
val_crits = stories_crits[1]

train_stories = read_dataset_component("../../pytorch/data/utils/train_stories.csv")[0:10]
examples_n = 5
indices = np.random.choice(len(val_stories), examples_n, replace=False)
print("[" + ", ".join(list(map(str, indices))) + "]")
indices = [56, 23, 14, 65, 60]

for i in range(1, examples_n + 1, 1):
    idx = indices[i - 1]
    story_example = str(i) + ". Story: " + val_stories[idx] + "\nCriticism: " + val_crits[idx] + "\n\n"
    prompt += story_example

for input_story in train_stories:
    story_prompt = prompt + str(examples_n + 1) + ". Story: " + input_story + "\nCriticism:"
    # Biasing against EOT (0, 50256), and \n
    print(story_prompt)
    import sys
    sys.exit()
    completion = openai.Completion.create(
        engine="gpt-neo-20b",
        prompt=story_prompt,
        max_tokens=40,
        typical_p=0.5,
        logit_bias=bad_word_dict,
        logprobs=30,
        stream=True)
    print(input_story)
    print("\n")

    # Print each token as it is returned
    for c in completion:
        print(c.choices[0].text, end='')

    print("\n=============\n")
