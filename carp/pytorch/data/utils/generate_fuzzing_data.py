import csv
import numpy as np
import openai

key = ""
with open ("api_key.txt") as f:
    key = f.read()

# nice try ;)
openai.api_key = key
openai.api_base = "https://api.goose.ai/v1"

prompt = "You are an editor of stories. Below is a set of stories and the the criticisms you have written for each manuscript.\n\n"

def read_validation_component(filepath):
    data = list()
    with open(filepath, newline='') as csvfile:   
        reader = csv.reader(csvfile,delimiter=",", quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            data.append(row[1])
    return data

stories = read_validation_component("val_stories.csv")
crits = read_validation_component("val_crits.csv")

# Filter out stories that are below seven words long.
stories_crits =  list(list(filter(lambda x: (len(x[0].split()) > 7), zip(stories, crits))))
stories_crits = [[ story for story, _ in stories_crits ],
       [ crit for _, crit in stories_crits ]]
stories = stories_crits[0]
crits = stories_crits[1]

#print(stories)
examples_n = 5
indices = np.random.choice(len(stories), examples_n, replace=False)
for i in range(1, examples_n+1, 1):
    idx = indices[i-1]
    story_example = str(i) + ". Story: " + stories[idx] + "\nCriticism: " + crits[idx] + "\n\n"
    prompt += story_example

input_story_idx = max(indices)+1 if max(indices) < len(stories) else min(indices) - 1
input_story = stories[input_story_idx]
#print(crits[30])
prompt += str(examples_n + 1) +". Story: " + input_story + "\nCriticism:"
#print(prompt)
# Create a completion, return results streaming as they are generated. Run with `python3 -u` to ensure unbuffered output.
completion = openai.Completion.create(
  engine="gpt-neo-20b",
  prompt=prompt,
  max_tokens=40,
  typical_p=0.5,
  logit_bias={"50256": -100},
  stream=True)
print(prompt)
# Print each token as it is returned
for c in completion:
  print (c.choices[0].text, end = '')

print("")