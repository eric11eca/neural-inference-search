import openai
import wandb
import uuid
import json
import pandas as pd

label_map = {
    "entailment": "absolutely true",
    "contradiction": "absolutely not true",
    "neutral": "may or may not be true"
}



openai.api_key = "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"

run = wandb.init(
    project="causal_scaffold_modeling", 
    name="gpt3_alignment_entailment"
)

def build_demonstration():
    examples = {
        "entailment": [],
        "contradiction": [],
        "neutral": []
    }
    df = pd.read_csv("./snli_train.csv")
    for i, row in df.iterrows():
        premise = row["premise"]
        hypothesis = row["hypothesis"]
        label = row["label"]
        alignments = json.loads(row["alignments"]) 

        align_pairs = ""
        for alignment in alignments:
            align_pairs += f'({alignment["align_p"]["text"]}; {alignment["align_h"]["text"]}), '
        
        prompt = f'Given a context "{premise}" the conclusion "{hypothesis}" is {label_map[label.lower()]}. List all phrases that support this statement.'

        examples[label.lower()].append({"prompt": prompt, "output": align_pairs})
    return examples

prediction_table = wandb.Table(columns=["prompt", "alignment"])


class Example:
    """Stores an input, output pair and formats it to prime the model."""
    def __init__(self, inp, out):
        self.input = inp
        self.output = out
        self.id = uuid.uuid4().hex

    def get_input(self):
        """Returns the input of the example."""
        return self.input

    def get_output(self):
        """Returns the intended output of the example."""
        return self.output

    def get_id(self):
        """Returns the unique ID of the example."""
        return self.id

    def as_dict(self):
        return {
            "input": self.get_input(),
            "output": self.get_output(),
            "id": self.get_id(),
        }


class Prompt:
    """The main class for a user to create a prompt for GPT3"""

    def __init__(self) -> None:
        self.examples = []
    
    def add_example(self, ex):
        """
        Adds an example to the object.
        Example must be an instance of the Example class.
        """
        assert isinstance(ex, Example), "Please create an Example object."
        self.examples.append(ex)

    def delete_example(self, id):
        """Delete example with the specific id."""
        if id in self.examples:
            del self.examples[id]

    def get_example(self, id):
        """Get a single example."""
        return self.examples.get(id, None)

    def get_all_examples(self):
        """Returns all examples as a list of dicts."""
        return {k: v.as_dict() for k, v in self.examples.items()}
    
    def craft_query(self, input):
        """Creates the query for the API request."""
        prompt = ""
        for example in self.examples:
            prompt += f"input: {example.get_input()} \n\n output: {example.get_output()} \n\n"
        prompt += input

        return prompt


if __name__ == '__main__':
    
    examples = build_demonstration()
    gpt_prompt = Prompt()
    for example in examples[:40]:
        demonstration = Example(example["prompt"], example["output"])
        gpt_prompt.add_example(demonstration)

    
    premise = "Ernest Jones is a British jeweler and watchmaker. Established in 1949, its first store was opened in Oxford Street, London. Ernest Jones specialises in diamonds and watches, stocking brands such as Gucci and Emporio Armani. Ernest Jones is part of the Signet Jewelers group."
    hypothesis = "The first Ernest Jones store was opened on the continent of Europe."
    problem = f'input: Given a context "{premise}" the conclusion "{hypothesis}" is {label_map["entailment"]}. List all phrases that support this statement.\n'

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=gpt_prompt.craft_query(problem),
        temperature=0.5,
        max_tokens=128,
        top_p=1.0,
        frequency_penalty=0.1,
        presence_penalty=0.0
    )

    print(response['choices'][0]['text'])
    prediction_table.add_data(problem, response['choices'][0]['text'])
    
    dev = pd.read_table("./data/causal_snli/original/dev.tsv")

    for i, row in dev[77:107].iterrows():
        premise = row["sentence1"]
        hypothesis = row["sentence2"]
        label = row["gold_label"]
        problem = f'input: Given a context "{premise}" the conclusion "{hypothesis}" is {label_map[label]}. List all phrases that support this statement.\n'
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=gpt_prompt.craft_query(problem),
            temperature=0.5,
            max_tokens=128,
            top_p=1.0,
            frequency_penalty=0.1,
            presence_penalty=0.0
        )

        print(response['choices'][0]['text'])

        prediction_table.add_data(problem, response['choices'][0]['text'])

    wandb.log({'gpt3-generation': prediction_table})
    wandb.finish()