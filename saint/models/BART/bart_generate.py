import torch

from tqdm import tqdm
from saint.utils.metric import (
    compute_f1,
    compute_rouge,
    compute_bleu_score,
    compute_exact_match
)


class BartAlignmentGenerator:

    def __init__(self, model):
        self.model = model
        model.model.eval()
        model.model.cuda()

    def _tokenization(self, sentence):
        tokenized = self.model.tokenizer.encode_plus(
            sentence, return_tensors="pt")
        tokenized.to('cuda')

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        return input_ids, attention_mask

    def generate(self, test_data):
        generation_output = []
        with torch.no_grad():
            for _, test_example in tqdm(enumerate(test_data)):
                prompt = test_example['prompt']
                target = test_example["output"]
                input_ids, attention_mask = self._tokenization(prompt)

                beam_outputs = self.model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=50,
                    num_beams=10,
                    top_k=10,
                    do_sample=True,
                    early_stopping=True,
                    num_return_sequences=5,
                    no_repeat_ngram_size=2
                )

                for beam_output in beam_outputs:
                    prediction = self.model.tokenizer.decode(
                        beam_output,
                        skip_special_tokens=True,
                        clean_up__tokenization_spaces=True)
                    f1_score = compute_f1(prediction, target)
                    em_score = compute_exact_match(prediction, target)
                    rouge_scores = compute_rouge(prediction, target)
                    bleu_score = compute_bleu_score(prediction, target)

                    result = {
                        "prompt": prompt,
                        "prediction": prediction,
                        "truth": target,
                        "F1": f1_score,
                        "EM": em_score,
                        "BLEU-4": bleu_score,
                        "ROUGEL": rouge_scores['rougeL'][1],
                    }
                    generation_output.append(result)
                    if rouge_scores['rougeL'][1] == 1:
                        break
        return generation_output
