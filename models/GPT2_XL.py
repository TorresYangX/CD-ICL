from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re


class GPT2_XL:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")

    def generate_text(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        max_input_length = 1024 - 10
        if input_ids.size(1) > max_input_length:
            input_ids = input_ids[:, -max_input_length:]
        output = self.model.generate(input_ids, max_length=len(input_ids[0]) + 2, do_sample=True, pad_token_id=self.tokenizer.eos_token_id)
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        last_answer_index = generated_text.rfind("Answer:")
        if last_answer_index != -1:
            rough_answer = generated_text[last_answer_index + len("Answer:"):].strip()
            if rough_answer:
                match = re.search(r'[^A-D]*([A-D])[^A-D]*', rough_answer, re.IGNORECASE)
                if match:
                    answer = match.group(1).upper()
                else:
                    answer = None  
            else:
                answer = None  
        else:
            answer = None
        return answer