import re
import math
import argparse
import logging
from typing import List
from openai import OpenAI
from transformers import AutoTokenizer
import os

logger = logging.getLogger(__name__)

class PRM:
    def __init__(self, reward_client, reward_model_name: str, tokenizer_path: str):
        self.reward_client = reward_client
        self.reward_model_name = reward_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._validate_token_length("Right")
        self._validate_token_length("Wrong")
        self.right_token_id = self.tokenizer.encode("Right", add_special_tokens=False)[-1]
        self.wrong_token_id = self.tokenizer.encode("Wrong", add_special_tokens=False)[-1]

        self.right_token_str = self.tokenizer.convert_ids_to_tokens([self.right_token_id])[0]
        self.wrong_token_str = self.tokenizer.convert_ids_to_tokens([self.wrong_token_id])[0]
    
    def _validate_token_length(self, word: str):
        tokens = self.tokenizer.tokenize(word)
        if len(tokens) > 1:
            raise ValueError(f"Word '{word}' is split into multiple tokens: {tokens}. This implementation only supports single token labels.")
    
    def get_reward_score(
        self,
        question: str,
        previous_steps: List[str],
        now_step: str,
        system_prompt: str,
    ) -> float:
        messages = [
            {"role": "user", "content": system_prompt + "\n\n Question: " + question}
        ]
        if len(previous_steps) > 0:
            prev_steps_str = "\n\n".join(previous_steps)
            messages.append({"role": "user", "content": f"{prev_steps_str}\n\nCurrent Step: {now_step}"})
        else:
            messages.append({"role": "user", "content": f"Current Step: {now_step}"})
        response = self.reward_client.chat.completions.create(
            model=self.reward_model_name,
            messages=messages,
            temperature=0.6,
            max_tokens=4096,
            logprobs=True,
            top_logprobs=10, 
        )
        
        choice = response.choices[0]
        generated_text = choice.message.content

        bracket_match = re.search(r'\[(Right|Wrong)\]', generated_text)
        if not bracket_match:
            bracket_match = re.search(r'\\boxed{(Right|Wrong)}', generated_text) or \
                        re.search(r'<(Right|Wrong)>', generated_text) or \
                        re.search(r'\b(Right|Wrong)\b', generated_text)
            
            if not bracket_match:
                logger.warning("Cannot find the label in the generated text.")
                return 0.5
        
        decision = bracket_match.group(1)           
        target_token = decision
        target_found = False

        for i in reversed(range(len(choice.logprobs.content))):
            top_logprobs = choice.logprobs.content[i].top_logprobs
            gen_tokens = [item.token for item in top_logprobs]
            gen_logprobs = [item.logprob for item in top_logprobs]

            if self.right_token_str in gen_tokens and self.wrong_token_str in gen_tokens:
                pos_index = gen_tokens.index(self.right_token_str)
                pos_logprob = gen_logprobs[pos_index]
                pos_prob = math.exp(pos_logprob)

                neg_index = gen_tokens.index(self.wrong_token_str)
                neg_logprob = gen_logprobs[neg_index]
                neg_prob = math.exp(neg_logprob)

            else:
                if self.right_token_str in gen_tokens:
                    pos_index = gen_tokens.index(self.right_token_str)
                    pos_logprob = gen_logprobs[pos_index]
                    pos_prob = math.exp(pos_logprob)
                    neg_logprob = min(gen_logprobs)
                    neg_prob = math.exp(neg_logprob)

                elif self.wrong_token_str in gen_tokens:
                    neg_index = gen_tokens.index(self.wrong_token_str)
                    neg_logprob = gen_logprobs[neg_index]
                    neg_prob = math.exp(neg_logprob)
                    pos_logprob = min(gen_logprobs)
                    pos_prob = math.exp(pos_logprob)
                else:
                    continue
            prob = pos_prob / (pos_prob + neg_prob) if (pos_prob + neg_prob) > 0 else 0.5
            reward_score = prob
            target_found = True
            return reward_score

        if not target_found:
            logger.warning(f"Cannot find the label {target_token} in the generated text.")
            return 0.5

def main():
    parser = argparse.ArgumentParser(description="Reward Model Scorer")
    parser.add_argument("--reward_api_base", type=str, required=True, help="Reward Model API Base URL")
    parser.add_argument("--reward_api_key", type=str, default="EMPTY", help="Reward Model API Key")
    parser.add_argument("--reward_model_name", type=str, required=True, help="Reward Model Name")
    parser.add_argument("--reward_tokenizer_path", type=str, required=True, help="Reward Model Tokenizer Path")    
    args = parser.parse_args()
    reward_model_name = args.reward_model_name
    reward_tokenizer_path = args.reward_tokenizer_path
    
    reward_client = OpenAI(
        base_url=args.reward_api_base,
        api_key=args.reward_api_key,
    )
    
    scorer = PRM(
        reward_client=reward_client,
        reward_model_name=reward_model_name,
        tokenizer_path=reward_tokenizer_path
    )
    system_prompt = """Your task is to analyse and critique the reasoning step below of a math problem step-by-step.
    Please follow the instructions and format below to respond:

    1. Judge the logical and calculation correctness of the step by yourself.
    2. Provide the judgement conclusion and the label as "Right" or "Wrong" in '<output>\\boxed{}</output>'.
    3. Use the following format to respond:
    <judge>
    [details reasoning]
    </judge>
    <output>
    According to the conclusion of the judgement, the label is: \\boxed{}
    </output>

    Here is the step:
    """
    question = "Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?"
    steps = [
      "To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.",
      "On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, (1/3 \\times 18 = 6) flamingos are taken back. So, they have (18 - 6 = 12) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has (12 + 6 = 18) pink flamingos and 6 white flamingos.",
      "On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By the end of Sunday morning, Sue has (18 + 18 = 36) pink flamingos and still 6 white flamingos.",
      "To find the difference, subtract the number of white flamingos from the number of pink flamingos: (36 - 6 = 30). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is (\\boxed{30})."
    ]
    previous_steps = []
    for step in steps:
        score = scorer.get_reward_score(
            question=question,
            previous_steps=previous_steps,
            now_step=step,
            system_prompt=system_prompt
        )
        previous_steps.append(step)
        logger.info(f"Reward Score: {score:.4f}")


if __name__ == "__main__":
    main()