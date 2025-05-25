import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Union
import random
import torch
import re
import json
import os

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

seed_question = "What is 1 + 1?"
seed_answer = "2" 

proposing_prompt = """
You will be given {num_questions} math question and an answer. You will need to generate a new question which is of a higher complexity than the questions provided, but solvable. Focus on questions that are solvable by reasoning, not just memorization. Explore new topics and concepts. Don't stick to the questions provided. Geometry, algebra, calculus, etc. are all fair game.
You will wrap your response in <question> and </question> tags, reasoning in <reasoning> and </reasoning> tags, and its answer in <answer> and </answer> tags. The answer should be a number. There should be only a single number in the answer. Only digits are allowed in the answer. No other text is allowed in the answer.

Here are the original questions and answers:
{questions_and_answers}
"""

solving_prompt = """ 
You will be given a math question. You will need to solve the question using step by step reasoning. You will wrap your reasoning in <reasoning> and </reasoning> tags, and your answer in <answer> and </answer> tags. The answer should be a number. There should be only a single number in the answer. Only digits are allowed in the answer. No other text is allowed in the answer.

Here is the question:
<question>
{question}
</question>
"""

question_and_answer_bank = [{
    "question": seed_question,
    "answer": seed_answer
}]
log_history = []

model = None 
ref_model = None 
tokenizer = None 
optimizer = None 
max_new_tokens = 8192
seed_questions_path = "seed_questions.json"

def generate_seed_questions(num_questions: int):
    global question_and_answer_bank
    logger.info(f"Generating {num_questions} seed questions.")
    if os.path.exists(seed_questions_path):
        with open(seed_questions_path, "r") as f:
            question_and_answer_bank = json.load(f)
        logger.info(f"Loaded existing seed questions from {seed_questions_path}.")
        return 
    else:
        question_and_answer_bank = []
        logger.info("No existing seed questions found. Starting fresh.")
    for i in range(num_questions):
        num_refs = min(3, len(question_and_answer_bank))
        incontext_examples = random.sample(question_and_answer_bank, num_refs)
        prompt = proposing_prompt.format(num_questions=num_refs, questions_and_answers=incontext_examples)
        message = [
            {"role": "user", "content": prompt},
        ]
        message = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        prompt_len = message.shape[1]
        response = model.generate(message, max_new_tokens=max_new_tokens)
        response_text = tokenizer.decode(response[0][prompt_len:], skip_special_tokens=False)
        logger.debug(f"Generated seed question raw response: {response_text}")
        try:
            question = response_text.split("<question>")[1].split("</question>")[0].strip()
            answer = response_text.split("<answer>")[1].split("</answer>")[0].strip()
            logger.info(f"Parsed seed question: {question}, answer: {answer}")
        except Exception as e:
            logger.error(f"Failed to parse question and answer from: {response_text}")
            raise Exception("Failed to parse question and answer") from e
        
        question_and_answer_bank.append({
            "question": question,
            "answer": answer
        })
    with open(seed_questions_path, "w") as f:
        json.dump(question_and_answer_bank, f)
    logger.info(f"Saved {len(question_and_answer_bank)} seed questions to {seed_questions_path}.")
    model.train()
def solve_question(question: str, training: bool = True): 
    logger.info(f"Solving question: {question}")
    prompt = solving_prompt.format(question=question)
    message = [
        {"role": "user", "content": prompt},
    ]
    message = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    prompt_len = message.shape[1]
    response = model.generate(message, max_new_tokens=max_new_tokens)
    response_text = tokenizer.decode(response[0][prompt_len:], skip_special_tokens=False)

    logger.debug(f"Solver raw response: {response_text}")
    try:
        answer = response_text.split("<answer>")[1].split("</answer>")[0].strip()
        logger.info(f"Parsed solver answer: {answer}")
    except Exception as e:
        logger.error(f"Failed to parse answer from: {response_text}")
        raise Exception("Failed to parse answer") from e
    return answer

def propose_question(training: bool = True):
    num_refs = min(3, len(question_and_answer_bank))
    incontext_examples = random.sample(question_and_answer_bank, num_refs)
    incontext_examples_str = "\n".join([f"<question>{qna['question']}</question>\n<answer>{qna['answer']}</answer>" for i, qna in enumerate(incontext_examples)])
    prompt = proposing_prompt.format(num_questions=len(incontext_examples), questions_and_answers=incontext_examples_str)
    message = [
        {"role": "user", "content": prompt},
    ]
    message = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    prompt_len = message.shape[1]
    response = model.generate(message, max_new_tokens=max_new_tokens)
    response_text = tokenizer.decode(response[0][prompt_len:], skip_special_tokens=False)
    logger.debug(f"Proposer raw response: {response_text}")
    return response_text
    
# Compute log probs of generated sequence under both models
def get_log_probs(model, input_ids):
    with torch.no_grad():
        logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected = log_probs.gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return selected.sum(dim=-1)  # total log prob for response

def fit():
    n_epochs = 10
    bsz = 1
    logger.info(f"Starting training for {n_epochs} epochs, batch size {bsz}.")
    for epoch in range(n_epochs):
        logger.info(f"Epoch {epoch+1}/{n_epochs} started.")
        # Proposer Phase 
        proposer_rewards = []
        proposer_model_log_probs_list = []
        proposer_ref_model_log_probs_list = []
        proposer_kl_divs = []
        question_input_ids_list = []
        question_raw_responses = []
        for _ in range(bsz):
            question_raw_response = propose_question()
            proposer_reward = proposer_reward_fn(question_raw_response)
            proposer_rewards.append(proposer_reward)
            question_input_ids = tokenizer.encode(question_raw_response, return_tensors="pt").to(model.device) 
            # Get logits for both models
            model_logits = model(question_input_ids).logits
            with torch.no_grad():
                ref_model_logits = ref_model(question_input_ids).logits
            # Compute log probs
            model_log_probs = torch.nn.functional.log_softmax(model_logits, dim=-1)
            ref_model_log_probs = torch.nn.functional.log_softmax(ref_model_logits, dim=-1)
            # Gather log probs for the actual tokens
            model_selected = model_log_probs.gather(2, question_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            ref_model_selected = ref_model_log_probs.gather(2, question_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            # Sum log probs for total log prob
            model_log_prob_sum = model_selected.sum(dim=-1)
            ref_model_log_prob_sum = ref_model_selected.sum(dim=-1)
            proposer_model_log_probs_list.append(model_log_prob_sum)
            proposer_ref_model_log_probs_list.append(ref_model_log_prob_sum)
            question_input_ids_list.append(question_input_ids)
            question_raw_responses.append(question_raw_response)
            # KL divergence per token (mean over sequence)
            kl = torch.nn.functional.kl_div(
                model_log_probs[:, :-1],  # [batch, seq, vocab]
                ref_model_log_probs[:, :-1].exp(),
                reduction='batchmean',
                log_target=False
            )
            proposer_kl_divs.append(kl)
            logger.info(f"Proposer Reward: {proposer_reward}")
            logger.debug(f"Proposer raw response: {question_raw_response}")

        # Proposer loss and update
        proposer_rewards_tensor = torch.tensor(proposer_rewards, dtype=torch.float32, device=model.device)
        proposer_avg_reward = proposer_rewards_tensor.mean()
        proposer_advantages = proposer_rewards_tensor - proposer_avg_reward

        proposer_model_log_probs_tensor = torch.cat(proposer_model_log_probs_list, dim=0)
        proposer_ref_model_log_probs_tensor = torch.cat(proposer_ref_model_log_probs_list, dim=0)
        proposer_kl_divs_tensor = torch.stack(proposer_kl_divs)

        kl_coeff = 0.1  # You can tune this coefficient
        proposer_loss = -((proposer_model_log_probs_tensor - proposer_ref_model_log_probs_tensor) * proposer_advantages).mean() + kl_coeff * proposer_kl_divs_tensor.mean()
        proposer_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        logger.info(f"Proposer loss: {proposer_loss.item():.4f}")

        # Solver Phase
        solver_rewards = []
        solver_model_log_probs_list = []
        solver_ref_model_log_probs_list = []
        solver_kl_divs = []
        solver_input_ids_list = []
        solver_raw_responses = []
        for i in range(bsz):
            # Try to extract the question and answer from the raw response
            try:
                question_text = question_raw_responses[i].split("<question>")[1].split("</question>")[0]
                original_answer = question_raw_responses[i].split("<answer>")[1].split("</answer>")[0]
                try:
                    original_answer = float(original_answer)
                except Exception as e:
                    logger.warning(f"Failed to convert answer to float: {original_answer}")
                    original_answer = None
            except Exception as e:
                logger.error(f"Failed to parse question/answer from proposer output: {question_raw_responses[i]}")
                continue

            if original_answer is None:
                solver_rewards.append(-10)
                logger.warning("Original answer is None, assigning reward -10.")
                continue

            solver_raw_response = solve_question(question_text)
            solver_reward = solver_reward_fn(solver_raw_response, original_answer)
            if solver_reward == 10:
                question_and_answer_bank.append({
                    "question": question_text,
                    "answer": original_answer
                })
                logger.info(f"Added new question to bank: {question_text} (answer: {original_answer})")
            solver_rewards.append(solver_reward)
            solver_input_ids = tokenizer.encode(solver_raw_response, return_tensors="pt").to(model.device)
            # Get logits for both models
            model_logits = model(solver_input_ids).logits
            with torch.no_grad():
                ref_model_logits = ref_model(solver_input_ids).logits
            # Compute log probs
            model_log_probs = torch.nn.functional.log_softmax(model_logits, dim=-1)
            ref_model_log_probs = torch.nn.functional.log_softmax(ref_model_logits, dim=-1)
            # Gather log probs for the actual tokens
            model_selected = model_log_probs.gather(2, solver_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            ref_model_selected = ref_model_log_probs.gather(2, solver_input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            # Sum log probs for total log prob
            model_log_prob_sum = model_selected.sum(dim=-1)
            ref_model_log_prob_sum = ref_model_selected.sum(dim=-1)
            solver_model_log_probs_list.append(model_log_prob_sum)
            solver_ref_model_log_probs_list.append(ref_model_log_prob_sum)
            solver_input_ids_list.append(solver_input_ids)
            solver_raw_responses.append(solver_raw_response)
            # KL divergence per token (mean over sequence)
            kl = torch.nn.functional.kl_div(
                model_log_probs[:, :-1],  # [batch, seq, vocab]
                ref_model_log_probs[:, :-1].exp(),
                reduction='batchmean',
                log_target=False
            )
            solver_kl_divs.append(kl)
            logger.info(f"Solver Reward: {solver_reward}")
            logger.debug(f"Solver raw response: {solver_raw_response}")

        if solver_rewards:
            solver_rewards_tensor = torch.tensor(solver_rewards, dtype=torch.float32, device=model.device)
            solver_avg_reward = solver_rewards_tensor.mean()
            solver_advantages = solver_rewards_tensor - solver_avg_reward

            solver_model_log_probs_tensor = torch.cat(solver_model_log_probs_list, dim=0) if solver_model_log_probs_list else torch.tensor([], device=model.device)
            solver_ref_model_log_probs_tensor = torch.cat(solver_ref_model_log_probs_list, dim=0) if solver_ref_model_log_probs_list else torch.tensor([], device=model.device)
            solver_kl_divs_tensor = torch.stack(solver_kl_divs) if solver_kl_divs else torch.tensor([], device=model.device)

            # GRPO loss with advantage + KL penalty for solver
            kl_coeff_solver = 0.1  # You can tune this coefficient separately if desired
            if solver_model_log_probs_tensor.numel() > 0 and solver_ref_model_log_probs_tensor.numel() > 0 and solver_kl_divs_tensor.numel() > 0:
                solver_loss = -((solver_model_log_probs_tensor - solver_ref_model_log_probs_tensor) * solver_advantages).mean() + kl_coeff_solver * solver_kl_divs_tensor.mean()
                solver_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                logger.info(f"Solver loss: {solver_loss.item():.4f}")

def proposer_reward_fn(question_raw_response: str): 
    reward = 0
    if "<question>" in question_raw_response \
        and "</question>" in question_raw_response \
        and "<answer>" in question_raw_response \
        and "</answer>" in question_raw_response:
        m = re.search(r"<answer>(.*?)</answer>", question_raw_response, re.DOTALL)
        if m:
            answer = m.group(1)
            try:
                answer = float(answer)
                reward = 10 
            except Exception as e:
                logger.warning(f"Failed to convert proposer answer to float: {answer}")
                return 1
        else:
            logger.warning("No <answer>...</answer> found in proposer response.")
            reward = -10
    else:
        reward = -10
    logger.debug(f"Proposer reward: {reward} for response: {question_raw_response}")
    return reward

def solver_reward_fn(answer_raw_response: str, original_answer: Union[int, float]):
    reward = 0
    if "<answer>" in answer_raw_response \
        and "</answer>" in answer_raw_response \
        and "<reasoning>" in answer_raw_response \
        and "</reasoning>" in answer_raw_response:
        m = re.search(r"<answer>(.*?)</answer>", answer_raw_response, re.DOTALL)
        if m:
            answer = m.group(1)
            try:
                answer = float(answer)
                if abs(answer - original_answer) < 1e-6:
                    reward = 10
            except Exception as e:
                logger.warning(f"Failed to convert solver answer to float: {answer}")
                reward = 1
        else:
            logger.warning("No <answer>...</answer> found in solver response.")
            reward = -10
    else:
        reward = -10
    logger.debug(f"Solver reward: {reward} for response: {answer_raw_response}")
    return reward

def main():
    global model, tokenizer, optimizer, ref_model
    model_name = "Qwen/Qwen3-0.6B" 
    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    ref_model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6) 
    num_questions = 1
    logger.info("Starting seed question generation.")
    generate_seed_questions(num_questions)
    logger.info("Starting training loop.")
    fit()  

if __name__ == "__main__":
    main()