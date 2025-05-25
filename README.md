# minAZR
<!-- To change the displayed image size, use HTML instead of Markdown: -->
<img src="assets/minAZR.png" alt="minAZR" width="600" height="250"/>

<!-- You can adjust the width (or use height) as needed. For example:
<img src="assets/minAZR.png" alt="minAZR" width="600"/>
-->
Picture Credits: Original paper + a little bit of ChatGPT edits.


This is a a minimal (~400 LoC) but functional implementation of the paper: Absolute Zero: Reinforced Self-play Reasoning with Zero Data. 

# Introduction 

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) by Karpathy, I have tried to take a stab at making a minimal and functional implementation of the paper [Absolute Zero: Reinforced Self-play Reasoning with Zero Data (AZR)](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner?tab=readme-ov-file), called minAZR. 

This paper is very important to understand. This is the first time, that a purely self-play solution with __NO__ data (synthetic or from a dataset) has been used to finetune the model. The model learned to first generate problems and then solve the same problem. 

The original repository has a very good implementation, but for a beginner, it may be difficult to understand. The main logic is buried deep inside vERL calls. 

# How to get started

1. Clone the repository 
2. `pip install -r requirements.txt` (uv pip is recommended) 
3. `python main.py` 

That's it. 

# Differences with the original repository 

As you might have guessed, since it's minimal and for educative purposes, I have not implemented a lot of bells and whistles which are present in the original repository. I am listing them here. 

1. This code is highly sequential in nature. Training for 10 epochs with a decent batch_size (64) will take a lot of time and memory. 

2. The original paper uses Task-relative REINFORCE++ (TRR++). I use a variant of GRPO where I take the mean of the batch as the baseline. 

3. Original paper uses coding and 3 subtasks i.e. Induction, Deduction and Abduction. I use just a simple math problem generator and solver. 

4. Proposer rewards are linked with solver. I have decoupled them. 


I will detail their implementation in my [blog](www.mlresearchengineer.substack.com) soon. 

# What's next 
My goal is to make the performance of this checkpoint similar to the finetuned version on GSM8k. So stay tuned :) 


