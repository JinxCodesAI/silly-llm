# Silly LLM

Silly LLM is a repository that builds on top of work outlined in **TinyStories: How Small Can Language Models Be and Still Speak
Coherent English?** paper

Overarching goal is to creates a series of scripts thatgo through following stages
1) Generates synthetic data using some permissive open source model
2) pretrain Small Language Model on that data
3) generate instruction fine-tuning dataset
4) Finetune SLM on that dataset
5) ......
