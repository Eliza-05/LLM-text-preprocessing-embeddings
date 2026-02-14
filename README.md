# LLM-text-preprocessing-embeddings
Este repositorio contiene la implementación y exploración de los fundamentos del preprocesamiento de texto para Large Language Models (LLMs), con un enfoque específico en la construcción y uso de embeddings.


## Assignment

Follow the excellent guide found in Chapter 2 of *Build a Large Language Model (From Scratch)* by Sebastian Raschka.

Download only these two files:
- Notebook: https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/ch02.ipynb
- Text: https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt

Run the notebook end-to-end in your environment (install `torch` and `tiktoken` if needed).

In your repo create `embeddings.ipynb` that contains:
- The core code from the book (you can copy or re-run sections)
- At least 4 markdown cells with your own explanations on why each major step matters for LLMs / agentic systems. Include an answer to:  
  **"Why do embeddings encode meaning, and how are they related to NN concepts?"**
- One small experiment: change `max_length` and `stride` → report how many samples you get and why overlap is useful

## Grading Part 1
- Notebook runs cleanly with outputs (10 pts)
- Clear personal explanations (25 pts)
- Experiment & understanding shown (15 pts)
