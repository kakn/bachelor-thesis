# Domain Coverage of Low-Resource Chatbots
This is the codebase for my bachelor thesis "Domain Coverage of Low-Resource Chatbots" (ITU, 2022). The project investigates how generative chatbots perform when trained on small, domain-specific datasets, comparing open vs. closed domain coverage using BLEU scores.

This code is quite outdated, my machine learning and software development skills have significantly improved since writing this. Since then, I’ve completed a master’s degree and gained professional software engineering experience. For more recent work, see master-thesis or music-generation-lstm.
Please see "bots/combo/combo.py" for an example of how the chatbots were created. Also, see "bachelor-thesis.pdf" for a better representation of what the project contains.

## Summary
I trained encoder-decoder LSTM chatbots on genre-labeled subsets of the Cornell Movie Dialogues Corpus (e.g. comedy, thriller, sci-fi). The aim was to see whether focused, low-resource models could generalize better than broader ones under limited data conditions.

## Key Points
- Small, domain-specific models often gave more coherent responses.
- Larger models had higher BLEU scores but less natural output.
- Open-domain models need significantly more data to perform well.
