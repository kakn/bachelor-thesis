from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction().method4
reference = [['is', 'how', 'die']]
candidate = ['how', 'is', 'you', 'doing', 'today']
score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
print(score)
