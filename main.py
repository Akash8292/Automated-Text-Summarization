import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from nltk.translate import meteor_score
import pdfplumber
from torchsummary import summary

# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Open a PDF
with pdfplumber.open('1.pdf') as pdf:         # Here you can add your text
    # Extract text from the first page
    text = pdf.pages[0].extract_text()

sentences = sent_tokenize(text)

# Tokenization, stopword removal, and lemmatization using NLTK
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenized = [word_tokenize(sentence) for sentence in sentences]
new_sentences = [[lemmatizer.lemmatize(word.lower()) for word in sentence if word.isalnum() and word.lower() not in stop_words] for sentence in tokenized]

# Convert filtered sentences back to text for TF-IDF
final_sentences = [' '.join(sentence) for sentence in new_sentences]

# Calculate TF-IDF scores for words in the sentences
tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(final_sentences)

# Calculate sentence importance based on TF-IDF scores
sentence_importance_tfidf = np.squeeze(np.asarray(tfidf.sum(axis=1)))

# Calculate sentence importance based on position
sentence_importance_position = np.array([(1 - i / len(sentences)) for i in range(len(sentences))])

# Calculate sentence importance based on cosine similarity between sentences
cosine_sim_matrix = cosine_similarity(tfidf, tfidf)
sentence_importance_cosine = np.squeeze(np.asarray(cosine_sim_matrix.sum(axis=1)))

# Combine the sentence importance scores
combined_importance = (
    0.6 * sentence_importance_tfidf +
    0.2 * sentence_importance_position +
    0.2 * sentence_importance_cosine
)

# Sort sentences based on their combined importance
sortsen = [sentence for _, sentence in sorted(zip(combined_importance, sentences), reverse=True)]

# Choose the top N sentences for the extractive summary
summary_length = 7  # Increased the summary length for more content
extractive_summary = ' '.join(sortsen[:summary_length])
print("\nExtractive Summary:")
print(extractive_summary)
print("\n\n")
# Load pre-trained model and tokenizer for abstractive summarization
model_name = 'facebook/bart-large-cnn'
model = BartForConditionalGeneration.from_pretrained(model_name)

# Print the summary of the model's structure
print(model)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Tokenize the text for abstractive summarization using the extractive summary
inputs = tokenizer.encode("summarize: " + extractive_summary, return_tensors='pt', max_length=512, truncation=True)

# Generate abstractive summary
summary_ids = model.generate(inputs, max_length=400, min_length=300, length_penalty=2.0, num_beams=4, early_stopping=True)
abstractive_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("\n\n\nAbstractive Summary:")
print(abstractive_summary)

# Generate summary from the abstractive summary
summary_ids = model.generate(tokenizer.encode(abstractive_summary, return_tensors='pt'), max_length=300,min_length=200, length_penalty=3.0, num_beams=4, early_stopping=True)
generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# Print the original text, extractive summary, abstractive summary, and the final generated summary
print("\n\n\nOriginal Text:")
print(text)

print("\n\n\nGenerated Summary:")
print(generated_summary)

# Calculate BLEU score with smoothing
bleu_score = sentence_bleu([text], generated_summary, smoothing_function=SmoothingFunction().method4)
print(f"\n\nBLEU Score: {bleu_score}")

# Calculate ROUGE scores
rouge = Rouge()
rouge_scores = rouge.get_scores(text, generated_summary)
print(f"\n\nROUGE Scores: {rouge_scores}")

# Tokenize the summaries for METEOR score calculation
tokenized_summary = word_tokenize(text)
tokenized_generated_summary = word_tokenize(generated_summary)

# Calculate METEOR score
meteor_score_value = meteor_score.single_meteor_score(tokenized_summary, tokenized_generated_summary)
print(f"\n\nMETEOR Score: {meteor_score_value}")

# Refine the summary length, content, or style
short_summary = ' '.join(generated_summary.split()[:120]) + '...'
print("\n\nRefined Summary:")
print(short_summary)

# Calculate accuracy by comparing the generated summary with the original text
def calculate_accuracy(generated_summary, reference_summary):
    generated_tokens = set(generated_summary.split())
    reference_tokens = set(reference_summary.split())
    matching_tokens = generated_tokens.intersection(reference_tokens)
    accuracy = len(matching_tokens) / len(generated_tokens)
    return accuracy

accuracy = calculate_accuracy(short_summary, text)
print(f"\nAccuracy: {accuracy}")
