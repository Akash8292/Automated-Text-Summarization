# Automated Text Summarization 
## About

# Abstract
This project is a Automated text summarization system that employs extractive and abstractive techniques. It processes a PDF document, tokenizes the text, removes stopwords, and lemmatizes the remaining words using NLTK. Sentences are ranked based on their importance, calculated using TF-IDF scores, sentence position, and cosine similarity. The top-ranked sentences form an extractive summary.

The system then uses the pre-trained `facebook/bart-large-cnn` model from Hugging Face's transformers library to generate an abstractive summary. This summary is further refined to create a final summary.

The quality of the generated summary is evaluated using BLEU, ROUGE, and METEOR scores. The system outputs the original text, the extractive and abstractive summaries, the final summary, and the evaluation scores. The end result is a concise, high-quality summary of the original text, demonstrating the effectiveness of combining different summarization techniques and evaluation metrics.

## Highlights

### Extractive Summerization 

![Screenshot 2024-02-17 224415](https://github.com/Akash8292/Automated-Text-Summarization-/assets/97883391/d0601fd1-f6f4-4d65-a4aa-4aa2441c411c)

### Abstractive Summerization 

![Screenshot 2024-02-17 224510](https://github.com/Akash8292/Automated-Text-Summarization-/assets/97883391/fb0ccca4-1545-451a-bba3-ad6332c5cf35)

### Generated summerization

![Screenshot 2024-02-17 224622](https://github.com/Akash8292/Automated-Text-Summarization-/assets/97883391/16bbe138-a1c6-4460-a822-e522596d42cc)


## Prerequisites
Python>=3.8

## Getting started
1. Download the repository and unzip it.
2. Install necessary packages using pip install -r requirements.txt.
3. In this code you can upload your own .pdf aur you can write your text insted of using a .pdf.

## Result

![Screenshot 2024-02-17 224701](https://github.com/Akash8292/Automated-Text-Summarization-/assets/97883391/0d4f3190-b324-4032-8494-0deb17e765a8)

## Future Work
In future, I'm looking forward to improve model's accuracy.
