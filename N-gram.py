#!/usr/bin/env python
# coding: utf-8

# In[8]:


import nltk
from nltk.util import ngrams
from collections import Counter
import PyPDF2


# In[12]:


nltk.download('punkt')


# In[13]:


pdf_path = "C:/Users/HONCHO/Downloads/traffic/adventure of sherlock.pdf"
with open(pdf_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])


# In[14]:


# Preprocess text
words = nltk.word_tokenize(text.lower())  # Tokenization


# In[15]:


# Define target phrase
phrase = ["i", "have", "no", "doubt", "that", "i"]
phrase_str = " ".join(phrase)


# In[16]:


# Count unigrams
unigram_counts = Counter(words)


# In[17]:


# Count N-grams
n_gram_counts = {}
for n in range(1, len(phrase) + 1):
    n_grams = list(ngrams(words, n))
    n_gram_counts[n] = Counter(n_grams)


# In[18]:


# Compute probabilities
total_words = sum(unigram_counts.values())
unigram_probs = {word: unigram_counts[word] / total_words for word in phrase}


# In[19]:


# Compute conditional probabilities for N-grams
n_gram_probs = {}
for n in range(2, len(phrase) + 1):
    n_gram_probs[n] = {
        n_gram: n_gram_counts[n][n_gram] / n_gram_counts[n-1][n_gram[:-1]]
        for n_gram in n_gram_counts[n] if n_gram[:-1] in n_gram_counts[n-1]
    }


# In[20]:


# Get counts of the target phrase and its components
phrase_count = text.lower().count(phrase_str)
partial_counts = { " ".join(phrase[:n]): text.lower().count(" ".join(phrase[:n])) for n in range(1, len(phrase) + 1) }


# In[21]:


# Compute probabilities
probabilities = {key: count / total_words for key, count in partial_counts.items()}

# Prepare results
results = [(key, partial_counts[key], probabilities[key]) for key in partial_counts]


# In[22]:


# Display results
for row in results:
    print(f"N-gram: {row[0]} | Count: {row[1]} | Probability: {row[2]:.6f}")


# In[ ]:




