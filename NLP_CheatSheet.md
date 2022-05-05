## Language Models

Language models are probabilistic machine models of language used for NLP comprehension tasks. They learn a probability of word occurrence over a sequence of words and use it to estimate the relative likelihood of different phrases. This is useful in many applications, such as speech recognition, optical character recognition, handwriting recognition, machine translation, spelling correction, and many other applications.

Common language models include:

## Statistical models

Bag of words (unigram model)
applications include term frequency, topic modeling, and word clouds  

- n-gram models  
- Neural Language Modeling (NLM).  
- Natural Language Processing  

Natural language processing (NLP) is concerned with enabling computers to interpret, analyze, and approximate the generation of human speech. Typically, this would refer to tasks such as generating responses to questions, translating languages, identifying languages, summarizing documents, understanding the sentiment of text, spell checking, speech recognition, and many other tasks. The field is at the intersection of linguistics, AI, and computer science.  


## Natural Language Toolkit  

Natural Language Toolkit (NLTK) is a Python library used for building Python programs that work with human language data for applying in statistical natural language processing (NLP).
  
  
NLTK contains text processing libraries for tokenization, parsing, classification, stemming, tagging and semantic reasoning. It also includes graphical demonstrations and sample data sets for NLP.
  
  
- Text Similarity in NLP
- Text similarity is a facet of NLP concerned with the similarity between texts. Two popular text similarity metrics are Levenshtein distance and cosine similarity.

Levenshtein distance, also called edit distance, is defined as the minimum number of edit operations (deletions, insertions, or substitutions) required to transform a text into another.

Cosine similarity measures the cosine of the angle between two vectors. To determine cosine similarity, text documents need to be converted into vectors.

## Language Prediction in NLP
  
Language prediction is an application of NLP concerned with predicting language given preceding language.

Auto-suggest and suggested replies are common forms of language prediction. Common approaches inlcude:

n-grams using Markov chains,
Long Short Term Memory (LSTM) using a neural network.

# Introduction to Regular Expressions


Optional Quantifiers in Regular Expressions
In Regular expressions, optional quantifiers are denoted by a question mark `?`. It indicates that a character can appear either 0 or 1 time. For example, the regular expression humou`?`r will match the text humour as well as the text humor.



## Character Sets in Regular Expressions
Regular expression character sets denoted by a pair of brackets `[]` will match any of the characters included within the brackets. For example, the regular expression con`[sc]`en`[sc]`us will match any of the spellings `consensus, concensus, consencus, and concencus.`

## Literals in Regular Expressions

In Regular expression, the `literals` are the simplest characters that will match the exact text of the literals. For example, the regex `monkey` will completely match the text `monkey` but will also match monkey in text `The monkeys like to eat bananas.`

## Wildcards in Regular expressions
In Regular expression, wildcards are denoted with the period `.` and it can match any single character (letter, number, symbol or whitespace) in a piece of text. For example, the regular expression `.........` will match the text `orangutan`, `marsupial`, or any other 9-character text.
  
## Regular Expression Ranges
Regular expression ranges are used to specify a range of characters that can be matched. Common regular expression ranges include:   
`[A-Z]`. : match any uppercase letter    
`[a-z].` : match any lowercase letter    
`[0-9]`. : match any digit    
`[A-Za-z]` : match any uppercase or lowercase letter.  
  
## Shorthand Character Classes in Regular Expressions
Shorthand character classes simplify writing regular expressions. For example, \w represents the regex range [A-Za-z0-9_], \d represents [0-9], \W represents [^A-Za-z0-9_] matching any character not included by \w, \D represents [^0-9] matching any character not included by \d.
  
## Kleene Star & Kleene Plus in Regular Expressions
In Regular expressions, the Kleene star(*) indicates that the preceding character can occur 0 or more times. For example, meo*w will match mew, meow, meooow, and meoooooooooooow. The Kleene plus(+) indicates that the preceding character can occur 1 or more times. For example, meo+w will match meow, meooow, and meoooooooooooow, but not match mew.
  
## Grouping in Regular Expressions
In Regular expressions, grouping is accomplished by open `( and close parenthesis )`. Thus the regular expression I love `(baboons|gorillas)` will match the text `I love baboons` as well as `I love gorillas`, as the grouping limits the reach of the | to the text within the parentheses.



# Text Preprocessing
In natural language processing, text preprocessing is the practice of cleaning and preparing text data. NLTK and re are common Python libraries used to handle many text preprocessing tasks.
  
## Noise Removal  

In natural language processing, noise removal is a text preprocessing task devoted to stripping text of formatting.

```python
import re
 
text = "Five fantastic fish flew off to find faraway functions. Maybe find another five fantastic fish? Find my fish with a function please!"
 
# remove punctuation
result = re.sub(r'[\.\?\!\,\:\;\"]', '', text)
 
print(result)
# Five fantastic fish flew off to find faraway functions Maybe find another five fantastic fish Find my fish with a function please
```
  
## Tokenization
In natural language processing, tokenization is the text preprocessing task of breaking up text into smaller components of text (known as tokens).    

```python
from nltk.tokenize import word_tokenize
 
text = "This is a text to tokenize"
tokenized = word_tokenize(text)
 
print(tokenized)
# ["This", "is", "a", "text", "to", "tokenize"]
```

  
## Text Normalization  

In natural language processing, normalization encompasses many text preprocessing tasks including stemming, lemmatization, upper or lowercasing, and stopwords removal.  
  
## Stemming
In natural language processing, stemming is the text preprocessing normalization task concerned with bluntly removing word affixes (prefixes and suffixes).    

```python
from nltk.stem import PorterStemmer
 
tokenized = ["So", "many", "squids", "are", "jumping"]
 
stemmer = PorterStemmer()
stemmed = [stemmer.stem(token) for token in tokenized]
 
print(stemmed)
# ['So', 'mani', 'squid', 'are', 'jump']

```
  
## Lemmatization  

In natural language processing, lemmatization is the text preprocessing normalization task concerned with bringing words down to their root forms. 

```Python
from nltk.stem import WordNetLemmatizer
 
tokenized = ["So", "many", "squids", "are", "jumping"]
 
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(token) for token in tokenized]
 
print(stemmed)
# ['So', 'many', 'squid', 'be', 'jump']

```

## Stopword Removal  

In natural language processing, stopword removal is the process of removing words from a string that don’t provide any information about the tone of a statement.  
  
```python
from nltk.corpus import stopwords 
 
# define set of English stopwords
stop_words = set(stopwords.words('english')) 
 
# remove stopwords from tokens in dataset
statement_no_stop = [word for word in word_tokens if word not in stop_words]
```

## Part-of-Speech Tagging

In natural language processing, part-of-speech tagging is the process of assigning a part of speech to every word in a string. Using the part of speech can improve the results of lemmatization.

# Rule-Based Chatbots


## Rule-Based Chatbots
Rule-based chatbots are structured as a dialog tree and often use regular expressions to match a user’s input to human-like responses. The aim is to simulate the back-and-forth of a real-life conversation, often in a specific context, like telling the user what the weather is like outside. In chatbot design, rule-based chatbots are closed-domain, also called dialog agents, because they are limited to conversations on a specific subject.

## Chatbot Intents
In chatbots design, an intent is the purpose or category of the user query. The user’s utterance gets matched to a chatbot intent. In rule-based chatbots, you can use regular expressions to match a user’s statement to a chatbot intent.

```python
import re
 
matching_intents = {'weather_intent': [r'weather.*on (\w+)']}
 
def match_reply(self, reply):
  for key, values in matching_intents.items():
    for regex_pattern in values:
      found_match = re.match(regex_pattern, reply.lower())
      if found_match and key == 'weather_intent':
        return weather_intent(found_match.groups()[0])
        
  return input("I did not understand you. Can you please ask your question again?")
```

## Chatbot Utterances
In chatbot design, an utterance is a statement that the user makes to the chatbot. The chatbot attempts to match the utterance to an intent.
   
## Chatbot Entities
In chatbot design, an entity is a value that is parsed from a user utterance and passed for use within the user response.  
  
  


```python

```
