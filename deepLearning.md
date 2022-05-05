# Long Short Term Memory Networks

## Long short-term memory networks (LSTMs) are often used in deep learning programs for natural language processing.



Information Persistence and Neural Networks

We do not create a new language every time we speak –– every human language has a persistent set of grammar rules and collection of words that we rely on to interpret it. As you read this article, you understand each word based on your knowledge of grammar rules and interpretation of the preceding and following words. At the beginning of the next sentence, you continue to utilize information from earlier in the passage to follow the overarching logic of the paragraph. Your knowledge of language is both persistent across interactions, and adaptable to new conversations.

Neural networks are a machine learning framework loosely based on the structure of the human brain. They are very commonly used to complete tasks that seem to require complex decision making, like speech recognition or image classification. Yet, despite being modeled after neurons in the human brain, early neural network models were not designed to handle temporal sequences of data, where the past depends on the future. As a result, early models performed very poorly on tasks in which prior decisions are a strong predictor of future decisions, as is the case in most human language tasks. However, more recent models, called recurrent neural networks (RNN), have been specifically designed to process inputs in a temporal order and update the future based on the past, as well as process sequences of arbitrary length.

RNNs are one of the most commonly used neural network architectures today. Within the domain of Natural Language Processing, they are often used in speech generation and machine translation tasks. Additionally, they are often used to solve speech recognition and optical character recognition tasks. Let’s dive into their implementation!

#### Neural Networks

A single neuron in a network, reference to graphic, takes in a single piece of input data, reference to graphic, and performs some data transformation to produce a single piece of output reference to graphic.    

![](https://content.codecademy.com/programs/chatbots/articles/lstm/perceptron.png)


 The very first network models, called `perceptrons`, were relatively simple systems that used many combinations of simple functions, like computing the slope of a line. While these transformations were very `simple in isolation`, together they resulted in sophisticated behavior for the entire system and allowed for large numbers of transformations to be strung together in order to compute advanced functions. However, the range of perceptron applications was still limited, and there were very simple functions that they just simply couldn’t compute. In attempting to solve this issue by layering perceptrons together, researchers ran into another problem: few computers at the time were able to store enough data to execute these programs.

#### Deep Neural Networks

By the early 2000s, when innovations in computer hardware allowed for more complex modeling techniques, researchers had already developed neural network systems that combined many layers of neurons, including convolutional neural networks (CNN), multi-layer perceptrons (MLP), and recurrent neural networks (RNN). All three of these architectures are called deep neural networks, because they have many layers of neurons that combine to create a “deep” stack of neurons. Each of these deep-learning architectures have their own relative strengths:

- MLP networks are comprised of layered perceptrons. They tend to be good at solving simple tasks, like applying a filter to each pixel in a photo.  
- CNN networks are designed to process image data, applying the same convolution function across an entire input image. This makes it simpler and more efficient to process images, which generally yields very high-dimensional output and requires a great deal of processing. 
- Finally, RNNs became widely adopted within natural language processing because they integrate a loop into the connections between neurons, allowing information to persist across a chain of neurons.  
  
![](https://content.codecademy.com/programs/chatbots/articles/lstm/unrolled-rnn.png). 

When the chain of neurons in an RNN is “rolled out,” it becomes easier to see that these models are made up of many copies of the same neuron, each passing information to its successor. Neurons that are not the first or last in a rolled out RNN are sometimes referred to as “hidden” network layers; the first and last neurons are called the “input” and “output” layers, respectively. The chain structure of RNNs places them in close relation to data with a clear temporal ordering or list-like structure — such as human language, where words obviously appear one after another. Standard RNNs are certainly the best fit for tasks that involve sequences, like the translation of a sentence from one language to another.

#### Long-term Dependencies

While the broad family of RNNs powers many of today’s advanced artificial intelligence systems, one particular RNN variant, the long short-term memory network (LSTM), has become popular for building realistic chatbot systems. When attempting to predict the next word in a conversation, words that were mentioned recently often provide enough information to make a strong guess. For instance, when trying to complete the sentence “The grass is always: ____,” most English-speakers do not need additional context to guess that the last word is “greener.” Standard RNNs are difficult to train, and fail to capture long-term dependencies well; they perform best on short sequences, when relevant context and the word to be predicted fall within a short distance of one another.  
   
![](https://content.codecademy.com/programs/chatbots/articles/lstm/unrolled-rnn-long-term-dependency.png))
 
 nrolled RNN showing long-term dependencies
 
 However, there are also many cases of longer sequences in which more context is needed to make an accurate prediction, and that context is less obvious and at a greater lexical distance from the word it determines than in the example above. For instance, consider trying to complete the sentence “It is winter and there has been little sunlight. The grass is always _____.” While the structure of the second sentence suggests that our word is probably an adjective related to grass, we need context from further back in the text to guess that the correct word is “brown.” As the gap between context and the word to predict grows, standard RNNs become less and less accurate. This situation is commonly referred to as the long-term dependency problem. The solution to this problem? A neural network specifically designed for long-term memory — the LSTM!

#### Long Short-term Memory Networks

Every model in the RNN family, including LSTMs, is a chain of repeating neurons at its base. Within standard RNNs, each layer of neurons will only perform a single operation on the input data.  
![](https://content.codecademy.com/programs/chatbots/articles/lstm/rnn-under-the-hood.png)  
   
 However, within an LSTM, groups of neurons perform four distinct operations on input data, which are then sequentially combined.   
 
![](https://content.codecademy.com/programs/chatbots/articles/lstm/lstm-cell.png)   
  
The most important aspect of an LSTM is the way in which the transformed input data is combined by adding results to state, or cell memory, represented as vectors. There are two states that are produced for the first step in the sequence and then carried over as subsequent inputs are processed: cell state, and hidden state.  
 
 ![](![image.png](attachment:image.png)). 
 
The cell state carries information through the network as we process a sequence of inputs. At each timestep, or step in the sequence, the updated input is appended to the cell state by a gate, which controls how much of the input should be included in the final product of the cell state. This final product, which is fed as input to the next neural network layer at the next timestep, is called a hidden state. The final output of a neural network is often the result contained in the final hidden state, or an average of the results across all hidden states in the network.

The persistence of the majority of a cell state across data transformations, combined with incremental additions controlled by the gates, allows for important information from the initial input data to be maintained in the neural network. Ultimately, this allows for information from far earlier in the input data to be used in decisions at any point in the model.



## Introduction to seq2seq (multiple code parts)
LSTMs are pretty extraordinary, but they’re only the tip of the iceberg when it comes to actually setting up and running a neural language model for text generation. In fact, an LSTM is usually just a single component in a larger network.
  
One of the most common neural models used for text generation is the sequence-to-sequence model, commonly referred to as seq2seq (pronounced “seek-to-seek”). A type of encoder-decoder model, seq2seq uses recurrent neural networks (RNNs) like LSTM in order to generate output, token by token or character by character.
  
So, where does seq2seq show up?
  
- Machine translation software like Google Translate
- Text summary generation
- Chatbots
- Named Entity Recognition (NER)
- Speech recognition
  
seq2seq networks have two parts:
  
An encoder that accepts language (or audio or video) input. The output matrix of the encoder is discarded, but its state is preserved as a vector.
  
A decoder that takes the encoder’s final state (or memory) as its initial state. We use a technique called `“teacher forcing”` to train the decoder to predict the following text (characters or words) in a target sequence given the previous text.  

do in parallel with this one https://keras.io/examples/nlp/lstm_seq2seq/

![image.png](attachment:image.png)

 In our case, we’ll be using TensorFlow with the Keras API to build a pretty limited English-to-Spanish translator (we’ll explain this later and you’ll get an opportunity to improve it).
   
We can import Keras from Tensorflow like this:  

```python
from tensorflow import keras
```

Also, do not worry about memorizing anything we cover here. The purpose of this lesson is for you to make sense of what each part of the code does and how you can modify it to suit your own needs. In fact, the code we’ll be using is mostly derived from  
  
https://keras.io/examples/nlp/lstm_seq2seq/  

First things first: preprocessing the text data. Noise removal depends on your use case — do you care about casing or punctuation? For many tasks they are probably not important enough to justify the additional processing. This might be the time to make changes.
   
We’ll need the following for our Keras implementation:  
  
-  vocabulary sets for both our input (English) and target (Spanish) data  
-  the total number of unique word tokens we have for each set  
-  the maximum sentence length we’re using for each language  
  
We also need to mark the start and end of each document (sentence) in the target samples so that the model recognizes where to begin and end its text generation (no book-long sentences for us!). One way to do this is adding `"<START>"` at the beginning and `"<END>"` at the end of each target document (in our case, this will be our Spanish sentences). For example, "Estoy feliz." becomes `"<START> Estoy feliz. <END>"`.

Before you dig into the instructions, read through the existing code in script.py and try to make sense of each line.



```python
from tensorflow import keras
import re
# Importing our translations
data_path = "resources/span-eng.txt"
# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:
  lines = f.read().split('\n')

# LISTS HOLD SENTENCES
input_docs = []
target_docs = []

# EMPTY VOCAB SETS
input_tokens = set()
target_tokens = set()
  
for line in lines:
  # Input and target sentences are separated by tabs
  input_doc, target_doc = line.split('\t')
    
  # Appending each input sentence to input_docs
  input_docs.append(input_doc)
  # Splitting words from punctuation
  target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
 
  target_doc = "<START> " + target_doc + " <END>"
  target_docs.append(target_doc)

  # Now we split up each sentence into words
  # and add each unique word to our vocabulary set
  for token in re.findall(r"[\w']+|[^\s\w]", input_doc): input_tokens.add(token)
    
  for token in target_doc.split(): target_tokens.add(token)

input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))

# NUMBER OF TOKENS
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

# MAX of list comprehension
try:
  max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
  max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])
except ValueError:
  pass

```

## Training Setup (part 1)
  
For each sentence, Keras expects a NumPy matrix containing one-hot vectors for each token. What’s a one-hot vector? In a one-hot vector, every token in our set is represented by a 0 except for the current token which is represented by a 1. For example given the vocabulary `["the", "dog", "licked", "me"]`, a one-hot vector for “dog” would look like `[0, 1, 0, 0].` 
    
In order to vectorize our data and later `translate it from vectors`, it’s helpful to have a `features dictionary `(and a `reverse features dictionary`) to easily translate between all the 1s and 0s and actual words. We’ll build out the following:
  
-  **a features dictionary for English ** 
-  **a features dictionary for Spanish**  
-  **a reverse features dictionary for English** (where the keys and values are swapped)  
-  **a reverse features dictionary for Spanish**

  
Once we have all of our features dictionaries set up, it’s time to vectorize the data! We’re going to need vectors to input into our encoder and decoder, as well as a vector of target data we can use to train the decoder.

Because each matrix is almost all zeros, we’ll `use numpy.zeros()` from the NumPy library to build them out.
  
```python
import numpy as np
 
encoder_input_data = np.zeros(
    (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
```
  
## Let’s break this down:
  
We defined a NumPy matrix of zeros called encoder_input_data with two arguments:
  
-  the shape of the matrix — in our case the `number of documents` (or sentences) by the maximum token sequence length (the longest sentence we want to see) by the number of unique tokens (or words)    
    
      
-  the data type we want — in our case NumPy’s float32, which can speed up our processing a bit



```python

import numpy as np
print('Number of samples:', len(input_docs))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# dict of words, and an index value
input_features_dict  = dict([(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict([(token, i) for i, token in enumerate(target_tokens)])
# Reverse-lookup token index to decode sequences back to something readable.
reverse_input_features_dict = dict((i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict((i, token) for token, i in target_features_dict.items())


encoder_input_data = np.zeros((len(input_docs), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
# Build out the decoder_input_data matrix:
decoder_input_data = np.zeros( (len(input_docs), max_decoder_seq_length,num_decoder_tokens ), dtype='float32')
# Build out the decoder_target_data matrix:
decoder_target_data = np.zeros((len(input_docs), max_decoder_seq_length,num_decoder_tokens ),dtype='float32')

print("\nHere's the first item in the encoder input matrix:\n", encoder_input_data[0], "\n\nThe number of columns should match the number of unique input tokens and the number of rows should match the maximum sequence length for input sentences.")

```

    Number of samples: 11
    Number of unique input tokens: 18
    Number of unique output tokens: 27
    Max sequence length for inputs: 4
    Max sequence length for outputs: 12
    
    Here's the first item in the encoder input matrix:
     [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]] 
    
    The number of columns should match the number of unique input tokens and the number of rows should match the maximum sequence length for input sentences.


## Training Setup (part 2)  

At this point we need to fill out the 1s in each vector. We can loop over each English-Spanish pair in our training sample using the features dictionaries to add a 1 for the token in question. For example, the dog sentence (["the", "dog", "licked", "me"]) would be split into the following matrix of vectors:  
```python
[
  [1, 0, 0, 0], # timestep 0 => "the"
  [0, 1, 0, 0], # timestep 1 => "dog"
  [0, 0, 1, 0], # timestep 2 => "licked"
  [0, 0, 0, 1], # timestep 3 => "me"
]
```
  
You’ll notice the vectors have timesteps — we use these to track where in a given document (sentence) we are.
   
To build out a three-dimensional NumPy matrix of one-hot vectors, we can assign a value of 1 for a given word at a given timestep in a given line:  
  
```python
matrix_name[line, timestep, features_dict[token]] = 1.
```  
Keras will fit — or train — the seq2seq model using these matrices of one-hot vectors:
  
-  the encoder input data    
-  the decoder input data  
-  the decoder target data  
  
Hang on a second, why build two matrices of decoder data? Aren’t we just encoding and decoding?
  
The reason has to do with a technique known as `teacher forcing` that most seq2seq models employ during training. Here’s the idea: we have a Spanish input token from the previous timestep to help train the model for the current timestep’s target token.


![image.png](attachment:image.png)  

- it puts all words to left side of current word thru decoder
- new word comes out, that gets appended to output
- repeat


```python
# line is a countern number
# 3 different matrixes get updated, encoder, decoder,decodertarget
for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
  # -----UPDATE ENCODER MATRIX
  for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
    encoder_input_data[line, timestep, input_features_dict[token]] = 1.
    
  for timestep, token in enumerate(target_doc.split()):
  	# -----UPDATE DECODER MATRIX
    decoder_input_data[line, timestep, target_features_dict[token]] = 1.

    # -----UPDATE DECODER TARGET MATRIX (>0 skip <start>, -1 its ahead by 1 timestamp cus first token isnt start, other is)
    if timestep > 0: decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.
```

## Encoder Training Setup 


It’s time for some deep learning!

Deep learning models in Keras are built in layers, where each layer is a step in the model.

Our encoder requires two layer types from Keras:

- An input layer, which defines a matrix to hold all the one-hot vectors that we’ll feed to the model.  
- An LSTM layer, with some output dimensionality.  

We can import these layers as well as the model we need like so:

```python
from keras.layers import Input, LSTM
from keras.models import Model
```
Next, we set up the input layer, which requires some number of dimensions that we’re providing. In this case, we know that we’re passing in all the encoder tokens, but we don’t necessarily know our batch size `(how many chocolate chip cookies sentences we’re feeding the model at a time)`. Fortunately, we can say `None` because the code is written to handle varying batch sizes, so we don’t need to specify that dimension.
  
```python
# the shape specifies the input matrix sizes
encoder_inputs = Input(shape=(None, num_encoder_tokens))
```
 
For the LSTM layer, we need to select the dimensionality (the size of the LSTM’s hidden states, which helps determine how closely the model molds itself to the training data — something we can play around with) and whether to return the state (in this case we do):  

```python
encoder_lstm = LSTM(100, return_state=True)
# we're using a dimensionality of 100
# so any LSTM output matrix will have 
# shape [batch_size, 100]
```
  
Remember, the only thing we want from the encoder is its final states. We can get these by `linking our LSTM layer with our input layer`:
   
   
```python
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
```  
  
`encoder_outputs` isn’t really important for us, so we can just discard it. However, the `states`, we’ll save in a list:  

  
    
```python
encoder_states = [state_hidden, state_cell]
```  
  




```python
#from prep import num_encoder_tokens

from tensorflow import keras
from keras.layers import Input, LSTM
from keras.models import Model

# Create the input layer:
encoder_inputs = Input(shape=(None,num_encoder_tokens))

# Create the LSTM layer:
encoder_lstm = LSTM(256,return_state=True)

# Retrieve the outputs and states:
encoder_outputs, state_hidden, state_cell  = encoder_lstm(encoder_inputs)

# Put the states together in a list:
encoder_states = [state_hidden,state_cell]

```

    Using TensorFlow backend.


## Decoder Training Setup  

The decoder looks a lot like the encoder (phew!), with an input layer and an LSTM layer that we use together:  
  
```python 
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(100, return_sequences=True, return_state=True)
# This time we care about full return sequences
```
However, with our decoder, we pass in the state data from the encoder, along with the decoder inputs. This time, we’ll keep the output instead of the states:  
 
```python
# The two states will be discarded for now
decoder_outputs, decoder_state_hidden, decoder_state_cell =  decoder_lstm(decoder_inputs, initial_state=encoder_states)
```  
  
We also need to run the output through a final activation layer, using the Softmax function, that will give us the probability distribution — where all probabilities sum to one — for each token. The final layer also transforms our LSTM output from a dimensionality of whatever we gave it (in our case, 10) to the number of unique words within the hidden layer’s vocabulary (i.e., the number of unique target tokens, which is definitely more than 10!).  
  
```python
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
 
decoder_outputs = decoder_dense(decoder_outputs)
```

Keras’s implementation could work with several layer types, but Dense is the least complex, so we’ll go with that. We also need to modify our import statement to include it before running the code  
  
from keras.layers import Input, LSTM, Dense




```python
# The decoder input and LSTM layers:
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)

# Retrieve the LSTM outputs and states:
# This time though, pass in the encoder_states as the initial_state
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Build a final Dense layer:
from keras.layers import Input, LSTM, Dense
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
 
# Filter outputs through the Dense layer:
decoder_outputs = decoder_dense(decoder_outputs)



```

# Build and Train seq2seq  

Alright! Let’s get model-building!
  
First, we define the seq2seq model using the Model() function we imported from Keras. To make it a seq2seq model, we feed it the encoder and decoder inputs, as well as the decoder output:
  
```python
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)  
```
  
Finally, our model is ready to train. First, we compile everything. Keras models demand two arguments to compile:
  
-  An optimizer (we’re using RMSprop, which is a fancy version of the widely-used gradient descent) to help minimize our error rate (how bad the model is at guessing the true next word given the previous words in a sentence).  

-  A loss function (we’re using the logarithm-based cross-entropy function) to determine the error rate.  
  
Because we care about accuracy, we’re adding that into the metrics to pay attention to while training. Here’s what the compiling code looks like:  

```python 
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])  
``` 
  
  
Next we need to fit the compiled model. To do this, we give the `.fit()` method the encoder and decoder input data (what we pass into the model), the decoder target data (what we expect the model to return given the data we passed in), and some numbers we can adjust as needed:
  
-  batch size (smaller batch sizes mean more time, and for some problems, smaller batch sizes will be better, while for other problems, larger batch sizes are better)
-  the number of epochs or cycles of training (more epochs mean a model that is more trained on the dataset, and that the process will take more time)
-  validation split (what percentage of the data should be set aside for validating — and determining when to stop training your model — rather than training)  
    
Keras will take it from here to get you a (hopefully) nicely trained seq2seq model:
    
```python
model.fit([encoder_input_data, decoder_input_data], 
          decoder_target_data,
          batch_size=10,
          epochs=100,
          validation_split=0.2)
```

##  calibration notes 
Fitting time! Because we don’t want to crash this exercise, we’ll make the batch size large and the number of epochs very small. (Note that small batch sizes are more prone to crashing a deep learning program in general, but in our case we care about time.)


```python
# Building the training model:
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print("Model summary:\n")
training_model.summary()
print("\n\n")

# Compile the model:
training_model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Choose the batch size
# and number of epochs:
batch_size = 50
epochs = 50

print("Training the model:\n")
# Train the model:

training_model.fit([encoder_input_data, decoder_input_data], 
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
```

    Model summary:
    
    Model: "model_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, None, 18)     0                                            
    __________________________________________________________________________________________________
    input_2 (InputLayer)            (None, None, 27)     0                                            
    __________________________________________________________________________________________________
    lstm_1 (LSTM)                   [(None, 256), (None, 281600      input_1[0][0]                    
    __________________________________________________________________________________________________
    lstm_2 (LSTM)                   [(None, None, 256),  290816      input_2[0][0]                    
                                                                     lstm_1[0][1]                     
                                                                     lstm_1[0][2]                     
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, None, 27)     6939        lstm_2[0][0]                     
    ==================================================================================================
    Total params: 579,355
    Trainable params: 579,355
    Non-trainable params: 0
    __________________________________________________________________________________________________
    
    
    
    Training the model:
    
    Train on 8 samples, validate on 3 samples
    Epoch 1/50
    8/8 [==============================] - 1s 85ms/step - loss: 1.3055 - accuracy: 0.0104 - val_loss: 1.4429 - val_accuracy: 0.1667
    Epoch 2/50
    8/8 [==============================] - 0s 2ms/step - loss: 1.2788 - accuracy: 0.1250 - val_loss: 1.4213 - val_accuracy: 0.1667
    Epoch 3/50
    8/8 [==============================] - 0s 2ms/step - loss: 1.2501 - accuracy: 0.1146 - val_loss: 1.3807 - val_accuracy: 0.1389
    Epoch 4/50
    8/8 [==============================] - 0s 2ms/step - loss: 1.2007 - accuracy: 0.1042 - val_loss: 1.2863 - val_accuracy: 0.0833
    Epoch 5/50
    8/8 [==============================] - 0s 2ms/step - loss: 1.0967 - accuracy: 0.0833 - val_loss: 1.2398 - val_accuracy: 0.0833
    Epoch 6/50
    8/8 [==============================] - 0s 2ms/step - loss: 1.0428 - accuracy: 0.0833 - val_loss: 1.2568 - val_accuracy: 0.1667
    Epoch 7/50
    8/8 [==============================] - 0s 2ms/step - loss: 1.0266 - accuracy: 0.1146 - val_loss: 1.1834 - val_accuracy: 0.0833
    Epoch 8/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.9432 - accuracy: 0.0833 - val_loss: 1.1818 - val_accuracy: 0.1667
    Epoch 9/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.9091 - accuracy: 0.1042 - val_loss: 1.1762 - val_accuracy: 0.1667
    Epoch 10/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.8772 - accuracy: 0.1042 - val_loss: 1.1889 - val_accuracy: 0.1944
    Epoch 11/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.8576 - accuracy: 0.1562 - val_loss: 1.2026 - val_accuracy: 0.1667
    Epoch 12/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.8511 - accuracy: 0.1146 - val_loss: 1.2215 - val_accuracy: 0.1667
    Epoch 13/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.8625 - accuracy: 0.1146 - val_loss: 1.1597 - val_accuracy: 0.2222
    Epoch 14/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.7753 - accuracy: 0.1458 - val_loss: 1.1621 - val_accuracy: 0.2222
    Epoch 15/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.7465 - accuracy: 0.1875 - val_loss: 1.1622 - val_accuracy: 0.2222
    Epoch 16/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.7226 - accuracy: 0.1458 - val_loss: 1.1693 - val_accuracy: 0.2222
    Epoch 17/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.7068 - accuracy: 0.1771 - val_loss: 1.2230 - val_accuracy: 0.1667
    Epoch 18/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.7224 - accuracy: 0.1146 - val_loss: 1.2804 - val_accuracy: 0.1667
    Epoch 19/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.7563 - accuracy: 0.1562 - val_loss: 1.1839 - val_accuracy: 0.2500
    Epoch 20/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.6401 - accuracy: 0.1875 - val_loss: 1.1924 - val_accuracy: 0.2222
    Epoch 21/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.6136 - accuracy: 0.2083 - val_loss: 1.2036 - val_accuracy: 0.1944
    Epoch 22/50
    8/8 [==============================] - 0s 3ms/step - loss: 0.5958 - accuracy: 0.1875 - val_loss: 1.2779 - val_accuracy: 0.1944
    Epoch 23/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.6026 - accuracy: 0.1979 - val_loss: 1.3051 - val_accuracy: 0.1667
    Epoch 24/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.6238 - accuracy: 0.2083 - val_loss: 1.2761 - val_accuracy: 0.2222
    Epoch 25/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.5530 - accuracy: 0.2188 - val_loss: 1.2636 - val_accuracy: 0.1944
    Epoch 26/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.5330 - accuracy: 0.2188 - val_loss: 1.3047 - val_accuracy: 0.2222
    Epoch 27/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.5137 - accuracy: 0.2083 - val_loss: 1.3004 - val_accuracy: 0.1667
    Epoch 28/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.5037 - accuracy: 0.2396 - val_loss: 1.3729 - val_accuracy: 0.1944
    Epoch 29/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.4933 - accuracy: 0.2083 - val_loss: 1.3413 - val_accuracy: 0.1667
    Epoch 30/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.5142 - accuracy: 0.2292 - val_loss: 1.3782 - val_accuracy: 0.2222
    Epoch 31/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.5727 - accuracy: 0.1458 - val_loss: 1.3373 - val_accuracy: 0.1667
    Epoch 32/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.4966 - accuracy: 0.2396 - val_loss: 1.3454 - val_accuracy: 0.2222
    Epoch 33/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.4436 - accuracy: 0.2396 - val_loss: 1.3378 - val_accuracy: 0.1944
    Epoch 34/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.4112 - accuracy: 0.2917 - val_loss: 1.3694 - val_accuracy: 0.2222
    Epoch 35/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.3946 - accuracy: 0.2812 - val_loss: 1.3672 - val_accuracy: 0.1944
    Epoch 36/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.3868 - accuracy: 0.2917 - val_loss: 1.4682 - val_accuracy: 0.2222
    Epoch 37/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.3972 - accuracy: 0.2604 - val_loss: 1.4392 - val_accuracy: 0.1111
    Epoch 38/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.4139 - accuracy: 0.2708 - val_loss: 1.4927 - val_accuracy: 0.2222
    Epoch 39/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.3810 - accuracy: 0.2708 - val_loss: 1.3962 - val_accuracy: 0.1944
    Epoch 40/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.3484 - accuracy: 0.3333 - val_loss: 1.4295 - val_accuracy: 0.2222
    Epoch 41/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.3288 - accuracy: 0.3438 - val_loss: 1.4087 - val_accuracy: 0.1944
    Epoch 42/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.3164 - accuracy: 0.3438 - val_loss: 1.4511 - val_accuracy: 0.2500
    Epoch 43/50
    8/8 [==============================] - 0s 1ms/step - loss: 0.3115 - accuracy: 0.3438 - val_loss: 1.4325 - val_accuracy: 0.1944
    Epoch 44/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.3353 - accuracy: 0.3125 - val_loss: 1.5222 - val_accuracy: 0.3056
    Epoch 45/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.4344 - accuracy: 0.2396 - val_loss: 1.4898 - val_accuracy: 0.1389
    Epoch 46/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.3704 - accuracy: 0.3021 - val_loss: 1.5211 - val_accuracy: 0.3056
    Epoch 47/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.3188 - accuracy: 0.3125 - val_loss: 1.4361 - val_accuracy: 0.2778
    Epoch 48/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.2800 - accuracy: 0.3646 - val_loss: 1.4774 - val_accuracy: 0.3056
    Epoch 49/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.2614 - accuracy: 0.3542 - val_loss: 1.4386 - val_accuracy: 0.3056
    Epoch 50/50
    8/8 [==============================] - 0s 2ms/step - loss: 0.2507 - accuracy: 0.3854 - val_loss: 1.4973 - val_accuracy: 0.3056





    <keras.callbacks.callbacks.History at 0x7f9cf2736c10>



## Setup for Testing  

Now our model is ready for testing! Yay! However, to generate some original output text, we need to redefine the seq2seq architecture in pieces. Wait, didn’t we just define and train a model?
  
Well, yes. But the model we used for training our network only works when we already know the target sequence. **This time, we have no idea what the Spanish should be for the English we pass in!** So we need a model that will decode step-by-step instead of using teacher forcing. To do this, we need a seq2seq network in individual pieces.  


To start, we’ll build an encoder model with our encoder inputs and the `placeholders for the encoder’s output states`:  

```python
encoder_model = Model(encoder_inputs, encoder_states) 
```

Next up, we need `placeholders for the decoder’s input states`, which we can build as input layers and store together. Why? We don’t know what we want to decode yet or what hidden state we’re going to end up with, so we need to do everything step-by-step. We need to pass the encoder’s final hidden state to the decoder, sample a token, and get the updated hidden state back. Then we’ll be able to (manually) pass the updated hidden state back into the network:  

```python 
latent_dim = 256
decoder_state_input_hidden = Input(shape=(latent_dim,))
 
decoder_state_input_cell = Input(shape=(latent_dim,))
 
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell] 
```

Using the decoder LSTM and decoder dense layer (with the activation function) that we trained earlier, we’ll create new decoder states and outputs:
  
```python
decoder_outputs, state_hidden, state_cell = 
    decoder_lstm(decoder_inputs, 
    initial_state=decoder_states_inputs)
 
# Saving the new LSTM output states:
decoder_states = [state_hidden, state_cell]
 
# Below, we redefine the decoder output
# by passing it through the dense layer:
decoder_outputs = decoder_dense(decoder_outputs)  
  
``` 

Finally, we can set up the decoder model. This is where we bring together:

-  the decoder inputs (the decoder input layer)
-  the decoder input states (the final states from the encoder)
-  the decoder outputs (the NumPy matrix we get from the final output layer of the decoder)
-  the decoder output states (the memory throughout the network from one word to the next)  

```python
decoder_model = Model(
  [decoder_inputs] + decoder_states_inputs,
  [decoder_outputs] + decoder_states)
```  


If loading the model 

```python
training_model = load_model('training_model.h5')
# These next lines are only necessary
# because we're using a saved model:
encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
```


```python
# Building the encoder test model:
encoder_model = Model(encoder_inputs, encoder_states)

latent_dim = 256
# Building the two decoder state input layers:
decoder_state_input_hidden = Input(shape=(latent_dim,))

decoder_state_input_cell = Input(shape=(latent_dim,))

# Put the state input layers into a list:
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

# Call the decoder LSTM:
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, 
    initial_state=decoder_states_inputs)

decoder_states = [state_hidden,state_cell]
# Redefine the decoder outputs:

decoder_outputs = decoder_dense(decoder_outputs)  

# Build the decoder test model:
decoder_model = Model(
  [decoder_inputs] + decoder_states_inputs,
  [decoder_outputs] + decoder_states)


```

## The Test Function
Finally, we can get to testing our model! To do this, we need to build a function that:

- accepts a NumPy matrix representing the test English sentence input  
- uses the encoder and decoder we’ve created to generate Spanish output  

Inside the test function, we’ll run our new English sentence through the encoder model. The `.predict()` method takes in new input (as a NumPy matrix) and gives us output states that we can pass on to the decoder:  

```python
# test_input is a NumPy matrix
# representing an English sentence
states = encoder.predict(test_input)
```
  
Next, we’ll build an empty NumPy array for our Spanish translation, giving it three dimensions:
```python
# batch size: 1
# number of tokens to start with: 1
# number of tokens in our target vocabulary
target_sequence = np.zeros((1, 1, num_decoder_tokens))  
```
  
Luckily, we already know the first value in our Spanish sentence — `"<Start>"!` So we can give` "<Start>"` a value of 1 at the first timestep:
  
```python
target_sequence[0, 0, target_features_dict['<START>']] = 1.
```   

Before we get decoding, we’ll need a string where we can add our translation to, word by word:
```python 
decoded_sentence = ''  
```

This is the variable that we will ultimately return from the function.


Because we’re transferring the states from the encoder to the decoder, it’s helpful to keep track of this transfer here. Create a new variable decoder_states_value and set it equal to encoder_states_value.


```python

def decode_sequence(test_input):
  # Encode the input as state vectors:
  encoder_states_value = encoder_model.predict(test_input)
  # Set decoder states equal to encoder final states
  decoder_states_value = encoder_states_value
    
  # Generate empty target sequence of length 1:
  target_seq = np.zeros((1, 1, num_decoder_tokens))
  
  # Populate the first token of target sequence with the start token:
  target_seq[0, 0, target_features_dict['<START>']] = 1.
  
  decoded_sentence = ''

  return decoded_sentence

for seq_index in range(10):
  test_input = encoder_input_data[seq_index: seq_index + 1]
  decoded_sentence = decode_sequence(test_input)
  print('-')
  print('Input sentence:', input_docs[seq_index])
  print('Decoded sentence:', decoded_sentence)
```

    -
    Input sentence: We'll see.
    Decoded sentence: 
    -
    Input sentence: We'll see.
    Decoded sentence: 
    -
    Input sentence: We'll try.
    Decoded sentence: 
    -
    Input sentence: We've won!
    Decoded sentence: 
    -
    Input sentence: Well done.
    Decoded sentence: 
    -
    Input sentence: What's up?
    Decoded sentence: 
    -
    Input sentence: Who cares?
    Decoded sentence: 
    -
    Input sentence: Who drove?
    Decoded sentence: 
    -
    Input sentence: Who drove?
    Decoded sentence: 
    -
    Input sentence: Who is he?
    Decoded sentence: 


## Test Function (part 2)  

At long last, it’s translation time. Inside the test function, we’ll decode the sentence word by word using the output state that we retrieved from the encoder (which becomes our decoder’s initial hidden state). We’ll also update the decoder hidden state after each word so that we use previously decoded words to help decode new ones.
  
To tackle one word at a time, we need a while loop that will run until one of two things happens (we don’t want the model generating words forever):
  
-  The current token is "<END>".  
-  The decoded Spanish sentence length hits the maximum target sentence length.  
    
Inside the while loop, the decoder model can use the current target sequence (beginning with the `"<START>"` token) and the current state (initially passed to us from the encoder model) to get a bunch of possible next words and their corresponding probabilities. In Keras, it looks something like this:  
    
```python
output_tokens, new_decoder_hidden_state, new_decoder_cell_state = decoder_model.predict([target_seq] + decoder_states_value)
``` 
Next, we can use NumPy’s `.argmax()` method to determine the token (word) with the highest probability and add it to the decoded sentence:

```python
# slicing [0, -1, :] gives us a
# specific token vector within the
# 3d NumPy matrix
sampled_token_index = np.argmax(output_tokens[0, -1, :])
    
# The reverse features dictionary
# translates back from index to Spanish
sampled_token = reverse_target_features_dict[sampled_token_index]
 
decoded_sentence += " " + sampled_token
```  
    
Our final step is to update a few values for the next word in the sequence:
  
```python
# Move to the next timestep 
# of the target sequence:
target_seq = np.zeros((1, 1, num_decoder_tokens))
target_seq[0, 0, sampled_token_index] = 1.
 
# Update the states with values from
# the most recent decoder prediction:
decoder_states_value = [
    new_decoder_hidden_state,
    new_decoder_cell_state]
```  
And now we can test it all out!  
  
You may recall that, because of platform constraints here, we’re using very little data. As a result, we can only expect our model to translate a handful of sentences coherently. Luckily, you will have an opportunity to try this out on your own computer with far more data to see some much more impressive results.


```python

def decode_sequence(test_input):
  encoder_states_value = encoder_model.predict(test_input)
  decoder_states_value = encoder_states_value
  target_seq = np.zeros((1, 1, num_decoder_tokens))
  target_seq[0, 0, target_features_dict['<START>']] = 1.
  decoded_sentence = ''
  
  stop_condition = False
  while not stop_condition:
    # Run the decoder model to get possible 
    # output tokens (with probabilities) & states
    output_tokens, new_decoder_hidden_state,new_decoder_cell_state = decoder_model.predict([target_seq] + decoder_states_value)

    
    sampled_token = ""
    
    # Choose token with highest probability
    sampled_token_index = np.argmax(
    output_tokens[0, -1, :])

    sampled_token = reverse_target_features_dict[
    sampled_token_index]

    decoded_sentence += " " + sampled_token

    # Exit condition: either hit max length
    # or find stop token.
    if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
      stop_condition = True

    # Update the target sequence (of length 1).
    
    # Move to the next timestep 
    # of the target sequence:
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, sampled_token_index] = 1.
    
    # Update the states with values from
    # the most recent decoder prediction:
    decoder_states_value = [
        new_decoder_hidden_state,
        new_decoder_cell_state]


  return decoded_sentence

for seq_index in range(10):
  test_input = encoder_input_data[seq_index: seq_index + 1]
  decoded_sentence = decode_sequence(test_input)
  print('-')
  print('Input sentence:', input_docs[seq_index])
  print('Decoded sentence:', decoded_sentence)
```

    -
    Input sentence: We'll see.
    Decoded sentence:  Ya veremos .
    -
    Input sentence: We'll see.
    Decoded sentence:  Ya veremos .
    -
    Input sentence: We'll try.
    Decoded sentence:  Lo intentaremos
    -
    Input sentence: We've won!
    Decoded sentence:  ¡ Hemos ganado
    -
    Input sentence: Well done.
    Decoded sentence:  Bien hecho .
    -
    Input sentence: What's up?
    Decoded sentence:  ¿ Qué hay ? <END>
    -
    Input sentence: Who cares?
    Decoded sentence:  ¿ A quién le
    -
    Input sentence: Who drove?
    Decoded sentence:  ¿ Quién condujo
    -
    Input sentence: Who drove?
    Decoded sentence:  ¿ Quién condujo
    -
    Input sentence: Who is he?
    Decoded sentence:  ¿ Quién condujo


## Generating Text with Deep Learning Review  

Congrats! You’ve successfully built a machine translation program using deep learning with Tensorflow’s Keras API.

While the translation output may not have been quite what you expected, this is just the beginning. There are many ways you can improve this program on your own device by using a larger data set, increasing the size of the model, and adding more epochs for training.

You could also convert the one-hot vectors into word embeddings during training to improve the model. Using embeddings of words rather than one-hot vectors would help the model capture that semantically similar words might have semantically similar embeddings (helping the LSTM generalize).

You’ve learned quite a bit, even beyond the Keras implementation:

- seq2seq models are deep learning models that use recurrent neural networks like LSTMs to generate output.
- In machine translation, seq2seq networks encompass two main parts:  
 
- The encoder accepts language as input and outputs state vectors.  
- The decoder accepts the encoder’s final state and outputs possible translations.
- Teacher forcing is a method we can use to train seq2seq decoders.
- We need to mark the beginning and end of target sentences so that the decoder knows what to expect at the beginning and end of sentences.
- One-hot vectors are a way to represent a given word in a set of words wherein we use 1 to indicate the current word and 0 to indicate every other word.
- Timesteps help us keep track of where we are in a sentence.
- We can adjust batch size, which determines how many sentences we give a model at a time.
- We can also tweak dimensionality and number of epochs, which can improve results with careful tuning.
- The Softmax function converts the output of the LSTMs into a probability distribution over words in our vocabulary.  

![image.png](attachment:image.png)

![image.png](attachment:image.png)

![image.png](attachment:image.png)

![image.png](attachment:image.png)

## Off-Platform Project: Machine Translation
Use Keras models with seq2seq neural networks to build a better translation tool.

Now that you have a basic understanding of how to use Keras models and seq2seq neural networks for some pretty rudimentary machine translation, it’s time to take that code off Codecademy’s platform and make it a lot better.


## Download your dataset
Select and download a language pair dataset for your translator here. http://www.manythings.org/anki/  

We encourage you to pick a language you have some familiarity with so you can better check the accuracy of the translations you generate. If you don’t have familiarity with any of the languages listed, don’t fear! You can also use Google Translate to get a sense of how well your model is working.
  
Unzip the folder and take a look at the [your-language].txt file in a notepad or code editor. You should see sentence pairs of English and the target language you picked.




## Set Up The Code
Download and unzip the machine translation code we used in the lesson. You should have following files:
https://content.codecademy.com/programs/chatbots/seq2seq/machine_translation_project.zip


preprocessing.py
training_model.py
test_function.py
While you could also have all of the code combined into one file, we wanted to break everything down to make it easier to understand how each piece works.

You can move the `[your-language].txt` file into the same directory (folder) as the machine translation code to make it slightly easier to set up.  

## Preprocessing

- Open preprocessing.py in a code editor or IDE.
- Change data_path to the file path of [your-language].txt. If it’s in the same directory as preprocessing.py, then all you need is the file name.
- In preprocessing.py, adjust the number of lines of training data you want to work with. We’re giving you a default of 500, but depending on how much you want to tax your computer, you can go up to 123,000 lines. Whatever you choose, this should be a much higher number than what we’ve used on the Codecademy platform.
- Run the code and make sure everything works, error-free. This may take a moment because you have a lot more training data than we used on the Codecademy platform. Remember that more training data leads to better results for machine translation.
- At the end of preprocessing.py, print out list(input_features_dict.keys())[:50], reverse_target_features_dict[50], and the length of input_tokens. Does each look how you expected?


## The Training Model
- Open training_model.py. This is where the training model is built and trained.

- Change the values for the following:

- latent_dim: Choose a latent dimensionality for your model. Keras’s documentation uses 256, but you can adjust as you see fit.

- batch_size: You can choose to adjust this or not at this point. This determines how many sentences are used at a time for training.

- epochs: This should be a larger number (Keras’s documentation uses 100) so that the seq2seq model has many chances to improve. Bear in mind that a larger number of epochs will also take your computer a lot longer to process. If you don’t have the ability to leave your computer running for several hours for this project, then choose a number that is less than 100.

- Run the code to generate your model. In the terminal, you should see a summary of the model printed out. Meanwhile, you’ll also see a new file in the directory called training_model.h5. This is where your seq2seq training model is saved so that it’s quicker for you to run your code during testing.
  
Note that you may get the following error when attempting to run your program on a regular computer that uses CPU processing:  
```
OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
OMP: Hint: This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
Abort trap: 6
```

Below the import statements on training.py, there’s are a couple lines commented out, which you can uncomment to make the program run. However, if you are concerned about your computer crashing, you may want to hold off completing this project until you have access to a faster computer with GPU processing.  

## The Test Model and Function

- Open test_function.py. You’ll see that we’ve imported several variables from the preprocessing and training steps of the program.
- In the for loop at the bottom of the file, increase the range if you want to see more sentences translated — this is up to you. Of course, more sentences will increase the amount of time your computer will require for the translation.
- When you feel ready for your computer to spend awhile training and translating, run the code. This is going to take awhile — from 20 minutes to several hours, so make sure your computer is plugged in. You should see each epoch appear in the terminal as a fraction of the total number of epochs you selected. For example, 16/100 indicates the program has reached the 16th epoch out of 100 total epochs. You’ll also see the loss go down for each epoch, which is very exciting — this is the model getting more accurate!
- When your computer finishes the full process, you’ll see the translations appear. You can use a speaker of that language or Google Translate to see how accurate they are.
- Congratulations on setting up your first deep learning program locally!  

## BONUS
Some ways to improve:

- Try including more lines of training data in preprocessing.py to see if your translations improve (this may also crash the program if there are too many!).
- Try out different values for the epochs, latent_dim, and batch_size to see how they change the performance of the model.
- Add in a step to convert new text into a NumPy matrix so that you can translate new sentences that aren’t in the dataset. (This may also require handling unknown tokens.)


```python

```
