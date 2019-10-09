import pandas as pd

with open('museo_annotato2.csv', encoding='ANSI') as f:
  dataset = pd.read_csv(f, sep=';', error_bad_lines = False, header= None)


dataset = dataset.iloc[:, [0, 1, 4]]
dataset = dataset.rename(index=str, columns={ 0: "sentiment", 1: "aspect_category", 4: "review"})
#dataset = dataset.rename(index=str, columns={ 0: "sentiment", 1: "aspect_category", 2: "review"})
(dataset.head(5))

import spacy
nlp = spacy.load('en_core_web_sm')

dataset.review = dataset.review.str.lower()

aspect_terms = []
for review in nlp.pipe(dataset.review):
    chunks = [chunk.root.text for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']
    aspect_terms.append(' '.join(chunks))
dataset['aspect_terms'] = aspect_terms
(dataset.head(10))

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation

aspect_categories_model = Sequential()
aspect_categories_model.add(Dense(512, input_shape=(6000,), activation='relu'))
aspect_categories_model.add(Dense(12, activation='softmax'))
aspect_categories_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.preprocessing.text import Tokenizer

vocab_size = 6000 # We set a maximum size for the vocabulary
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(dataset.review)
aspect_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(dataset.aspect_terms))

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

label_encoder = LabelEncoder()
integer_category = label_encoder.fit_transform(dataset.aspect_category.fillna('0'))
dummy_category = to_categorical(integer_category)

new_review = "The Egyptian collection is ugly"

chunks = [(chunk.root.text) for chunk in nlp(new_review).noun_chunks if chunk.root.pos_ == 'NOUN']
new_review_aspect_terms = ' '.join(chunks)
new_review_aspect_tokenized = tokenizer.texts_to_matrix([new_review_aspect_terms])

new_review_category = label_encoder.inverse_transform(aspect_categories_model.predict_classes(new_review_aspect_tokenized))
#print(new_review_category)

sentiment_terms = []
for review in nlp.pipe(dataset['review']):
        if review.is_parsed:
            sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
        else:
            sentiment_terms.append('')
dataset['sentiment_terms'] = sentiment_terms
(dataset.head(10))

sentiment_model = Sequential()
sentiment_model.add(Dense(512, input_shape=(6000,), activation='relu'))
sentiment_model.add(Dense(4, activation='softmax'))
sentiment_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

sentiment_tokenized = pd.DataFrame(tokenizer.texts_to_matrix(dataset.sentiment_terms))

label_encoder_2 = LabelEncoder()
integer_sentiment = label_encoder_2.fit_transform(dataset.sentiment.fillna('0'))
dummy_sentiment = to_categorical(integer_sentiment)

sentiment_model.fit(sentiment_tokenized, dummy_sentiment, epochs=5, verbose=1)

new_review = "This Egyptian collection is cool"

chunks = [(chunk.root.text) for chunk in nlp(new_review).noun_chunks if chunk.root.pos_ == 'NOUN']
new_review_aspect_terms = ' '.join(chunks)
new_review_aspect_tokenized = tokenizer.texts_to_matrix([new_review_aspect_terms])

new_review_category = label_encoder_2.inverse_transform(sentiment_model.predict_classes(new_review_aspect_tokenized))
#print(new_review_category)

test_reviews = [
    "Good, nice staff",
    "The staff was very pleasant.",
    "The price was too high for the museum",
    "The Egyptian collection was terrible",
    "This museum is boring"
]

# Aspect preprocessing
test_reviews = [review.lower() for review in test_reviews]
test_aspect_terms = []
for review in nlp.pipe(test_reviews):
    chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']
    test_aspect_terms.append(' '.join(chunks))
test_aspect_terms = pd.DataFrame(tokenizer.texts_to_matrix(test_aspect_terms))

# Sentiment preprocessing
test_sentiment_terms = []
for review in nlp.pipe(test_reviews):
    if review.is_parsed:
        test_sentiment_terms.append(' '.join([token.lemma_ for token in review if (
                    not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))
    else:
        test_sentiment_terms.append('')
test_sentiment_terms = pd.DataFrame(tokenizer.texts_to_matrix(test_sentiment_terms))

# Models output
test_aspect_categories = label_encoder.inverse_transform(aspect_categories_model.predict_classes(test_aspect_terms))
test_sentiment = label_encoder_2.inverse_transform(sentiment_model.predict_classes(test_sentiment_terms))
for i in range(5):
    print(
        "Review " + str(i + 1) + " is expressing a  " + test_sentiment[i] + " opinion about " + test_aspect_categories[
            i])

