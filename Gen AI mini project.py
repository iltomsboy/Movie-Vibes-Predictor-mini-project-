from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#ASSUMPTION: I want to decide whether it’s worth watching a newly released movie (New Movie 2026) in theaters. 
# Since I’m unsure, I created the following code to analyze available data and help me determine if seeing it would be worthwhile.

#ChatGPT generated
reviews = [
    # POSITIVE
    "This movie was fantastic! A must-watch.",
    "Amazing storyline and great acting!",
    "Loved the cinematography and the soundtrack.",
    "An emotional and powerful movie experience.",
    "Great performances and a very engaging plot.",
    "The direction was excellent and very creative.",
    "One of the best movies I have seen this year.",
    "The characters were deep and well developed.",
    "A beautiful story told in a brilliant way.",
    "Highly entertaining from start to finish.",
    "The visuals were stunning and immersive.",
    "A very inspiring and well-written movie.",

    # NEGATIVE
    "I did not enjoy this movie at all.",
    "The plot was dull and predictable.",
    "Very boring and way too long.",
    "The acting was weak and unconvincing.",
    "I expected much more from this movie.",
    "Poor script and disappointing ending.",
    "The movie lacked emotion and originality.",
    "It was hard to stay interested until the end.",
    "Not worth the time, very forgettable.",
    "The story made no sense at all.",
    "Bad pacing and unnecessary scenes.",
    "A complete waste of time.",

    # MIXED / SLIGHTLY POSITIVE
    "Nice visuals but the story could have been better."
]

#ChatGPT generated
labels = [
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive", "positive",
    "positive", "positive",

    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative", "negative",
    "negative", "negative",

    "negative"
]

#converting text into numbers
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(reviews)

#split into train data and test data
x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42)

#model training >> Naive Bayes Classifier
model = MultinomialNB()
model.fit(x_train, y_train)

#model testing and evaluation
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)

#results
if accuracy > 0.8:
  print('Good vibes. Book the ticket!')
else:
  print('Needs more work!')


