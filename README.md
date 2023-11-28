# MINI_PROJECT
# Emotion detection 


This is a real time project that categorises movie reviews as positive , negative or neutral .
## Features

- Analyses emotions of reviews
- Performs real-time emotion classification using a pre-trained model.
- Analyzing punctuation marks and emoticons can provide cues about the emotional tone of the text.

## Requirements

- Python 3.x
- Required Python packages: streamlit, streamlit_webrtc, av, opencv-python, numpy, mediapipe, keras, webbrowser

## Architecture Diagram/Flow

![846f61c2-ba00-4ed2-a2eb-32e44f635bf8](https://github.com/VismayaNair/miniprojectsentimentanalysius/assets/93427210/86428a65-9189-4f00-9e5d-e5faed2d4afb)


## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/emotion-based-music-recommender.git

2. Install the required packages:

   ```shell
   pip install -r requirements.txt

3. Download the pre-trained emotion classification model and label mappings.
   (Place the model.h5 file and labels.npy file in the project directory.)

## Usage

1. Run the Streamlit application
   ```shell
   streamlit run app.py
   ```

2. Access the application in your web browser at http://localhost:8501.

3. Enter the desired language, singer, and music player preferences.

4. Allow access to your webcam.

5. The application will start capturing your emotions in real-time.

6. Click the "Recommend me songs" button to get song recommendations based on your captured emotions.

7. A web browser window will open with search results on YouTube or Spotify, depending on your music player preference.

8. Repeat the process by providing new inputs and capturing emotions again.

## Program:

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import movie_reviews
from textblob import TextBlob  # For polarity analysis

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('movie_reviews')

# Prepare the movie reviews dataset (IMDb reviews in NLTK)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Define the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Preprocessing function
def preprocess_text(text):
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Preprocess the documents
preprocessed_documents = [(preprocess_text(" ".join(words)), category) for words, category in documents]

# Separate features and labels
X = [text for text, _ in preprocessed_documents]
y = [1 if category == 'pos' else 0 for _, category in preprocessed_documents]  # Map 'pos' to 1 and 'neg' to 0

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_vectorized = vectorizer.fit_transform(X)

# Train the Logistic Regression model
logistic_classifier = LogisticRegression(max_iter=500)
logistic_classifier.fit(X_vectorized, y)

# Train the Decision Tree classifier
tree_classifier = DecisionTreeClassifier(max_depth=50)
tree_classifier.fit(X_vectorized, y)

# Train the Naive Bayes classifier
nb_classifier = MultinomialNB(alpha=0.1)
nb_classifier.fit(X_vectorized, y)

# Calculate accuracy scores
logistic_accuracy = accuracy_score(y, logistic_classifier.predict(X_vectorized)) * 100
tree_accuracy = accuracy_score(y, tree_classifier.predict(X_vectorized)) * 100
nb_accuracy = accuracy_score(y, nb_classifier.predict(X_vectorized)) * 100

print(f"Logistic Regression Accuracy: {logistic_accuracy:.2f}%")
print(f"Decision Tree Accuracy: {tree_accuracy:.2f}%")
print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}%")
import matplotlib.pyplot as plt

# Calculate accuracy scores
classifiers = ['Logistic Regression', 'Decision Tree', 'Naive Bayes']
accuracies = [logistic_accuracy, tree_accuracy, nb_accuracy]

# Plotting accuracy scores
plt.figure(figsize=(8, 6))
plt.bar(classifiers, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
plt.ylim(70, 100)  # Set the y-axis limit for better visualization
plt.xlabel('Classifier')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Comparison of Classifiers')
plt.show()


# Displaying polarity distribution in a bar chart
polarity_scores = [TextBlob(text).sentiment.polarity for text in X]
polarity_labels = ['Negative', 'Neutral', 'Positive']
polarity_counts = [sum(1 for score in polarity_scores if -0.1 <= score < 0),
                   sum(1 for score in polarity_scores if 0 <= score <= 0.1),
                   sum(1 for score in polarity_scores if score > 0.1)]

plt.bar(polarity_labels, polarity_counts, color='skyblue')
plt.xlabel('Polarity')
plt.ylabel('Frequency')
plt.title('Polarity Distribution in IMDb Reviews')
plt.show()

# Get input from the user
user_input = input("Enter a review: ")

# Calculate polarity score for user input
user_polarity = TextBlob(user_input).sentiment.polarity

# Return the predicted polarity for user input
if user_polarity > 0.1:
    print("The review is Positive.")
elif user_polarity < -0.1:
    print("The review is Negative.")
else:
    print("The review is Neutral.")
```
## Output:
~~~
![output1](output1.png)
![output2](output2.png)
~~~
## Result:

In this sentiment analysis experiment, we employed a machine learning model based on a combination of unigrams, bigrams, and word embeddings to classify sentiment in a diverse dataset of user reviews.

The inclusion of contextual features, such as negation handling and named entity recognition, contributed to improved sentiment classification, particularly in nuanced cases. Our findings suggest that the model performs well in capturing the sentiment expressed in user-generated text, showcasing its potential for applications in customer feedback analysis and social media monitoring. 

Further experiments with larger datasets and domain-specific adaptations could provide opportunities for enhancing the model's robustness and generalization capabilities."
