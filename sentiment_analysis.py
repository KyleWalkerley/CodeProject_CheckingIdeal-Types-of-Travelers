import pandas as pd

# # Load sentiment data from reviews
# reviews_sentiment = pd.read_csv("data/CommentsReview_output.csv", encoding='latin1')

# # Quick shape and column check
# print("✅ Loaded CommentsReview_output:", reviews_sentiment.shape)

# # Step 1: Basic sentiment averages
# print("\n📊 Average sentiment scores:")
# print(reviews_sentiment[['Overall Positive', 'Overall Neutral', 'Overall Negative']].mean())

# # Step 2: Count reviews by dominant sentiment
# def dominant_sentiment(row):
#     scores = {
#         'Positive': row['Overall Positive'],
#         'Neutral': row['Overall Neutral'],
#         'Negative': row['Overall Negative']
#     }
#     return max(scores, key=scores.get)

# reviews_sentiment['dominant_sentiment'] = reviews_sentiment.apply(dominant_sentiment, axis=1)

# print("\n🔍 Dominant sentiment breakdown:")
# print(reviews_sentiment['dominant_sentiment'].value_counts())

# Press ctrl /

# Load structured survey data
survey = pd.read_csv("data/QuestionResponse.csv", encoding='latin1')
print("✅ Loaded QuestionResponse:", survey.shape)

# Step 1: Preview available question topics
print("\n🧠 Survey topics:")
print(survey['voc_questiontopicname'].value_counts())

# Step 2: Convert msfp_response to numeric (ignore errors)
survey['msfp_response'] = pd.to_numeric(survey['msfp_response'], errors='coerce')

# Step 3: Average score per question topic
print("\n📊 Average response score per topic:")
print(survey.groupby('voc_questiontopicname')['msfp_response'].mean().round(2))

# Step 4: Count of valid (non-NaN) responses per topic
print("\n🧮 Valid response counts per topic:")
print(survey.groupby('voc_questiontopicname')['msfp_response'].count())