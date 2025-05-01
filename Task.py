from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd
import os

# Create a Spark session
spark = SparkSession.builder \
    .appName("Fake News Classification") \
    .getOrCreate()

# Function to create output directory if it doesn't exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Create an output directory for CSV files
output_dir = "output"
ensure_dir(output_dir)

##########################
# TASK 1: Load & Basic Exploration
##########################
print("\n=== TASK 1: Load & Basic Exploration ===")

# Load the CSV file
news_df = spark.read.option("header", "true").option("inferSchema", "true").csv("fake_news_sample.csv")

# Create a temporary view
news_df.createOrReplaceTempView("news_data")

# Show first 5 rows
print("First 5 rows:")
news_df.show(5, truncate=False)

# Count total number of articles
article_count = news_df.count()
print(f"Total number of articles: {article_count}")

# Retrieve distinct labels
distinct_labels = news_df.select("label").distinct().collect()
print("Distinct labels:")
for label in distinct_labels:
    print(label["label"])

# Save DataFrame to CSV for Task 1
task1_output_path = os.path.join(output_dir, "task1_output.csv")
news_df.toPandas().to_csv(task1_output_path, index=False)
print(f"Task 1 output saved to {task1_output_path}")

##########################
# TASK 2: Text Preprocessing
##########################
print("\n=== TASK 2: Text Preprocessing ===")

# Convert text to lowercase
from pyspark.sql.functions import lower
news_df = news_df.withColumn("text_lower", lower(news_df["text"]))

# Tokenize text
tokenizer = Tokenizer(inputCol="text_lower", outputCol="words")
tokenized_df = tokenizer.transform(news_df)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned_df = remover.transform(tokenized_df)

# Create a temporary view (optional)
cleaned_df.createOrReplaceTempView("cleaned_news")

# Select relevant columns
preprocessed_df = cleaned_df.select("id", "title", "filtered_words", "label")

# Show sample of preprocessed data
print("Preprocessed data sample:")
preprocessed_df.show(3, truncate=False)

# Save to CSV for Task 2
task2_output_path = os.path.join(output_dir, "task2_output.csv")
# Convert to pandas for easier CSV handling (especially with array columns)
preprocessed_df_pandas = preprocessed_df.toPandas()
preprocessed_df_pandas.to_csv(task2_output_path, index=False)
print(f"Task 2 output saved to {task2_output_path}")

##########################
# TASK 3: Feature Extraction
##########################
print("\n=== TASK 3: Feature Extraction ===")

# TF-IDF Vectorization
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
featurized_df = hashingTF.transform(preprocessed_df)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_df)
tfidf_df = idf_model.transform(featurized_df)

# Label Indexing
labelIndexer = StringIndexer(inputCol="label", outputCol="label_index")
labelIndexer_model = labelIndexer.fit(tfidf_df)
final_df = labelIndexer_model.transform(tfidf_df)

# Show sample of feature-extracted data
print("Feature extracted data sample:")
final_df.select("id", "filtered_words", "features", "label_index").show(3, truncate=False)

# Save to CSV for Task 3
task3_output_path = os.path.join(output_dir, "task3_output.csv")
# Note: Converting feature vectors to CSV is challenging, this is simplified
final_df_for_csv = final_df.select("id", "label_index")
final_df_for_csv.toPandas().to_csv(task3_output_path, index=False)
print(f"Task 3 output saved to {task3_output_path}")

##########################
# TASK 4: Model Training
##########################
print("\n=== TASK 4: Model Training ===")

# Split the data
(training_data, test_data) = final_df.randomSplit([0.8, 0.2], seed=42)
print(f"Training data size: {training_data.count()}")
print(f"Test data size: {test_data.count()}")

# Train Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label_index", maxIter=10)
lr_model = lr.fit(training_data)

# Make predictions
predictions = lr_model.transform(test_data)

# Show sample predictions
print("Sample predictions:")
predictions.select("id", "title", "label_index", "prediction").show(5)

# Save to CSV for Task 4
task4_output_path = os.path.join(output_dir, "task4_output.csv")
predictions.select("id", "title", "label_index", "prediction").toPandas().to_csv(task4_output_path, index=False)
print(f"Task 4 output saved to {task4_output_path}")

##########################
# TASK 5: Evaluate the Model
##########################
print("\n=== TASK 5: Model Evaluation ===")

# Evaluate model performance
evaluator = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Calculate F1 Score
evaluator.setMetricName("f1")
f1_score = evaluator.evaluate(predictions)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")

# Save results to CSV
evaluation_results = pd.DataFrame([
    ["Accuracy", accuracy],
    ["F1 Score", f1_score]
], columns=["Metric", "Value"])

task5_output_path = os.path.join(output_dir, "task5_output.csv")
evaluation_results.to_csv(task5_output_path, index=False)
print(f"Task 5 output saved to {task5_output_path}")

print("\n=== All tasks completed successfully! ===")
spark.stop()