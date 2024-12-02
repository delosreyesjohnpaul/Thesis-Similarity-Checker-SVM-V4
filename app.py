from flask import Flask, render_template, request, send_from_directory
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
import os
from werkzeug.utils import secure_filename
import pymysql
import re
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# MySQL database connection
def get_database_data():
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='similaritydb'
    )
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM similaritydataset")
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    data = pd.DataFrame(rows, columns=columns)
    connection.close()
    return data

# Fetch and preprocess the dataset
data = get_database_data()
preprocessed_texts = tfidf_vectorizer.transform(data['plagiarized_text'].fillna(""))

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

def get_snippets(source_text, input_text, ngram_size=5, context_window=10):
    source_text_clean = preprocess_text(source_text)
    input_text_clean = preprocess_text(input_text)

    source_words = source_text_clean.split()
    input_words = input_text_clean.split()

    input_ngrams = set([' '.join(input_words[i:i + ngram_size]) for i in range(len(input_words) - ngram_size + 1)])
    source_ngrams = [' '.join(source_words[i:i + ngram_size]) for i in range(len(source_words) - ngram_size + 1)]

    matching_snippets = []
    for i, ngram in enumerate(source_ngrams):
        if ngram in input_ngrams:
            start_index = max(0, i - context_window)
            end_index = min(len(source_words), i + ngram_size + context_window)
            snippet = ' '.join(source_words[start_index:end_index])
            matching_snippets.append(snippet)

    return sorted(set(matching_snippets), key=lambda snippet: source_text_clean.find(snippet))

def detect(input_text):
    if not input_text.strip():
        return "No text provided", [], 0, 100

    input_text = preprocess_text(input_text)
    vectorized_text = tfidf_vectorizer.transform([input_text])
    prediction = model.predict(vectorized_text)

    if prediction[0] == 0:
        return "No Similarity Detected", [], 0, 100

    cosine_similarities = cosine_similarity(vectorized_text, preprocessed_texts)[0]
    plagiarism_sources = []

    threshold = 0.35
    total_similarity = 0
    for i, similarity in enumerate(cosine_similarities):
        if similarity > threshold:
            total_similarity += similarity
            plagiarism_percentage = round(similarity * 100, 2)
            source_title = data['source_text'].iloc[i]
            source_text = data['plagiarized_text'].iloc[i]
            matching_snippets = get_snippets(source_text, input_text)
            
            if not matching_snippets and plagiarism_percentage >= 30:
                matching_snippets = [source_text[:200] + '...']

            plagiarism_sources.append((source_title, plagiarism_percentage, matching_snippets))

    plagiarism_sources.sort(key=lambda x: x[1], reverse=True)
    total_plagiarism_percentage = min(round((total_similarity / len(cosine_similarities)) * 100, 2), 100)
    unique_percentage = 100 - total_plagiarism_percentage
    detection_result = "Similarity Detected" if plagiarism_sources else "No Similarity Detected"
    return detection_result, plagiarism_sources, total_plagiarism_percentage, unique_percentage

def extract_text_from_file(file):
    text = ""
    if file.filename.endswith('.pdf'):
        reader = PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    elif file.filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    return text.strip()

def plot_pie_chart(plagiarism_percentage, unique_percentage):
    labels = ['Plagiarized', 'Unique']
    sizes = [plagiarism_percentage, unique_percentage]
    colors = ['#ff9999', '#66b3ff']

    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    pie_chart_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pie_chart.png')
    plt.savefig(pie_chart_path)
    plt.close()
    return pie_chart_path

def plot_similarity_graph(plagiarism_sources):
    if not plagiarism_sources:
        return None

    sources = [source[0] for source in plagiarism_sources]
    percentages = [source[1] for source in plagiarism_sources]

    max_sources = 10
    if len(sources) > max_sources:
        sources = sources[:max_sources]
        percentages = percentages[:max_sources]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(sources, percentages, color='skyblue')
    plt.xlabel('Sources')
    plt.ylabel('Plagiarism Percentage (%)')
    plt.title('Similarity Index')
    plt.xticks(rotation=45, ha='right')

    for bar, percentage in zip(bars, percentages):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{percentage}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    similarity_graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'similarity_graph.png')
    plt.savefig(similarity_graph_path)
    plt.close()
    return similarity_graph_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    input_text = request.form.get('text', "").strip()
    files = request.files.getlist("files[]")

    for file in files:
        if file and (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            input_text += "\n" + extract_text_from_file(file)

    word_count = len(input_text.split())
    detection_result, plagiarism_sources, plagiarism_percentage, unique_percentage = detect(input_text)
    pie_chart_path = plot_pie_chart(plagiarism_percentage, unique_percentage)

    similarity_graph_path = None
    if plagiarism_sources:
        similarity_graph_path = plot_similarity_graph(plagiarism_sources)

    return render_template('index.html', 
                           result=detection_result, 
                           plagiarism_sources=plagiarism_sources, 
                           word_count=word_count,
                           total_results=len(plagiarism_sources), 
                           plagiarism_percentage=plagiarism_percentage, 
                           unique_percentage=unique_percentage,
                           pie_chart_path=pie_chart_path,
                           similarity_graph_path=similarity_graph_path)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
