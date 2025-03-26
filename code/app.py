from flask import Flask, request, jsonify, render_template
import os
from PyPDF2 import PdfReader
import openai
import faiss
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)

# Backend functions (reuse your existing functions)
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, max_tokens=3000, overlap_tokens=500):
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk))
        start = end - overlap_tokens
    return chunks

def generate_embeddings(chunks, model="text-embedding-ada-002"):
    client = openai.Client()
    embeddings = []
    response = client.embeddings.create(input=chunks, model=model)
    for data in response.data:
        embeddings.append(data.embedding)
    return embeddings

def store_embeddings_in_faiss(embeddings, chunks):
    embedding_dim = len(embeddings[0])
    embedding_matrix = np.array(embeddings, dtype='float32')
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embedding_matrix)
    chunk_mapping = {i: chunk for i, chunk in enumerate(chunks)}
    return index, chunk_mapping

def process_query(query, index, chunk_mapping, model="text-embedding-ada-002", top_k=3):
    client = openai.Client()
    query_embedding = client.embeddings.create(input=query, model=model).data[0].embedding
    query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    retrieved_chunks = [chunk_mapping[idx] for idx in indices[0] if idx != -1]
    context = "\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # You can use GPT-3.5 or GPT-4
        messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
                ],
        max_tokens=300,
        temperature=0.7
    )
    answer = response.choices[0].message.content.strip()
    return answer

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')  # HTML file for the frontend

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(file)

    # Chunk the extracted text
    chunks = chunk_text(pdf_text)
    

    # Generate embeddings for the chunks
    embeddings = generate_embeddings(chunks)

    # Store embeddings in FAISS
    index, chunk_mapping = store_embeddings_in_faiss(embeddings, chunks)

    # Save the index and chunk mapping in memory for querying
    app.config['index'] = index
    app.config['chunk_mapping'] = chunk_mapping

    return jsonify({"message": "PDF processed successfully", "chunks": len(chunks)})

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    index = app.config.get('index')
    chunk_mapping = app.config.get('chunk_mapping')

    if index is None or chunk_mapping is None:
        return jsonify({"error": "No PDF has been processed yet"}), 400

    # Process the query
    answer = process_query(query, index, chunk_mapping)

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)