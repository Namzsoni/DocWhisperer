import os
from PyPDF2 import PdfReader
import tiktoken
import openai
from dotenv import load_dotenv  # Import dotenv to load environment variables
import faiss
import numpy as np

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or "your_openai_api_key_here"

def upload_pdf(file_path, upload_dir="uploads"):
    """
    Validates and simulates uploading a PDF file by moving it to an upload directory.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not file_path.endswith(".pdf"):
        raise ValueError(f"The file {file_path} is not a PDF file.")
    
    # Ensure the upload directory exists
    os.makedirs(upload_dir, exist_ok=True)
    
    # Move the file to the upload directory
    file_name = os.path.basename(file_path)
    uploaded_path = os.path.join(upload_dir, file_name)
    os.rename(file_path, uploaded_path)
    
    print(f"File '{file_path}' uploaded successfully to '{uploaded_path}'.")
    return uploaded_path

def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    print(f"Extracting text from '{pdf_path}'...")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, max_tokens=3000, overlap_tokens=500):
    """
    Splits text into smaller chunks based on token limits with optional overlap.

    Args:
        text (str): The full text to be chunked.
        max_tokens (int): The maximum number of tokens per chunk.
        overlap_tokens (int): The number of overlapping tokens between chunks.

    Returns:
        list: A list of text chunks.
    """
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")  # Use the tokenizer for OpenAI models

    # Tokenize the text
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk))  # Decode tokens back into text
        start = end - overlap_tokens  # Move the window with overlap

    print(f"Text split into {len(chunks)} chunks.")
    return chunks

def generate_embeddings(chunks, model="text-embedding-ada-002"):
    """
    Generates embeddings for a list of text chunks using OpenAI's embedding model.

    Args:
        chunks (list): A list of text chunks.
        model (str): The OpenAI embedding model to use.

    Returns:
        list: A list of embeddings (one for each chunk).
    """
    client = openai.Client()
    embeddings = []
    response = client.embeddings.create(input=chunks, model=model)
    for data in response.data:
        #print(f"Generating embedding for chunk {i + 1}/{len(chunks)}...")
        embeddings.append(data.embedding)
    print("Embeddings generated for all chunks.")
    return embeddings

def store_embeddings_in_faiss(embeddings, chunks):
    """
    Stores embeddings in a FAISS index for similarity search.

    Args:
        embeddings (list): A list of embeddings (vectors).
        chunks (list): A list of text chunks corresponding to the embeddings.

    Returns:
        faiss.IndexFlatL2: The FAISS index containing the embeddings.
        dict: A mapping of chunk IDs to text chunks.
    """
    # Convert embeddings to a NumPy array
    embedding_dim = len(embeddings[0])  # Dimension of each embedding
    embedding_matrix = np.array(embeddings, dtype='float32')

    # Create a FAISS index
    index = faiss.IndexFlatL2(embedding_dim)  # L2 (Euclidean) distance
    index.add(embedding_matrix)  # Add embeddings to the index

    # Create a mapping of chunk IDs to text chunks
    chunk_mapping = {i: chunk for i, chunk in enumerate(chunks)}

    print(f"Stored {len(embeddings)} embeddings in the FAISS index.")
    return index, chunk_mapping

def process_query(query, index, chunk_mapping, model="text-embedding-ada-002", top_k=3):
    """
    Processes a user query by retrieving the most relevant chunks and generating an answer.

    Args:
        query (str): The user's question.
        index (faiss.IndexFlatL2): The FAISS index containing the embeddings.
        chunk_mapping (dict): A mapping of chunk IDs to text chunks.
        model (str): The OpenAI embedding model to use for the query.
        top_k (int): The number of top chunks to retrieve.

    Returns:
        str: The generated answer.
    """
    # Generate embedding for the query
    print("Generating embedding for the query...")
    client = openai.Client()
    query_embedding = client.embeddings.create(input=query, model=model).data[0].embedding

    # Convert query embedding to NumPy array
    query_vector = np.array(query_embedding, dtype='float32').reshape(1, -1)

    # Search the FAISS index for the top_k most similar chunks
    print(f"Searching for the top {top_k} most relevant chunks...")
    distances, indices = index.search(query_vector, top_k)

    # Retrieve the corresponding chunks
    retrieved_chunks = [chunk_mapping[idx] for idx in indices[0] if idx != -1]
    print(f"Retrieved {len(retrieved_chunks)} relevant chunks.")

    # Combine the retrieved chunks as context
    context = "\n".join(retrieved_chunks)

    # Generate an answer using the language model
    print("Generating an answer using the language model...")
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

def main():
    """
    Main function to simulate the workflow.
    """
    # Simulate uploading a PDF
    pdf_path = input("Please enter the path to the PDF file you want to upload: ")
    try:
        uploaded_file_path = upload_pdf(pdf_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    # Extract text from the uploaded PDF
    try:
        pdf_text = extract_text_from_pdf(uploaded_file_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("\nExtracted Text:")
    print(pdf_text[:500])  # Print the first 500 characters of the extracted text

    # Chunk the extracted text
    chunks = chunk_text(pdf_text)
    print("\nFirst Chunk:")
    print(chunks[0])  # Print the first chunk for verification
    print(f"\nTotal Chunks Created: {len(chunks)}")

    # Generate embeddings for the chunks
    try:
        embeddings = generate_embeddings(chunks)
        print(f"\nGenerated {len(embeddings)} embeddings.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    # Store embeddings in FAISS
    try:
        index, chunk_mapping = store_embeddings_in_faiss(embeddings, chunks)
        print("\nEmbeddings stored in FAISS index.")
    except Exception as e:
        print(f"Error storing embeddings in FAISS: {e}")
        return

    # Process user queries
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        try:
            answer = process_query(query, index, chunk_mapping)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()