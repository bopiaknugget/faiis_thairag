    from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, Response
import json
import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)
logger.info("Logger initialized for Flask application")



# Configuration for FAISS and vLLM hosts
FAISS_DIMENSION = 1024  # Adjust based on your model's embedding dimension
FAISS_INDEX_PATH = 'faiss_index.index'  # Path to save/load the FAISS index
#VLLM_HOST = os.environ.get('VLLM_HOST', '172.17.0.1:8000')


# Initialize or load FAISS index
def initialize_faiss_index():
    if os.path.exists(FAISS_INDEX_PATH):
        # Load existing index if available
        index = faiss.read_index(FAISS_INDEX_PATH)
    else:
        # Create a new FAISS index
        index = faiss.IndexFlatL2(FAISS_DIMENSION)
    return index


# Initialize FAISS index
faiss_index = initialize_faiss_index()
documents = []

logger.info("Successfully initialized FAISS index.")

# Flask app setup
app = Flask(__name__)
#Use LaBSE for  effective embedding
embedding_model = SentenceTransformer("sentence-transformers/LaBSE")
# Function to generate embeddings
def generate_embedding(text):
    embedding = embedding_model.encode(text)  # Returns a NumPy array
    return embedding.astype("float32")  # Ensuring FAISS compatibility


# Rerank documents based on cosine similarity
def rerank_documents(query_embedding, document_embeddings):
    query_embedding = query_embedding.reshape(1, -1)
    similarities = cosine_similarity(query_embedding, document_embeddings)
    ranked_documents = sorted(enumerate(similarities.flatten()), key=lambda x: x[1], reverse=True)
    return ranked_documents


# Flask route for index page
@app.route("/", methods=["GET"])
def index():
    return "Welcome to OpenThaiRAG with FAISS!", 200


# Flask route for indexing text
@app.route("/index", methods=["POST"])
def index_text():
    try:
        # Get text from request
        data = request.get_json()
        text = data.get("text")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Generate embedding for the text
        embedding = generate_embedding(text)

        # Add document to FAISS and documents list
        faiss_index.add(np.array([embedding]))
        documents.append({"text": text, "embedding": embedding.tolist()})

        # Save index for persistence
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        doc_id = len(documents) - 1  # Using list index as document ID

        return jsonify({
            "message": "Text indexed successfully",
            "id": doc_id
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Flask route for deleting indexed documents
@app.route("/delete/<int:doc_id>", methods=["DELETE"])
def delete_documents(doc_id):
    try:
        # Since FAISS doesn't support direct deletion, we clear and re-add remaining documents
        if doc_id == '*':
            faiss_index.reset()
            documents.clear()
            message = "All documents deleted successfully"
        else:
            if 0 <= doc_id < len(documents):
                documents.pop(doc_id)
                faiss_index.reset()
                for doc in documents:
                    faiss_index.add(np.array([doc['embedding']]))
                message = f"Document with id {doc_id} deleted successfully"
            else:
                return jsonify({"error": "Invalid document ID"}), 400

        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        return jsonify({"message": message}), 200

    except Exception as e:
        logger.error(f"Error deleting documents: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Flask route for listing indexed documents
@app.route("/list", methods=["GET"])
def list_documents():
    try:
        # Get query parameters
        query = request.args.get('query', '')
        limit = int(request.args.get('limit', 10))
        offset = int(request.args.get('offset', 0))

        # Filter documents by query and paginate
        filtered_docs = [doc for doc in documents if query.lower() in doc['text'].lower()]
        paginated_docs = filtered_docs[offset:offset + limit]

        # Prepare the response
        documents_response = [
            {"id": i, "text": doc['text'][:50] + "...", "embedding": doc['embedding']}
            for i, doc in enumerate(paginated_docs)
        ]

        return jsonify({
            "message": "Documents retrieved successfully",
            "documents": documents_response,
            "total": len(filtered_docs),
            "offset": offset,
            "limit": limit
        }), 200

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Flask route for handling user queries
@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("prompt", "")
    
    # Initialize conversation history
    conversation_history = ""
    
    # Loop to continue conversation based on user inputs
    while True:
        # Generate query embedding
        query_embedding = generate_embedding(query_text)

        # Perform similarity search with FAISS
        D, I = faiss_index.search(np.array([query_embedding]), k=10)
        retrieved_documents = [documents[i] for i in I[0] if i != -1]

        # Re-rank retrieved documents
        document_embeddings = [doc['embedding'] for doc in retrieved_documents]
        ranked_indices = rerank_documents(query_embedding, np.array(document_embeddings))
        top_documents = [retrieved_documents[i] for i, _ in ranked_indices[:3]]

        
     # Get the system prompt from the environment variable or use the default
        system_prompt = os.environ.get(
            'SYSTEM_PROMPT',
            'คุณคือ OpenThaiGPT พัฒนาโดยสมาคมผู้ประกอบการปัญญาประดิษฐ์ประเทศไทย (AIEAT)'
        )

        # Construct a detailed system message
        system_message = (
            f"คุณคือผู้ช่วย AI ที่มีความเชี่ยวชาญและซื่อสัตย์ {system_prompt} "
            "งานของคุณคือการอ่านเอกสารที่ให้มาและตอบคำถามตามข้อมูลในเอกสารนั้น "
            "หากคุณไม่พบคำตอบในเอกสาร ให้แจ้งผู้ใช้ว่าคุณไม่สามารถหาคำตอบได้จากข้อมูลที่มี"
        )

        # Combine the top documents into a single text block
        documents_text = "\n\n".join([doc.get('text') for doc in top_documents])

        # Construct the user prompt with explicit instructions
        prompt = (
            f"จากเอกสารต่อไปนี้:\n\n{documents_text}\n\n"
            f"กรุณาตอบคำถามต่อไปนี้โดยอ้างอิงจากข้อมูลในเอกสารเท่านั้น: {query_text}"
        )

        # Build the final prompt in the chat format
        prompt_chatml = (
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        logger.info(f"Prompt: {prompt_chatml}")

        # Make the API request
        response = requests.post(
            'https://api.aieat.or.th/v1/completions',
            json={
                "model": ".",  # Replace with the appropriate model identifier
                "prompt": prompt_chatml,
                "max_tokens": data.get("max_tokens", 1024),
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        
        # Extract and decode the response text for Thai language
        response_text = response.json().get("choices", [{}])[0].get("text", "")
        
        # Append response to conversation history and output it
        conversation_history += f"\nUser: {query_text}\nAssistant: {response_text}\n"
        
        # Return response to user and prepare for next input
        print(f"Assistant: {response_text}")
        query_text = input("User: ")  # Get next user input (this is for looped, conversational mode)

        # Break condition if user decides to end conversation
        if query_text.lower() in ["exit", "quit"]:
            break
    
    return {"conversation_history": conversation_history}


# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
