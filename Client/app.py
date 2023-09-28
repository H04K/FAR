import numpy as np
from PIL import Image
import tritonclient.http as httpclient
import requests
import argparse
from transformers import BertTokenizer,AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
tokenizer_sf = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
client = httpclient.InferenceServerClient(url="localhost:8000")
import time
import streamlit as st


def compute_similarity(embedding1, embedding2):
    """
    Calcule la similarité cosinus entre deux embeddings.

    Args:
        embedding1 (numpy.ndarray): Le premier embedding.
        embedding2 (numpy.ndarray): Le deuxième embedding.

    Returns:
        float: La similarité cosinus entre les deux embeddings.
    """
    # Assurez-vous que les embeddings ont la même forme (même dimension)
    if embedding1.shape != embedding2.shape:
        raise ValueError("Les embeddings doivent avoir la même dimension.")

    # Normalisez les embeddings (optionnel mais recommandé)
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)

    # Calculez la similarité cosinus
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))

    return similarity[0][0]


def encode_sf(text):
    senteces= [text]
    encoded_input = tokenizer(senteces, padding=True, truncation=True, return_tensors='pt')
    return encoded_input
def infer_sf(text,model_name):
   
    np_input_data = np.asarray([str.encode(text)])
    # Set Inputs
    input_tensors = [
        httpclient.InferInput("input_data", [1], datatype="BYTES"),
    ]
    input_tensors[0].set_data_from_numpy(np_input_data.reshape([1]), binary_data=False)
    # Set outputs
    outputs_sf = [
        httpclient.InferRequestedOutput("embedding")
    ]
    # Query
    query_response = client.infer(model_name=model_name,inputs=input_tensors,outputs=outputs_sf)
    # Affichez le temps écoulé
    # Output    
    embedding = query_response.as_numpy("embedding")
    return  embedding


def infer(text,model_name):
    
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    print("Tokenized text:", tokenized_text)

    # Convert tokens to input_ids
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    print("Input IDs:", input_ids)
    print(type(input_ids))
    input_data = np.array(input_ids, dtype=np.int32)
    print(input_data)
    print(type(input_data))
    
    # Set Inputs
    input_tensors = [
        httpclient.InferInput("input_data", input_data.shape, datatype="INT32"),
    ]
    input_tensors[0].set_data_from_numpy(input_data)

    # Set outputs
    outputs = [
        httpclient.InferRequestedOutput("embedding")
    ]
    print(len(input_tensors))
    print(len(outputs))
    query_response = client.infer(model_name=model_name, inputs=input_tensors, outputs=outputs)

    return query_response

def main():
    st.title("Text Embedding and Similarity")

    model_name = st.selectbox("Select Model", ["LaBSE", "python_bert_sf", "python_bert"])

    if model_name == "LaBSE":
        text1 = st.text_input("Enter Text 1", "I love Paris")
        text2 = st.text_input("Enter Text 2", "I love France")

        if st.button("Compute Similarity"):
            start_time = time.time()
            embedding_1 = infer_sf(text1, model_name)
            embedding_2 = infer_sf(text2, model_name)
            similarity = compute_similarity(embedding_1, embedding_2)
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Similarity between Text 1 and Text 2: {similarity * 100:.2f}%")
            st.write(f"Text 1 Length: {len(text1)}")
            st.write(f"Text 2 Length: {len(text2)}")
            st.write(f"Inference Time: {elapsed_time:.2f} seconds")

    elif model_name == "python_bert_sf":
        text = st.text_input("Enter Text", "I love Paris")

        if st.button("Compute Embedding"):
            tokenized_text = tokenizer_sf.tokenize(text)
            input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
            input_data = np.array(input_ids, dtype=np.int32)
            query_response = infer(text, model_name)
            embedding = query_response.as_numpy("output_data")
            st.write("Embedding Shape:", embedding.shape)
            st.write("Embedding:", embedding)

    elif model_name == "python_bert":
        st.write("Embedding and Similarity Calculation for Predefined Texts")

    else:
        st.write("Please select a valid model.")

if __name__ == "__main__":
    main()