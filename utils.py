import faiss
from PIL import Image
from datasets import load_dataset
import streamlit as st
import numpy as np

@st.cache_data()
def load_faiss_index(index_path):
    index = faiss.read_index(index_path)
    return index

@st.cache_data()
def get_dataset(dataset_name, local_path):
    dataset = load_dataset(dataset_name, cache_dir=local_path)
    return dataset

def retrieve_similar_images(query, model, index, top_k=3):

    # Image Query: Yet to implement
    if query.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        query = Image.open(query)

    query_features = model.encode(query)
    query_features = query_features.astype(np.float32).reshape(1, -1)

    distances, indices = index.search(query_features, top_k)

    return query, distances[0], indices[0]