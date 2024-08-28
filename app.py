import json
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from utils import load_faiss_index, get_dataset, retrieve_similar_images
from sentence_transformers import SentenceTransformer

# Favicon and Title
st.set_page_config(page_title="VectorVision - Image Similarity Search",
                   page_icon="🐱", layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

# Sidebar
with st.sidebar:
    st.title("VectorVision - Image Similarity Search")
    st.markdown('''
    ## About
    This app is an Image Search Engine built using:
    - [Streamlit](https://streamlit.io)
    - [HuggingFace](https://huggingface.co/)
    ''')
    add_vertical_space(4)
    st.write("Made with :sparkling_heart: by [Sarthak Thakur](https://sarthak247.github.io)")

# Get default dataset
dataset_name = "pouya-haghi/imagenet-subset"
local_path = 'data'
dataset = get_dataset(dataset_name, local_path)

# Get model
model = SentenceTransformer('clip-ViT-B-32', device='cuda')

def main():
    # Main App
    st.header("VectorVision - Image Similarity Search")

    # Select LLM
    option = st.selectbox('Select Embedding Model', ('OpenAI CLIP', 'Google SigLip'))

    # Select top-k similarity search
    k = st.number_input(label="Top-K value [1-10]",
                        min_value=1,
                        max_value=10,
                        step=1,
                        value = 6 # default
    )
    

    # Accept Questions
    query = st.text_input("Prompt", value="a cat")
    # if st.button("Find images"):
    if option == 'Google SigLip':
        pass # yet to implement
    elif option == 'OpenAI CLIP':
        # Check for existing store or create one
        store_name = 'imagenet_subset_clip.index'
        index = load_faiss_index(store_name)
        query, distances, indices = retrieve_similar_images(query, model, index, top_k=k)
        num_cols = 3
        cols = st.columns(num_cols)

        for i, distance in enumerate(distances):
            col = cols[i % num_cols]  # Assign image to the correct column
            image = dataset['validation'][int(indices[i])]['image']
            new_image = image.resize((600, 400))
            col.image(new_image, use_column_width=True)  # Display image with column width
            col.caption(f'Score: {round(float(distance), 2)}')

if __name__ == '__main__':
    main()