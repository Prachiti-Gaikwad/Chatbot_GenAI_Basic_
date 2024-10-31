import streamlit as st
import openai
from astrapy import DataAPIClient
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncio
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set up Streamlit layout 
st.title("LangChain Chat with AstraDB Vector Search and OpenAI")
st.markdown("Chat using a LangChain-powered chatbot with vector embeddings stored in AstraDB!")

# Get OpenAI API Key and AstraDB Token from the user
astra_db_token = ''

openai_api_key = ""

# Initialize OpenAI API
if openai_api_key:
    openai.api_key = openai_api_key

# Initialize AstraDB client
def init_astra_client(astra_token):
    client = DataAPIClient(astra_token)
    db = client.get_database_by_api_endpoint("https://0952448a-3f79-4c1c-80ab-4d49966bf65e-us-east-2.apps.astra.datastax.com")
    return db

# Function to truncate the embedding size if it exceeds 1000 dimensions
def truncate_embedding(embedding, target_dim=1000):
    if len(embedding) > target_dim:
        return embedding[:target_dim]
    return embedding

# Function to store vectors and responses in AstraDB
def store_embedding_in_db(db, user_input, embedding, response):
    collection = db.get_collection('surendra')
    collection.insert_one({
        'user_input': user_input,
        'embedding': embedding,  # Convert to list for JSON storage
        'response': response
    })

# Function to retrieve all vectors from AstraDB
def get_all_embeddings_from_db(db):
    collection = db.get_collection('surendra')
    data = collection.find({})
    return list(data)

# Function to calculate cosine similarity between vectors
def find_most_similar_embedding(query_embedding, all_embeddings):
    all_vectors = np.array([np.array(doc['embedding']) for doc in all_embeddings])
    similarities = cosine_similarity([query_embedding], all_vectors)
    most_similar_index = np.argmax(similarities)
    return all_embeddings[most_similar_index]

# OpenAI's function to generate embedding for the input text
async def async_generate_embedding(text):
    response = await openai.Embedding.acreate(input=text, model="text-embedding-ada-002")
    return response['data'][0]['embedding']

# Function to generate a response using OpenAI's GPT model
async def async_generate_response(prompt):
    response = await openai.Completion.acreate(
        engine="gpt-3.5-turbo-instruct",  # Correct engine name
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# Fallback Local Model (Offline Mode)
def local_fallback_response(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize AstraDB if the token is provided
if astra_db_token:
    try:
        astra_db = init_astra_client(astra_db_token)
        st.success("Connected to AstraDB successfully")
    except Exception as e:
        st.error(f"Error connecting to AstraDB: {e}")

# Define prompt templates for LangChain
prompt_template_general = """
You are an AI assistant specialized in general agriculture.
Provide relevant and helpful information about: {input}
"""

prompt_template_soil = """
You are an expert in soil health and management. Help farmers understand how to improve soil quality.
Question: {input}
"""

# Text input for user prompt
user_input = st.text_input("Ask something:")

# Memory to store conversation context
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()

# Main async function to handle user input and responses
async def main():
    if user_input:
        try:
            # Step 1: Generate embedding for user input
            query_embedding = await async_generate_embedding(user_input)

            # Step 2: Truncate embedding to 1000 dimensions if necessary
            truncated_embedding = truncate_embedding(query_embedding)

            # Step 3: Retrieve stored embeddings from AstraDB
            if astra_db_token:
                all_embeddings = get_all_embeddings_from_db(astra_db)
                if all_embeddings:
                    # Step 4: Find the most similar embedding (context)
                    most_similar_doc = find_most_similar_embedding(truncated_embedding, all_embeddings)
                    similar_context = most_similar_doc['user_input']
                    similar_response = most_similar_doc['response']
                    st.write(f"Found similar context: {similar_context}")
                    st.write(f"Previous response: {similar_response}")
                else:
                    st.write("No similar context found.")

            # Step 5: Create a prompt using LangChain template
            if "soil" in user_input.lower():
                prompt = prompt_template_soil.format(input=user_input)
            else:
                prompt = prompt_template_general.format(input=user_input)

            # Step 6: Generate a response from OpenAI based on the current prompt
            response = await async_generate_response(prompt)

            # Step 7: Store the new user input, embedding, and response in AstraDB
            if astra_db_token:
                store_embedding_in_db(astra_db, user_input, truncated_embedding, response)
                st.success("Message and embedding stored in AstraDB")

            # Update conversation memory
            #memory.add_memory(user_input=user_input, assistant_response=response)
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(response)



            # Display the response
            st.write(f"Assistant: {response}")

        except Exception as e:
            st.error(f"Error during processing: {e}")

# Run the main async function
if user_input:
    asyncio.run(main())

# Retrieve chat history or vectors from AstraDB
if st.button("Retrieve Chat History"):
    try:
        if astra_db_token:
            history = get_all_embeddings_from_db(astra_db)
            st.write("Chat History:")
            for msg in history:
                st.write(f"User: {msg['user_input']}")
                st.write(f"Assistant: {msg['response']}")
    except Exception as e:
        st.error(f"Error retrieving chat history: {e}")

# Embedding Visualization
# if st.button("Visualize Embeddings"):
#     try:
#         if astra_db_token:
#             all_embeddings = get_all_embeddings_from_db(astra_db)
#             embeddings = [doc['embedding'] for doc in all_embeddings]
#             tsne = TSNE(n_components=2, random_state=42)
#             reduced_embeddings = tsne.fit_transform(np.array(embeddings))

#             # Plot the embeddings
#             plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
#             st.pyplot(plt)
#     except Exception as e:
#         st.error(f"Error visualizing embeddings: {e}")

if 'positive_feedback_count' not in st.session_state:
    st.session_state.positive_feedback_count = 0

# Feedback mechanism
if st.button("Thumbs Up 👍"):
    st.session_state.positive_feedback_count += 1
    st.write("Thanks for your feedback! 👍")
if st.button("Thumbs Down 👎"):
    st.write("Sorry to hear that. We'll try to improve! 👎")

# Analytics Dashboard
st.sidebar.title("Analytics")
total_queries = len(get_all_embeddings_from_db(astra_db)) if astra_db_token else 0
st.sidebar.metric("Total Queries", total_queries)
#positive_feedback_count = 0  # You can store and count feedback separately
average_response_time = 100  # Placeholder, replace with actual timing data
st.sidebar.metric("Positive Feedback", st.session_state.positive_feedback_count)
st.sidebar.metric("Response Time (ms)", average_response_time)
