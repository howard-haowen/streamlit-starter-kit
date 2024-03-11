import os
import pandas as pd
import random # for generating random response in dev
import streamlit as st
import time

#===<Start>Import from llama-index framework===
from llama_index.core import (Document,
                              VectorStoreIndex,
                              Settings,
                              StorageContext,
                              load_index_from_storage,
                              PromptTemplate,)
from llama_index.core.query_engine import PandasQueryEngine
# For using models hosted on Azure OpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
#===<End>Import from llama-index framework===

#===<Start>Global variables===
DATA_URL = "https://data.moa.gov.tw/Service/OpenData/TransService.aspx?UnitId=wxV177kLhEE3&IsTransData=1"
DATASET_PATH = './dataset.json' # for custom data file
SAMPLE_SIZE = 50 # for the number of records sampled from the dataset
INDEX_STORAGE_DIR = './storage' # for storing vectors
LLM_TEMP = 0 # for LLM temperature
SIM_TOP_K = 3 # for the number of nodes to retrieve
PROMPT_MESSAGE = "Ask questions here..."
PROCESSING_MESSAGE= "Got it! I'm on it..."
SAMPLE_Q1 = "我想找給貓吃的罐頭，要含有南瓜"
SAMPLE_Q2 = "I'm looking for food for adult dogs. They should contain pumpkin and originate from Taiwan."
ROBOT_MESSAGE = f"""
Talk to me! I'm your guide for pet products! 🐕🐈 
What are you looking for today?
You can ask in Chinese or English.
Here're just some ideas to get you started with:
Q1: {SAMPLE_Q1}
Q2: {SAMPLE_Q2}
"""
# For generating fake responses
RANDOM_RESPONSES = "foo bar yo man look".split()
#===<End>Global variables===

#===<Start>Azure OpenAI API secrets===
api_key = st.secrets.Azure.api_key
azure_endpoint = st.secrets.Azure.azure_endpoint
llm_deployment_name = st.secrets.Azure.llm_deployment_name
embed_deployment_name = st.secrets.Azure.embed_deployment_name
#===<End>Azure OpenAI API secrets===

#===<Start>Utility functions===
@st.cache_data
def load_data2df():
    return pd.read_json(DATA_URL)

@st.cache_resource
def build_index_from_docs():
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
    return index

def preprocess_df(df):
    resampled_df = df.sample(n=SAMPLE_SIZE, random_state=1)
    resampled_df.columns = [
    "product_id",
    "product_name",
    "product_type",
    "ways_of_handling",
    "weight",
    "ingredients",
    "nutritions",
    "meant_for_what_pet",
    "usage_instructions",
    "storage_conditions",
    "product_origin",
    "dealer_name",
    ]
    resampled_df.to_json('./dataset.json', orient='records')
    return resampled_df

def read_dataset():
    if os.path.exists(DATASET_PATH):
        resampled_df = pd.read_json(DATASET_PATH)
    else:
        df = load_data2df()
        resampled_df = preprocess_df(df)        
    return resampled_df    

def create_docs(df):
    text_columns = ['product_name', 'product_type', 'ways_of_handling', 
                'ingredients', 'nutritions', 'meant_for_what_pet',
                'usage_instructions', 'storage_conditions', 'product_origin',]
    metadata_columns = ['product_id', 'product_name', 'product_type',
                    'weight','meant_for_what_pet', 'product_origin','dealer_name',]

    documents = []
    for _, row in df.iterrows():
        text = " ".join(row[col] for col in text_columns)
        metadata = {col: row[col] for col in metadata_columns}
        doc = Document(text=text, metadata=metadata)
        documents.append(doc)
    return documents

def load_index():
    if not os.path.exists(INDEX_STORAGE_DIR):
        # buidl index from documents
        index = build_index_from_docs()
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    return index
#===<End>Utility functions===

#===<Start>Customize embedding and LLM model===
api_version = "2023-07-01-preview"
llm = AzureOpenAI(
    model="gpt-35-turbo-16k",
    deployment_name=llm_deployment_name,
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    temperature=LLM_TEMP,
)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=embed_deployment_name,
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)
Settings.llm = llm
Settings.embed_model = embed_model
#===<End>Customize embedding and LLM model===

#===<Start>Customize QA prompt===
new_qa_tmpl_str = (
    "You're a pet food expert, and your task is to suggest food products based on the query.\n"
    "Your suggestions should be appropriate for the type of pet mentioned in the query.\n"
    "Use metadata to filter out results if necessary.\n"
    "In your answer, always specify the product id, its name, and what pets it is meant for.\n"
    "For each product you suggest, always state why it's being recommended.\n"
    "Always answer in the same langauge as the query language.\n"
    "Use a friendly and conversational tone to reply."

    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
new_qa_tmpl = PromptTemplate(new_qa_tmpl_str)
#===<Start>Customize QA prompt===


st.set_page_config(
    page_title="Chat for pet products", 
    page_icon="🐕", 
    layout="centered", 
    initial_sidebar_state="auto",
    menu_items=None)
st.title("🤖 Intelligent shopping guide for pet products")
st.info(f"""
        - This app allows you to use natural languages 💬 to find out the pet products you're looking for, 
        powered by LlamaIndex 🦙, 
        and inspired by this [Streamlit app](https://llamaindex-chat-with-docs.streamlit.app/).
        - The dataset for pet products is taken from [here]({DATA_URL}).
        For this demo, only 50 products are sampled from the dataset.
        - Tech stack:
            - LLM and Embedding: Azure OpenAI
            - LLM Orchestration Framework: LlamaIndex
            - Backend Coding IDE: Google Colab
            - Frontend Coding IDE: GitHub Codespaces
            - Code Hosting: GitHub
            - Web UI Framework: Streamlit
            - Web APP Hosting: Streamlit Cloud
        """)

df = read_dataset()
documents = create_docs(df)
index = load_index()

def stream_init_robot_message():
    for word in ROBOT_MESSAGE.split(" "):
        yield word + " "
        time.sleep(0.05)

# Query engine based on Vector DB
if "vt_query_engine" not in st.session_state.keys():
    vt_query_engine = index.as_query_engine(similarity_top_k=SIM_TOP_K)
    vt_query_engine.update_prompts({"response_synthesizer:text_qa_template": new_qa_tmpl}) 
    st.session_state.vt_query_engine = vt_query_engine

# Query engine based on Pandas experssion
if "pd_query_engine" not in st.session_state.keys(): 
    st.session_state.pd_query_engine = PandasQueryEngine(df=df, verbose=True)

# Initialize chat history
if "messages" not in st.session_state.keys():
    with st.chat_message("assistant"):
        st.write_stream(stream_init_robot_message)
    st.session_state.messages = [
        {"role": "assistant", "content": ROBOT_MESSAGE}
    ]

# Response section for topK retrieval
def show_retrieval_resp(vt_response):
    # Text response
    response_type = "TopK Retrieval"
    response_message = vt_response.response
    st.markdown(f"### 🗃️ {response_type}")
    st.markdown(response_message)
    hist_message = {"role": "assistant", "content": response_type + ": " + response_message}
    st.session_state.messages.append(hist_message)
    # Metadata
    if product_ids := [v.get('product_id') for v in vt_response.metadata.values()]:
        st.write("Retrieved documents:")
        df_to_show = df[df['product_id'].isin(product_ids)]
        st.dataframe(df_to_show)

# Response section for pandas expression
def show_pandas_resp(pd_response):
    # Text response
    response_type = "Pandas expression"
    pd_expression = pd_response.metadata['pandas_instruction_str']
    st.markdown(f"### 🐼 {response_type}")
    st.markdown(f"""
                ```python
                {pd_expression}
                ```
                """)
    message = {"role": "assistant", "content": response_type + ": " + pd_expression}
    st.session_state.messages.append(message)
    # Try to show a dataframe given a pandas expression
    try:
        df_to_show = eval(pd_expression)
        st.dataframe(df_to_show)
    except:
        pass

# User input messages
if prompt := st.chat_input(PROMPT_MESSAGE):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Iterate over messages saved in st.session_state.messages
for message in st.session_state.messages:
    # In order not to show the initial robot message, 
    # which is streamed via st.write_stream.
    if len(st.session_state.messages) > 1:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Generate a response when the last speaker is not assistant 
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner(PROCESSING_MESSAGE):
            vt_response = st.session_state.vt_query_engine.query(prompt)
            pd_response = st.session_state.pd_query_engine.query(prompt)
            response1 = random.choice(RANDOM_RESPONSES)
            response2 = random.choice(RANDOM_RESPONSES)

            show_retrieval_resp(vt_response)
            show_pandas_resp(pd_response)


