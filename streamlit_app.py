import os
import pandas as pd
import random # for generating random response in dev
import streamlit as st

#===<Start>Import from llama-index framework===
from llama_index.core import (Document,
                              SimpleDirectoryReader,
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
ROBOT_MESSAGE = "What kinds of pet products are you looking for today?"
PROMPT_MESSAGE = "Ask questions here..."
PROCESSING_MESSAGE= "Got it! I'm on it..."
RANDOM_RESPONSES = "foo bar yo man look".split()
#===<End>Global variables===

#===<Start>Azure OpenAI API secrets===
api_key = st.secrets.Azure.api_key
azure_endpoint = st.secrets.Azure.azure_endpoint
llm_deployment_name = st.secrets.Azure.llm_deployment_name
embed_deployment_name = st.secrets.Azure.embed_deployment_name
#===<End>Azure OpenAI API secrets===

@st.cache_data
def load_data2df():
    return pd.read_json(DATA_URL)

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

@st.cache_resource
def build_index_from_docs():
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
    return index

def load_index():
    if not os.path.exists(INDEX_STORAGE_DIR):
        # buidl index from documents
        index = build_index_from_docs()
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    return index

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

#===<Start>Web UI===
st.set_page_config(page_title="Chat with the Streamlit docs, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("🤖: Talk to me! I'm your guide for pet products! 🐕🐈")
st.info("""
        Use natural languages to find out pet products you're looking for, 
        powered by LlamaIndex 💬🦙, 
        and inspired by this [Streamlit app](https://llamaindex-chat-with-docs.streamlit.app/).
        """)

df = read_dataset()
documents = create_docs(df)
index = load_index()

#===<Start>Manage session state===
# Query engine based on Vector DB
if "vt_query_engine" not in st.session_state.keys(): 
    st.session_state.vt_query_engine = index.as_query_engine(similarity_top_k=SIM_TOP_K)
# Query engine based on Pandas experssion
if "pd_query_engine" not in st.session_state.keys(): 
    st.session_state.pd_query_engine = PandasQueryEngine(df=df, verbose=True)
# Chat history
if "messages" not in st.session_state.keys(): 
    st.session_state.messages = [
        {"role": "assistant", "content": ROBOT_MESSAGE}
    ]
#===<End>Manage session state===
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

def show_pandas_resp(pd_response):
    # Text response
    response_type = "Pandas expression"
    response_message = pd_response.response
    st.markdown(f"### 🐼 {response_type}")
    st.markdown(response_message)
    message = {"role": "assistant", "content": response_type + ": " + response_message}
    st.session_state.messages.append(message)
    # Metadata
    pd_expression = pd_response.metadata['pandas_instruction_str']
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

            col1, col2 = st.columns(2)
            with col1:
                show_retrieval_resp(vt_response)
            with col2:
                show_pandas_resp(pd_response)


