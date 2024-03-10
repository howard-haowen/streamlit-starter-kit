import os
import pandas as pd
import streamlit as st

#===<Start>Global variables===
DATA_URL = "https://data.moa.gov.tw/Service/OpenData/TransService.aspx?UnitId=wxV177kLhEE3&IsTransData=1"
DATASET_PATH = './dataset.json' # for custom data file
SAMPLE_SIZE = 50 # for the number of records sampled from the dataset
INDEX_STORAGE_DIR = './storage' # for storing vectors
LLM_TEMP = 0 # for LLM temperature
SIM_TOP_K = 3 # for the number of nodes to retrieve
#===<End>Global variables===

st.title('🎈App Name')
st.write('Hello world!')

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


resampled_df = read_dataset()
st.dataframe(resampled_df.sample(3))