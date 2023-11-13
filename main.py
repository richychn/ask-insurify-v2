import streamlit as st
from query import Query
import json
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Prepare variables for Huggingface Inference API
huggingface_api_token = os.getenv("HUGGINGFACE_API")
headers = {
    "Authorization": f"Bearer {huggingface_api_token}",
    "Content-Type": "application/json"
}
embedding_endpoint = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-small-en-v1.5"
llm_endpoint = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

# Create Streamlit UI
st.title("Ask Insurify About Car Insurance")
question = st.text_input("Question: ", "")
query_button = st.button("Get Answers")

if query_button:
    with st.spinner('Querying...'):
        embedding_payload = {
            "inputs": [question],
            "options": {
                "wait_for_model": True
            }
        }
        try:
            embedding_response = requests.post(embedding_endpoint, headers=headers, json=embedding_payload)
            if embedding_response.status_code == 200:
                embedding = json.loads(embedding_response.text)
                query = Query(question, embedding)
                prompt = query.get_prompt()
                llm_payload = {
                    "inputs": prompt,
                    "parameters": {
                        "return_full_text": False,
                        "num_return_sequences": 2,
                        "max_new_tokens": 250
                    },
                    "options": {
                        "wait_for_model": False
                    }
                }
                try: 
                    llm_response = requests.post(llm_endpoint, headers=headers, json=llm_payload)
                    if llm_response.status_code == 200:
                        answer = json.loads(llm_response.text)[0]['generated_text']
                        st.write(answer)
                        for idx, source in enumerate(query.sources):
                            st.write(f"Source {idx + 1}: {source}")
                    else:
                        raise Exception(llm_response.status_code, llm_response.text)
                except Exception as e:
                    st.write("LLM error occurred - please try again later or reach out for help.")
                    st.write(f"Details: {e}")
            else:
                raise Exception(embedding_response.status_code, embedding_response.text)
        except Exception as e:
            st.write("Embedding error occurred - please try again later or reach out for help.")
            st.write(f"Details: {e}")