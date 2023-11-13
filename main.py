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
st.set_page_config(page_title='Ask Insurify')

st.header("Ask Insurify About Car Insurance")
question = st.text_input("Get your car insurance questions answered by an AI using Insurify's articles.", "")
query_button = st.button("Get Answers")

if query_button or question != "":
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

with st.sidebar:
    st.subheader("About the app")
    st.info("This application uses a large language model to generate answers based on [Insurify](https://insurify.com)'s articles.")
    st.write("\n\n")
    st.markdown("**Resources used**")
    st.markdown("* [Zephyr-7B-Alpha LLM for Text Generation](https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha)")
    st.markdown("* [FlagEmbedding for Embedding Articles and Question](https://huggingface.co/BAAI/bge-small-en-v1.5)")
    st.markdown("* [LlamaIndex for Retrieval and Querying](https://www.llamaindex.ai)")
    st.markdown("* [HuggingFace Inference API for Hosting Models](https://huggingface.co/docs/api-inference/index)")
    st.write("\n\n")
    st.divider()
    st.caption("Created by [Richy Chen](https://linkedin/com/in/richychen/) using [Streamlit](https://streamlit.io/)🎈.")
