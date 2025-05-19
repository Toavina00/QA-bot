import os
import getpass

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline


from huggingface_hub import login

import gradio as gr


load_dotenv()

if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")

login(os.environ["HUGGINGFACEHUB_API_TOKEN"])

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHAT_MODEL = "google/gemma-3-1b-it"

PROMPT_TEMPLATE="""\
    ### Instruction
    Answer the question by leveraging the information in the following context

    ### Context
    %s

    ### Question
    %s

    ### Output
"""

huggingface_embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
llm_model = ChatHuggingFace(
    llm=HuggingFacePipeline(
        pipeline=pipeline(
            task="text-generation",
            model="google/gemma-3-1b-it",
            max_new_tokens=512,
            repetition_penalty=1.03,
        ),
    )
)

def inference(prompt, files):

    chunks = UnstructuredLoader(files).load_and_split(
        RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
        )
    )
    

    vectordb  = Chroma.from_documents(chunks, huggingface_embedding)
    retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={"k": 3})

    docs = retriever.invoke(prompt)

    context = "\n".join(docs)

    return llm_model.invoke(PROMPT_TEMPLATE % (context, prompt))


interface = gr.Interface(
    title="QA Bot",
    description="Ask questions based on your uploaded documents.",
    fn=inference,
    inputs=[
        gr.Textbox(lines=2, label="Question"),
        gr.File(file_count="multiple", label="Upload Documents")
    ],
    outputs=gr.Textbox(label="Answer"),
)


def main():
    interface.launch(server_name="127.0.0.1", server_port= 7860)


if __name__ == "__main__":
    main()
