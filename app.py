from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import LanceDB
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

import lancedb

def main():
    load_dotenv()
    
    embeddings = OpenAIEmbeddings()
    
    db = lancedb.connect("./tmp/lancedb")
    table = db.create_table(
        "my_table",
        data=[
            {
            "vector": embeddings.embed_query("Hello World"),
            "text": "Hello World",
            "id": "1",
            }
            ],
        mode="overwrite",
        )
    # print(os.getenv("OPENAI_API_KEY"))
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDF 🦜🔗💬")
    
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)
        
        # split langchain
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_text(text)
        
        # st.write(chunks)
        
        # kb = FAISS.from_texts(chunks, embeddings)
        kb = LanceDB.from_texts(chunks, embeddings, connection=table)
        # input
        
        question_prompt = st.text_input("Ask any question about your uploaded PDF")
        if question_prompt:
            docs = kb.similarity_search(question_prompt)
            # st.write(docs)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type='stuff')
            response = chain.run(input_documents=docs, question=question_prompt)
            
            st.write(response)
        
    
if __name__ == "__main__":
    main()