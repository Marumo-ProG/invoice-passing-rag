from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import LLMChain
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.prompts import PromptTemplate
import tempfile

def create_docs(user_pdf_list):
    for filename in user_pdf_list:
        print(f"Processing file: {filename.name}")

        # Save uploaded PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(filename.getbuffer())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()

        # Embeddings for vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector = FAISS.from_documents(pages, embeddings)
        retriever = vector.as_retriever()

        # Prompt template
        template = """Extract all the following values: invoice number, invoice date, due date, total amount, vendor name, and line items from the following document (create a JSON output with the extracted values only):
{context}

The fields and the values in the above content may be jumbled up as they are extracted from the PDF, so you will need to use your best judgement to extract the correct values. The line items should be extracted as a list of dictionaries, where each dictionary contains the description, quantity, unit price, and total price for each line item. If any of the fields are missing or cannot be extracted, please return null for that field.
"""
        prompt = PromptTemplate.from_template(template)

        # Correct LLM
        pipe = pipeline(
            "text-generation",
            model="google/flan-t5-small",
            max_new_tokens=512
        )   

        llm = HuggingFacePipeline(pipeline=pipe)

        # Create chain
        document_chain = LLMChain(llm=llm, prompt=prompt)

        # Retrieval chain
        retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )

        response = retrieval_chain.invoke({"input":"What are the invoice number, invoice date, due date, total amount, vendor name, and line items in the document?"})
        answer = response['answer']

        print("*************DONE*************")
        return answer
