'''
    This is where the actual logic for extracting information from the invoice is defined. It uses the LangChain library to create a document from the uploaded PDF and then uses a language model to extract key information from the document.
'''

# import necessary libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate



def create_docs(user_pdf_list):

    for filename in user_pdf_list:
        print(f"Processing file: {filename.name}")

        # create a temp file for the uploaded PDF and write the contents to it
        with open(filename.name, "wb") as f:
            f.write(filename.getbuffer())
        


        # loader = PyPDFLoader(filename.name)
        loader = PyPDFLoader(filename.name)

        pages = loader.load_and_split()

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vector = FAISS.from_documents(pages, embeddings)

        # create a template for the querying of the data, this prompt will be given to the LLM as part of the context for the question answering
        template = """Extract all the following values: invoice number, invoice date, due date, total amount, vendor name, and line items from the following document (create a JSON output with the extracted values only):
          {context}
        
            The fields and the values in the above content may be jumpled up as they are extracted from the PDF, so you will need to use your best judgement to extract the correct values. The line items should be extracted as a list of dictionaries, where each dictionary contains the description, quantity, unit price, and total price for each line item. If any of the fields are missing or cannot be extracted, please return null for that field.
        """

        prompt = PromptTemplate.from_template(template)

        llm = HuggingFaceEmbeddings(temperature=0,model_name="sentence-transformers/all-MiniLM-L6-v2")

        retriever = vector.as_retriever()

        document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

        retrieval_chain = create_retrieval_chain(retriever=retriever, combine_documents_chain=document_chain)

        response = retrieval_chain.invoke("What are the invoice number, invoice date, due date, total amount, vendor name, and line items in the document?")
        answer = response['answer']

        print("*************DONE*************")
        return answer




