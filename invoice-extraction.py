'''
    This is the file that defines the UI the user will interact with using Streamlit. It also defines the logic for handling user input and displaying results.

    This allow the user to upload an invoice, and then it will display the extracted information in a structured format. The user can also choose to download the extracted information as a CSV file.
'''

# import necessary libraries
import streamlit as st
from dotenv import load_dotenv
import os
import invoiceutil as iu

def main():
    load_dotenv()  # Load environment variables from .env file

    st.set_page_config(page_title="Invoice Extraction", page_icon="📄", layout="centered")
    st.title("Invoice Extraction")
    st.subheader("Upload your invoice and extract key information")

    # file uploads
    pdf = st.file_uploader("Upload PDF Invoice", type=["pdf"], accept_multiple_files=True)

    submit = st.button("Extract Information")

    if submit and pdf:
        for uploaded_file in pdf:
            with st.spinner("Extracting information..."):
                df = iu.create_docs(pdf)
                st.write(df)
            
            st.write("I hope I was able to save you time")

# invoking the main function to run the program

if __name__ == "__main__":
    main()