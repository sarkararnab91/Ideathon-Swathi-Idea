import streamlit as st
from utils import get_table_data,parse_pdf
def main():
    with st.form("my_form"):
        # File upload
        uploaded_file = st.file_uploader("Upload a pdf or text file")
        # Input fields
        mcq_count = st.number_input("No of MCQs", min_value=3, max_value=20,placeholder=3)
        topic = st.text_input("Provide a topic", max_chars=100,placeholder="backpropagation algorithm")
        difficulty = st.text_input("Provide Quiz difficulty", max_chars=100, placeholder="simple or complex")
        if uploaded_file is not None:
            vectorstore = parse_pdf(uploaded_file)
        button = st.form_submit_button("Create quiz")
    st.write("Outside the form")

if __name__ == "__main__":
    main()