
from copysafe import analysis, generate_document_vector, check_language, translate_text, model, tokenizer, source_data, source_vectors
import streamlit as st

def main():
    st.title("BERT-based Document Evaluation")

    # User input text area
    user_input = st.text_area("Enter your article here:", "")

    # Button to check plagiarism
    if st.button("Evaluate"):
        if user_input:
            with st.spinner("Analyzing..."):
                # Check language of the document
                document_language = check_language(user_input)

                # Translate non-English document to English for consistency
                if document_language != "en":
                    user_input = translate_text(user_input, document_language, "en")

                # Generate vector for the user input
                document_vector = generate_document_vector(user_input, model, tokenizer)
                
                # Perform plagiarism analysis
                response = analysis(document_vector, source_vectors, source_data, 0.85)

            # Display results in text area
            st.success("Evaluation completed. Results will be shown here.")
            st.text(f"Similarity Score: {response[1]}")
            st.text(
                f"Decision: {'Matched Detected' if response[0] else 'No Match Detected'}"
            )
            st.text_area("Article submitted:", user_input, height=200)
            if response[0]:
                st.text_area("Most Similar Article:", response[2], height=200)
        else:
            st.warning("Please enter an article for evaluation.")


if __name__ == "__main__":
    main()