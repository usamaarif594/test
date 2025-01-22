import streamlit as st
from io import BytesIO
from datetime import datetime
import pdfkit

def generate_pdf():
    """Generates a PDF from predefined HTML content."""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_file_name = f"search_results_{current_time}.pdf"

    
    # Define HTML content
    html_content = """
    <html>
    <head>
        <title>Streamlit App PDF</title>
    </head>
    <body>
        <h1>Streamlit App Content</h1>
        <p>This is the content of your Streamlit app saved as a PDF.</p>
    </body>
    </html>
    """

    # Save HTML as PDF in memory
    pdf_bytes = BytesIO()
    pdfkit.from_string(html_content, output_path=pdf_bytes)
    pdf_bytes.seek(0)

    # Provide download button
    st.download_button(
        label="Download App Content as PDF",
        data=pdf_bytes.getvalue(),
        file_name=pdf_file_name,
        mime="application/pdf"
    )

st.title('Screenshot as PDF')
if st.button("Generate PDF"):
    generate_pdf()
    st.write("PDF Generated!")
