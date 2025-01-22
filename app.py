import streamlit as st
from io import BytesIO
from datetime import datetime
import pdfkit

def generate_pdf_from_html():
    """Generates a PDF from Streamlit app content."""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_file_name = f"search_results_{current_time}.pdf"

    # Capture Streamlit's rendered content
    html_content = st._get_report_ctx().session_id  # Access the current session
    pdf_bytes = BytesIO()

    # Generate PDF using pdfkit or weasyprint
    pdfkit.from_string(html_content, pdf_bytes)  # Use pdfkit to convert HTML to PDF
    pdf_bytes.seek(0)

    # Provide download button
    st.download_button(
        label="Download App Content as PDF",
        data=pdf_bytes,
        file_name=pdf_file_name,
        mime="application/pdf"
    )

if st.button("Generate PDF"):
    generate_pdf_from_html()
    st.write("PDF Generated!")
