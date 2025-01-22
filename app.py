from mss import mss
from PIL import Image
from io import BytesIO

def take_screenshot_as_pdf():
    """Takes a screenshot of the entire screen and provides it for download as a PDF."""
    with mss() as sct:
        screenshot = sct.shot(output=None)  # Take a screenshot and store it in memory
        img = Image.open(BytesIO(screenshot))
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_file_name = f"search_results_{current_time}.pdf"
        
        # Save screenshot as PDF in memory
        pdf_bytes = BytesIO()
        img.convert("RGB").save(pdf_bytes, format="PDF")
        pdf_bytes.seek(0)  # Move to the beginning of the file

        # Provide download button
        st.download_button(
            label="Download Screenshot as PDF",
            data=pdf_bytes,
            file_name=pdf_file_name,
            mime="application/pdf"
        )
if st.button('ss'):
    take_screenshot_as_pdf()
    st.write('DOne")
