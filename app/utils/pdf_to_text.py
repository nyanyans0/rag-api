import PyPDF2

def pdf_to_text(pdf_path, output_path):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        
        # Initialize an empty string to store the text
        text = ""
        
        # Iterate through all the pages
        for page in pdf_reader.pages:
            # Extract text from the page and add it to our string
            text += page.extract_text() + "\n"
    
    # Write the extracted text to a file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(text)

# Usage
pdf_path = 'data/NVIDIAAn.pdf'
output_path = 'data/nvidia_document.txt'
pdf_to_text(pdf_path, output_path)