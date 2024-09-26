import pdfplumber

def pdf_to_text_with_spaces(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract the text and add a space after each line
            page_text = page.extract_text(x_tolerance=1.5, y_tolerance=1.5)
            if page_text:
                # Add space where a line ends but not a period or hyphen
                page_text = "\n".join(line + " " if line and not line.endswith((".", "-", " ")) else line for line in page_text.split("\n"))
                text += page_text + "\n"  # Add a newline for page separation

    return text.strip()
# Usage example
file_path = "/home/alex/Downloads/Troubleshooting_Guide_Issue_1.pdf"
parsed_text = pdf_to_text_with_spaces(file_path)
with open("troubleshooting.txt", "w") as f:
    f.write(parsed_text)
    
print("Written pdf to file")