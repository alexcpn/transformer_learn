import fitz  # PyMuPDF



# Specify the PDF file path
pdf_path = "./data/WINNF-TS-1014-V1.4_r2.pdf"

dict_pages={}
# Open the PDF file
with fitz.open(pdf_path) as pdf_document:
    # Read each page and print the text
    print("\nPages Content:")
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)  # Load each page
        page_text = page.get_text()  # Extract text from the page
        dict_pages[page_number]=page_text
        #print(f"\n--- Page {page_number + 1} ---")
        print(f"\n--- Page {page_number + 1} ---")
        if page_number== 30:
            print(page_text.encode('utf-8', 'replace').decode('utf-8'))
            print("-"*80)
      