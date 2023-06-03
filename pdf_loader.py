import os
import PyPDF2


def load_single_document(file_path: str):
    # Loads a single document from file path
    if file_path[-4:] == '.txt':
        with open(file_path, 'r') as f:
            return f.read()

    elif file_path[-4:] == '.pdf':
        pdfFileObj = open(file_path, 'rb')
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        text = ''
        for page in pdfReader.pages:
            text += page.extract_text()
        return text

    elif file_path[-4:] == '.csv':
        with open(file_path, 'r') as f:
            return f.read()

    else:
        raise Exception('Invalid file type')


def load_documents(source_dir: str):
    # Loads all documents from source documents directory
    all_files = os.listdir(source_dir)
    return [load_single_document(f"{source_dir}/{file_path}") for file_path in all_files if file_path[-4:] in ['.txt', '.pdf', '.csv']]
