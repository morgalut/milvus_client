import os
from docx import Document
from pptx import Presentation
from pdfminer.high_level import extract_text

def extract_text_from_file(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    elif ext == ".pdf":
        return extract_text(file_path)

    elif ext == ".pptx":
        prs = Presentation(file_path)
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        for run in paragraph.runs:
                            text_runs.append(run.text)
        return "\n".join(text_runs)

    else:
        return None
