#!/usr/bin/env python3
"""
Script to convert all PDF, DOCX, and legacy DOC files in a folder to text files,
using pdfplumber for PDFs, python-docx for DOCX, and LibreOffice headless for DOC.

Usage:
    python convert_docs_to_text.py

Requirements:
    pip install pdfplumber python-docx
    LibreOffice (or soffice) must be installed and on your PATH
"""
import os
import subprocess
import tempfile
import pdfplumber
import docx


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF using pdfplumber."""
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if page.width > page.height:
                mid = page.width / 2
                left = page.within_bbox((0, 0, mid, page.height)).extract_text() or ""
                right = page.within_bbox((mid, 0, page.width, page.height)).extract_text() or ""
                text_chunks.extend([left, right])
            else:
                text_chunks.append(page.extract_text() or "")
    return "\n\n".join(text_chunks)


def extract_docx_text(docx_path: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    doc = docx.Document(docx_path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_doc_text(doc_path: str) -> str:
    """Extract text from a legacy DOC file using LibreOffice headless conversion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run([
            'soffice', '--headless', '--convert-to', 'txt:Text',
            '--outdir', tmpdir, doc_path
        ], check=True)
        txt_name = os.path.splitext(os.path.basename(doc_path))[0] + '.txt'
        txt_path = os.path.join(tmpdir, txt_name)
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()


def convert_files_to_text(input_dir: str, output_dir: str) -> None:
    """
    Convert all PDFs, DOCX, and DOC files in input_dir into text files in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        ext = os.path.splitext(fname)[1].lower()
        src = os.path.join(input_dir, fname)
        base = os.path.splitext(fname)[0]
        dest = os.path.join(output_dir, base + '.txt')
        try:
            if ext == '.pdf':
                text = extract_pdf_text(src)
            elif ext == '.docx':
                text = extract_docx_text(src)
            elif ext == '.doc':
                text = extract_doc_text(src)
            else:
                print(f"Skipping unsupported: {fname}")
                continue
            with open(dest, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Converted: {fname} → {base}.txt")
        except Exception as e:
            print(f"Error converting {fname}: {e}")



if __name__ == '__main__':
    input_directory = '/home/mh1f25/scratch/upcast_pro/odrl_extraction/row_data_processing/data_sharing_agreement_1'
    output_directory = '/home/mh1f25/scratch/upcast_pro/odrl_extraction/row_data_processing/data_sharing_agreement_1_text'
    convert_files_to_text(input_directory, output_directory)

