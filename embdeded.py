# embed_pdfs.py
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

## load env variable
load_dotenv()

# Setup
PDF_DIR = "pdfs"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")

# Load and combine all PDFs
print("🔍 Reading PDFs...")
all_text = ""
for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, filename)
        print(f"📄 Processing {filename}...")
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            if page.extract_text():
                all_text += page.extract_text()

# Split text
splitter = CharacterTextSplitter(
    separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
)
chunks = splitter.split_text(all_text)

# Embed & Save
print("🧠 Creating embeddings...")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
vectorstore.save_local("vectorstore")
print("✅ Embeddings saved to ./vectorstore")
