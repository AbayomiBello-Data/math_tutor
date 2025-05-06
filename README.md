This is an interactive, AI-powered math tutoring assistant that helps students understand math problems step-by-step using contextual guidance from textbook materials. Built with LangChain, OpenAI GPT, FAISS vector search, and Streamlit, this app simulates a guided learning experience focused strictly on the content provided in math PDFs.
**This is a DEMO Application for The Developer Academy client**

**test application here:** https://mathtuto.streamlit.app/
**🚀 Features**
🧠 LLM-powered tutor: Guides students step-by-step using textbook-based responses (not just answers)

📚 PDF ingestion & vector storage: Automatically parses and embeds textbook PDFs for knowledge grounding

🔍 Math-only filter: Rejects non-math questions using a classifier

🧼 Content moderation: Filters out harmful or inappropriate content

💬 Streamlit chat UI: User-friendly interface with streaming token-based responses

🎨 Custom-styled messages: Distinguishes between user and assistant replies with clean styling

**🧩 How It Works**
PDF Parsing & Embedding: All PDF files placed in the /pdfs folder are parsed and converted into vector embeddings using OpenAIEmbeddings, then stored locally using FAISS.

Question Handling:

Filters out greetings or harmful content

Uses a classifier to verify if a question is math-related

Searches for relevant content in the vectorstore

Response Generation: An OpenAI GPT model is prompted using a carefully crafted template to return only process-driven explanations based on the matched textbook content.

**🛠️ Installation & Setup**
Clone the repository


git clone https://github.com/your-username/guided-math-tutor.git
cd guided-math-tutor
Install dependencies


pip install -r requirements.txt
Prepare .env file

Create a .env file with your OpenAI API key:


OPENAI_API_KEY=your-openai-api-key
Add PDFs

Place math textbooks or notes into the pdfs/ directory.

Embed the documents

python embed_pdfs.py
Launch the app

streamlit run app.py
**
📦 Directory Structure**

.
├── app.py               # Main Streamlit chatbot app
├── embed_pdfs.py        # PDF parser and vectorstore builder
├── pdfs/                # Folder to store PDF textbook files
├── vectorstore/         # Saved FAISS vector index
├── .env                 # Environment variables (API key)
├── requirements.txt     # Python dependencies
**✨ Example Use Case**
User: How do I solve a quadratic equation?

Tutor: Start by identifying the coefficients a, b, and c from the equation ax² + bx + c = 0. Then apply the quadratic formula...

**🧠 Tech Stack**
OpenAI GPT-3.5 – Language generation

LangChain – Prompting, document chaining, vector search

FAISS – Vector similarity search

Streamlit – Frontend UI

PyPDF2 – PDF parsing

📜 License
This project is licensed under the MIT License.
