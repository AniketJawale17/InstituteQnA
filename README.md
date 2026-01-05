# InstituteQnA

A comprehensive Question Answering system for COEP Technological University built using Retrieval-Augmented Generation (RAG) technology.

## 🚀 Features

- **RAG-Powered Q&A**: Intelligent question answering using vector search and LLMs
- **Multi-Source Data**: Processes information from website and PDF documents
- **RESTful API**: FastAPI-based endpoints for easy integration
- **Multi-LLM Support**: Works with Azure OpenAI and Google Gemini
- **Interactive Mode**: Command-line interface for testing
- **Batch Processing**: Handle multiple queries efficiently

## 📋 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env  # Add your API keys

# 3. Test the RAG system
python examples/quick_start.py

# 4. Start the API server
python app.py
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions.

## 📚 Documentation

- **[RAG System Overview](RAG_SYSTEM_README.md)** - Complete RAG documentation
- **[Setup Guide](SETUP_GUIDE.md)** - Installation and configuration
- **[Implementation Summary](RAG_IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[Usage Examples](examples/)** - Code examples and scripts

## 🏗️ Architecture

```
User Query → RAG Pipeline → [Retrieval → Context → LLM] → Answer
```

## 🎯 Use Cases

- Admission requirements and procedures
- Department and program information
- Fee structure and scholarships
- Campus facilities and services
- Hostel information and rules
- General university queries

## 📂 Project Structure

```
InstituteQnA/
├── institute_qna/
│   ├── rag/                    # RAG system modules
│   ├── data_extraction/        # Web scraping
│   └── data_preprocess/        # Data processing
├── examples/                   # Usage examples
├── ug_admission_data/         # Vector database
├── app.py                     # FastAPI application
└── requirements.txt           # Dependencies
```

## 🔌 API Endpoints

- `POST /query` - Ask a question
- `POST /batch_query` - Multiple questions
- `GET /health` - System health check
- `POST /Extract` - Extract web data
- `POST /Process` - Process PDF documents

## 💡 Example Usage

```python
from institute_qna.rag import RAGPipeline

pipeline = RAGPipeline()
response = pipeline.query("What are the admission requirements?")
print(response['answer'])
```

## 🌐 Data Sources





# Websites containing informations

1. https://www.coeptech.ac.in/about-us/about-university/ - About University
2. https://www.coeptech.ac.in/hostel/hostel-admissions/ - About Hostel Information
3. https://www.coeptech.ac.in/hostel/rules-and-regulations/ - Hostel Rules and Regulations
4. https://www.coeptech.ac.in/student-corner/student-services/student-helpline/ - Contact Information
5. https://www.coeptech.ac.in/student-corner/student-clubs/ - Clubs Information
6. https://www.coeptech.ac.in/facilities/facilities-manager/facilities-for-differently-abled-individuals/ - Differently Abled facilities
7. https://www.coeptech.ac.in/useful-links/university-sections/ - University Sections





