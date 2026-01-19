# AI Infusion for BA - POC

This is a Proof of Concept (POC) for infusing AI into Business Analyst (BA) services to improve the quality, coverage, and speed of Business Requirements Documents (BRDs).

## Features

- **BRD Project Management**: Isolated projects with persistent embeddings storage
- **Document Management**: Upload business docs, MOMs, technical docs, transcripts per project
- **Advanced LlamaIndex Embeddings**: Uses MarkdownElementNodeParser and VectorStoreIndex for superior document processing
- **Global Embedding Cache**: Reuses embeddings across projects for common documents to save processing time
- **Multimodal Support**: Better handling of complex document structures and layouts
- **Word Template Integration**: Editable BRD template (BRD_Template.docx) with placeholders
- **AI-Powered Generation**: GPT-5 generates content for each BRD section
- **Word Document Download**: Filled BRD templates downloadable as .docx files
- **Context-Aware**: Intelligent retrieval of relevant document chunks using LlamaIndex

## Setup

1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Set up environment variables in `.env`:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `LLAMA_CLOUD_API_KEY`: Your LlamaCloud API key
6. Run the application: `python main.py`

The server will start on `http://0.0.0.0:8000`.

## API Endpoints

- `GET /health`: Health check
- `POST /create_project`: Create a new BRD project (optional "name" parameter)
- `GET /projects`: List all BRD projects with names and metadata
- `POST /projects/{project_id}/upload`: Upload document to project (form-data with file)
- `GET /projects/{project_id}/documents`: List documents in project
- `DELETE /projects/{project_id}/documents/{document_id}`: Delete document from project
- `POST /projects/{project_id}/generate_brd`: Generate BRD for project (JSON with "requirements" field) - returns .docx download
- `GET /brd_template`: Get BRD template structure

## Usage Flow

1. Create a BRD project: `POST /create_project` → returns `project_id`
2. Upload documents to project: `POST /projects/{project_id}/upload`
3. View uploaded documents: `GET /projects/{project_id}/documents`
4. Generate BRD: `POST /projects/{project_id}/generate_brd` with requirements
5. Download formatted BRD from the response

## Features

- **Persistent Storage**: Embeddings stored per project, reusable across sessions
- **Document Management**: Add/remove documents from BRD projects
- **Template-Based Generation**: BRD generated with structured template
- **Context-Aware**: Uses similarity search on uploaded documents

## Usage

1. Upload relevant documents using the `/upload` endpoint.
2. Provide requirements in the `/generate_brd` endpoint to generate an enhanced BRD.

## Technologies Used

- FastAPI: Backend framework
- OpenAI: AI processing
- LlamaParse: Document parsing
- FAISS: Vector search for embeddings
- Python: Programming language
