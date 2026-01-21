from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import json
import tempfile
import shutil
from datetime import datetime
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext, Settings
from llama_index.core.node_parser import MarkdownElementNodeParser
from services.openai_brd_service import OpenAIBRDService
from typing import List, Dict

load_dotenv()

app = FastAPI(title="AI Infusion for BA", description="POC for enhancing BRD quality with AI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
llama_parse = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))
llm = LlamaOpenAI(model="gpt-5.1", api_key=os.getenv("OPENAI_API_KEY"))
embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY"))

# CRITICAL FIX: Set global LlamaIndex settings to ensure consistent embedding model
Settings.embed_model = embed_model
Settings.llm = llm

brd_service = OpenAIBRDService()

# Directory for storing BRD projects
PROJECTS_DIR = "brd_projects"

# Dummy BRD Template with placeholders
BRD_TEMPLATE = """
# Business Requirements Document (BRD)

## 1. Introduction
{introduction}

## 2. Business Objectives
{business_objectives}

## 3. Scope
### In-Scope
{in_scope}

### Out-of-Scope
{out_of_scope}

## 4. Business Requirements
{business_requirements}

## 5. Functional Requirements
{functional_requirements}

## 6. Non-Functional Requirements
{non_functional_requirements}

## 7. Assumptions & Constraints
{assumptions_constraints}

## 8. Risks & Dependencies
{risks_dependencies}

## 9. Acceptance Criteria
{acceptance_criteria}

## 10. Open Questions
{open_questions}
"""

if not os.path.exists(PROJECTS_DIR):
    os.makedirs(PROJECTS_DIR)

def get_project_path(project_id: str) -> str:
    """Get the path for a BRD project directory."""
    return os.path.join(PROJECTS_DIR, project_id)

def load_project_metadata(project_id: str) -> Dict:
    """Load project metadata including document and BRD lists."""
    project_path = get_project_path(project_id)
    metadata_path = os.path.join(project_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            # Ensure brds array exists for backward compatibility
            if "brds" not in metadata:
                metadata["brds"] = []
            return metadata
    return {"documents": [], "brds": []}

def save_project_metadata(project_id: str, metadata: Dict):
    """Save project metadata."""
    project_path = get_project_path(project_id)
    os.makedirs(project_path, exist_ok=True)
    metadata_path = os.path.join(project_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

@app.post("/create_project")
async def create_brd_project(name: str = None):
    """Create a new BRD project with optional name for easy reference."""
    project_id = str(uuid.uuid4())
    project_path = get_project_path(project_id)
    os.makedirs(project_path, exist_ok=True)

    # Initialize metadata with optional name
    metadata = {
        "documents": [],
        "name": name,
        "created_at": datetime.now().isoformat()
    }
    save_project_metadata(project_id, metadata)

    response = {
        "project_id": project_id,
        "message": "BRD project created successfully"
    }
    if name:
        response["name"] = name

    return JSONResponse(content=response)

@app.get("/projects")
async def list_projects():
    """List all BRD projects with their names and metadata."""
    if not os.path.exists(PROJECTS_DIR):
        return JSONResponse(content={"projects": []})

    projects = []
    for item in os.listdir(PROJECTS_DIR):
        project_path = os.path.join(PROJECTS_DIR, item)
        if os.path.isdir(project_path):
            metadata = load_project_metadata(item)
            projects.append({
                "project_id": item,
                "name": metadata.get("name"),
                "document_count": len(metadata.get("documents", [])),
                "brd_count": len(metadata.get("brds", [])),
                "created_at": metadata.get("created_at")
            })

    return JSONResponse(content={"projects": projects})

@app.post("/projects/{project_id}/upload")
async def upload_document_to_project(project_id: str, file: UploadFile = File(...)):
    """Upload a document to a specific BRD project using LlamaIndex for efficient embeddings."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check if project exists
    project_path = get_project_path(project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="BRD project not found")

    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        file_name = os.path.basename(file.filename)
        document_name, file_extension = os.path.splitext(file_name)
        document_id = f"{project_id}_{document_name}"

        # Create project embeddings directory
        project_embedding_dir = os.path.join(project_path, "embeddings")
        os.makedirs(project_embedding_dir, exist_ok=True)

        embeddings_file = os.path.join(project_embedding_dir, f"{file_name}_embeddings")

        print(f"DEBUG: Processing {file_name} for project {project_id}")
        print(f"DEBUG: Embeddings file path: {embeddings_file}")

        if file_extension.lower() in ['.pdf', '.docx', '.xlsx']:
            try:
                # Check if embeddings exist and are compatible
                recreate_embeddings = True
                if os.path.exists(embeddings_file):
                    try:
                        # Try to load and test the existing index
                        storage_context = StorageContext.from_defaults(persist_dir=embeddings_file)
                        test_index = load_index_from_storage(storage_context)
                        # Test with a dummy query to check compatibility
                        test_retriever = test_index.as_retriever(similarity_top_k=1)
                        test_nodes = test_retriever.retrieve("test")
                        print(f"DEBUG: Existing embeddings for {file_name} are compatible, reusing")
                        recreate_embeddings = False
                    except Exception as e:
                        print(f"DEBUG: Existing embeddings incompatible ({str(e)}), will recreate")
                        import shutil
                        shutil.rmtree(embeddings_file)
                        recreate_embeddings = True
                else:
                    print(f"DEBUG: No existing embeddings found for {file_name}, creating new")

                if recreate_embeddings:
                    # Create new embeddings using LlamaIndex
                    print(f"DEBUG: Creating new embeddings for {file_name}")

                    # Use LlamaParse with markdown result type for better structure
                    llama_parser = LlamaParse(result_type="markdown", api_key=os.getenv("LLAMA_CLOUD_API_KEY"))
                    documents = llama_parser.load_data(temp_file_path)
                    print(f"DEBUG: Parsed {len(documents)} documents from {file_name}")

                    # Use MarkdownElementNodeParser for better multimodal support
                    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=4)  # Reduced workers for stability
                    nodes = node_parser.get_nodes_from_documents(documents)
                    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
                    print(f"DEBUG: Created {len(base_nodes)} base nodes and {len(objects)} objects")

                    # Create VectorStoreIndex with multimodal support
                    index = VectorStoreIndex(base_nodes + objects, embed_model=embed_model)
                    print(f"DEBUG: Created VectorStoreIndex")

                    # Persist embeddings
                    index.storage_context.persist(embeddings_file)
                    print(f"DEBUG: Persisted embeddings to {embeddings_file}")

                # Update project metadata
                project_metadata = load_project_metadata(project_id)
                doc_info = {
                    "id": str(uuid.uuid4()),
                    "filename": file_name,
                    "uploaded_at": datetime.now().isoformat(),
                    "type": "business_document",
                    "embedding_path": embeddings_file
                }
                project_metadata["documents"].append(doc_info)
                save_project_metadata(project_id, project_metadata)

                print(f"DEBUG: Successfully processed {file_name} for project {project_id}")
                return JSONResponse(content={
                    "message": "Document uploaded and processed successfully",
                    "document_id": doc_info["id"],
                    "embedding_created": True
                })

            except Exception as e:
                print(f"ERROR: Failed to create embeddings for {file_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF, DOCX, and XLSX are supported.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/projects/{project_id}/documents")
async def list_project_documents(project_id: str):
    """List all documents uploaded to a BRD project."""
    project_path = get_project_path(project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="BRD project not found")

    metadata = load_project_metadata(project_id)
    return JSONResponse(content={"documents": metadata["documents"]})

@app.delete("/projects/{project_id}/documents/{document_id}")
async def delete_project_document(project_id: str, document_id: str):
    """Delete a document from a BRD project."""
    project_path = get_project_path(project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="BRD project not found")

    metadata = load_project_metadata(project_id)

    # Find and remove document from metadata
    doc_to_remove = None
    for doc in metadata["documents"]:
        if doc["id"] == document_id:
            doc_to_remove = doc
            break

    if not doc_to_remove:
        raise HTTPException(status_code=404, detail="Document not found in project")

    # Note: For POC, we're not rebuilding the vectorstore after deletion
    # In production, you'd need to rebuild or mark chunks as deleted
    metadata["documents"].remove(doc_to_remove)
    save_project_metadata(project_id, metadata)

    return JSONResponse(content={"message": "Document deleted successfully"})

@app.post("/projects/{project_id}/generate_brd")
async def generate_brd_for_project(project_id: str, requirements: str):
    """Generate BRD for a specific project, store it in the project, and return metadata."""
    print(f"DEBUG: Starting BRD generation for project {project_id}")
    print(f"DEBUG: Requirements: {requirements}")

    project_path = get_project_path(project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="BRD project not found")

    # Check if project has documents
    metadata = load_project_metadata(project_id)
    if not metadata["documents"]:
        raise HTTPException(status_code=400, detail="No documents uploaded to this project yet")

    print(f"DEBUG: Found {len(metadata['documents'])} documents in project")

    # Collect all embedding paths and document IDs for this project
    embedding_paths = []
    input_document_ids = []
    for doc in metadata["documents"]:
        embedding_path = doc.get("embedding_path")
        if embedding_path and os.path.exists(embedding_path):
            embedding_paths.append(embedding_path)
            input_document_ids.append(doc["id"])

    print(f"DEBUG: Found {len(embedding_paths)} valid embedding paths")
    print(f"DEBUG: Input document IDs: {input_document_ids}")

    if not embedding_paths:
        raise HTTPException(status_code=500, detail="No valid embeddings found for project")

    # Load and combine indices from all documents
    combined_nodes = []
    for embedding_path in embedding_paths:
        try:
            print(f"DEBUG: Loading index from {embedding_path}")
            storage_context = StorageContext.from_defaults(persist_dir=embedding_path)
            index = load_index_from_storage(storage_context)
            print(f"DEBUG: Successfully loaded index")

            # Get nodes from index
            retriever = index.as_retriever(similarity_top_k=10)
            nodes = retriever.retrieve(requirements)
            print(f"DEBUG: Retrieved {len(nodes)} nodes from {embedding_path}")
            combined_nodes.extend(nodes)
        except Exception as e:
            print(f"ERROR: Failed to load/process index from {embedding_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"DEBUG: Total combined nodes: {len(combined_nodes)}")

    # Extract context from retrieved nodes
    context_parts = []
    for i, node in enumerate(combined_nodes[:10]):  # Limit to top 10
        if hasattr(node, 'text') and node.text:
            context_parts.append(f"Context {i+1}: {node.text[:500]}...")  # Truncate for logging
        elif hasattr(node, 'content') and node.content:
            context_parts.append(f"Context {i+1}: {node.content[:500]}...")

    context = "\n\n".join(context_parts)
    print(f"DEBUG: Final context length: {len(context)} characters")

    if not context.strip():
        context = "No relevant context found in uploaded documents."
        print("WARNING: No context found, using default message")

    print("DEBUG: Calling BRD service to generate content")

    # Create BRDs directory in project
    brds_dir = os.path.join(project_path, "brds")
    os.makedirs(brds_dir, exist_ok=True)

    # Generate unique BRD ID and filename
    brd_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"BRD_{timestamp}_{brd_id[:8]}.docx"
    output_path = os.path.join(brds_dir, output_filename)

    # Generate and fill Word document
    filled_path = brd_service.generate_brd_word(
        requirements=requirements,
        context=context,
        template_path="BRD_Template.docx",
        output_path=output_path
    )

    print(f"DEBUG: BRD generated successfully at {filled_path}")

    # Record BRD in project metadata
    brd_info = {
        "id": brd_id,
        "filename": output_filename,
        "file_path": output_path,
        "generated_at": datetime.now().isoformat(),
        "requirements": requirements,
        "input_document_ids": input_document_ids,
        "document_count": len(input_document_ids)
    }

    metadata["brds"].append(brd_info)
    save_project_metadata(project_id, metadata)

    print(f"DEBUG: BRD recorded in metadata: {brd_id}")

    return JSONResponse(content={
        "message": "BRD generated and stored successfully",
        "brd_id": brd_id,
        "filename": output_filename,
        "input_documents_used": len(input_document_ids),
        "generated_at": brd_info["generated_at"]
    })

@app.get("/projects/{project_id}/brds")
async def list_project_brds(project_id: str):
    """List all BRDs generated for a project."""
    project_path = get_project_path(project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="BRD project not found")

    metadata = load_project_metadata(project_id)
    brds = metadata.get("brds", [])

    # Add document filenames for better UX
    documents = metadata.get("documents", [])
    doc_id_to_filename = {doc["id"]: doc["filename"] for doc in documents}

    for brd in brds:
        input_doc_filenames = []
        for doc_id in brd.get("input_document_ids", []):
            filename = doc_id_to_filename.get(doc_id, f"Unknown (ID: {doc_id})")
            input_doc_filenames.append(filename)
        brd["input_document_filenames"] = input_doc_filenames

    return JSONResponse(content={"brds": brds})

@app.get("/projects/{project_id}/brds/{brd_id}/download")
async def download_project_brd(project_id: str, brd_id: str):
    """Download a specific BRD from a project."""
    project_path = get_project_path(project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="BRD project not found")

    metadata = load_project_metadata(project_id)
    brds = metadata.get("brds", [])

    # Find the BRD
    brd_info = None
    for brd in brds:
        if brd["id"] == brd_id:
            brd_info = brd
            break

    if not brd_info:
        raise HTTPException(status_code=404, detail="BRD not found in project")

    file_path = brd_info["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="BRD file not found on disk")

    return FileResponse(
        path=file_path,
        filename=brd_info["filename"],
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )

@app.delete("/projects/{project_id}/brds/{brd_id}")
async def delete_project_brd(project_id: str, brd_id: str):
    """Delete a BRD from a project."""
    project_path = get_project_path(project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="BRD project not found")

    metadata = load_project_metadata(project_id)
    brds = metadata.get("brds", [])

    # Find and remove BRD from metadata
    brd_to_remove = None
    for brd in brds:
        if brd["id"] == brd_id:
            brd_to_remove = brd
            break

    if not brd_to_remove:
        raise HTTPException(status_code=404, detail="BRD not found in project")

    # Remove the file if it exists
    file_path = brd_to_remove.get("file_path")
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    # Remove from metadata
    metadata["brds"].remove(brd_to_remove)
    save_project_metadata(project_id, metadata)

    return JSONResponse(content={"message": "BRD deleted successfully"})

@app.get("/brd_template")
async def get_brd_template():
    """Get the BRD template structure."""
    return JSONResponse(content={"template": BRD_TEMPLATE})

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
