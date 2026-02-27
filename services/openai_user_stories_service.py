import os
import json
import requests
import base64
import time
from datetime import datetime
from typing import Dict, List
from copy import deepcopy
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document
from docx.table import Table
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image
import pandas as pd
from io import BytesIO
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext, Settings
from llama_index.core.node_parser import MarkdownElementNodeParser
from openpyxl.utils import get_column_letter

load_dotenv()


class OpenAIUserStoriesService:
    """
    Enterprise-grade User Stories generator.
    Pipeline:
    BRD Documents → LlamaIndex RAG → GPT-5.1 (JSON) → Excel Export
    """

    def __init__(
        self,
        model: str = "gpt-5.1",
        max_output_tokens: int = 14000,
        temperature: float = 0.2,
        timeout: int = 600
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        # Initialize LlamaIndex components for RAG
        self.llm = LlamaOpenAI(model="gpt-5.1", api_key=api_key)
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=api_key)
        
        # Set global LlamaIndex settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

    # ------------------------------------------------------------------
    # GPT RESPONSE HANDLING
    # ------------------------------------------------------------------

    def _extract_text(self, response) -> str:
        """
        Extract text from OpenAI Responses API response object.
        Supports both .output_text and structured .output[].content[] shapes.
        """
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()

        texts = []
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))

        return "\n".join(texts).strip()

    # ------------------------------------------------------------------
    # PROMPT
    # ------------------------------------------------------------------

    def _system_prompt(self) -> str:
        """Load system prompt for user stories generation."""
        return """
You are a Senior Business Analyst with deep BFSI experience.
You specialize in converting Business Requirements into detailed User Stories.
You respond ONLY in valid JSON following the provided schema.
Use BFSI / CBUAE terminology and follow Agile best practices.
"""

    # ------------------------------------------------------------------
    # PROMPT TEMPLATES
    # ------------------------------------------------------------------

    def _user_prompt_template(self) -> str:
        """Template for user stories generation prompt."""
        return """
Generate user stories based on the provided BRD content for version {version}.

CRITICAL RULES:
- The "description" fields in the JSON SCHEMA are your specific instructions for what to generate for that key.
- Do NOT include the "description" keys in your JSON output; only return the data keys and their generated values.
- No markdown, no explanations.
- Use BFSI / CBUAE terminology.
- Use clear, actionable language.
- Do NOT invent information not present in the BRD content.
- Format all text content using dot bullet points (starting with "• ") instead of paragraphs.

=== BRD CONTENT ===
{brd_content}

=== VERSION ===
{version}

=== JSON SCHEMA ===
{schema_json}
"""

    # ------------------------------------------------------------------
    # RAG PIPELINE FOR BRD EMBEDDINGS
    # ------------------------------------------------------------------

    def create_brd_embeddings(self, brd_file_path: str, embeddings_path: str) -> bool:
        """
        Create embeddings for a BRD document using LlamaIndex.
        This enables RAG-based retrieval for better User Stories generation.
        """
        try:
            print(f"DEBUG: Creating embeddings for BRD: {brd_file_path}")
            
            # Use LlamaParse to extract content from BRD document
            llama_parser = LlamaParse(result_type="markdown", api_key=os.getenv("LLAMA_CLOUD_API_KEY"))
            documents = llama_parser.load_data(brd_file_path)
            print(f"DEBUG: Parsed {len(documents)} documents from BRD")

            # Use MarkdownElementNodeParser for better multimodal support
            node_parser = MarkdownElementNodeParser(llm=self.llm, num_workers=4)
            nodes = node_parser.get_nodes_from_documents(documents)
            base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
            print(f"DEBUG: Created {len(base_nodes)} base nodes and {len(objects)} objects")

            # Create VectorStoreIndex with multimodal support
            index = VectorStoreIndex(base_nodes + objects, embed_model=self.embed_model)
            print(f"DEBUG: Created VectorStoreIndex")

            # Persist embeddings
            index.storage_context.persist(embeddings_path)
            print(f"DEBUG: Persisted BRD embeddings to {embeddings_path}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to create BRD embeddings: {e}")
            import traceback
            traceback.print_exc()
            return False

    def retrieve_brd_context(self, embeddings_path: str, query: str, top_k: int = 30) -> str:
        """
        Retrieve relevant context from BRD embeddings using RAG.
        """
        try:
            if not os.path.exists(embeddings_path):
                print(f"DEBUG: No embeddings found at {embeddings_path}, using fallback")
                return ""

            # Load the index from storage
            storage_context = StorageContext.from_defaults(persist_dir=embeddings_path)
            index = load_index_from_storage(storage_context)
            
            # Retrieve relevant nodes
            retriever = index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
            
            # Extract context from retrieved nodes
            context_parts = []
            for i, node in enumerate(nodes[:10]):  # Limit to top 10
                if hasattr(node, 'text') and node.text:
                    context_parts.append(f"Context {i+1}: {node.text[:500]}...")
                elif hasattr(node, 'content') and node.content:
                    context_parts.append(f"Context {i+1}: {node.content[:500]}...")

            context = "\n\n".join(context_parts)
            print(f"DEBUG: Retrieved BRD context length: {len(context)} characters")
            
            return context
            
        except Exception as e:
            print(f"ERROR: Failed to retrieve BRD context: {e}")
            return ""

    # ------------------------------------------------------------------
    # JSON GENERATION
    # ------------------------------------------------------------------

    def generate_user_stories_json(self, brd_content: str, version: str, schema_json: str) -> Dict:
        user_prompt = self._user_prompt_template().format(
            brd_content=brd_content,
            version=version,
            schema_json=schema_json
        )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            max_output_tokens=self.max_output_tokens,
            timeout=self.timeout
        )

        raw = self._extract_text(response)

        if not raw:
            raise RuntimeError("GPT returned empty output")

        print(f"DEBUG: Raw LLM response preview: {raw[:500]}...")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from GPT: {e}\n\n{raw}")

        # Save raw response for debugging
        os.makedirs("llm_responses", exist_ok=True)
        with open(
            f"llm_responses/user_stories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return data

    # ------------------------------------------------------------------
    # EXCEL EXPORT
    # ------------------------------------------------------------------

    def export_to_excel(self, user_stories_data: Dict, output_path: str) -> str:
        """Export user stories to Excel with multiple sheets for stories, epics, and dependencies."""
        try:
            # Use a context manager to ensure the file is always closed properly
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                
                # --- SHEET 1: USER STORIES ---
                if 'user_stories' in user_stories_data:
                    expected_columns = [
                        'story_id', 'title', 'user_role', 'description', 
                        'acceptance_criteria', 'priority', 'effort_estimate', 
                        'brd_reference', 'version'
                    ]
                    
                    clean_stories_data = []
                    for story in user_stories_data['user_stories']:
                        clean_story = {}
                        for col in expected_columns:
                            value = story.get(col) or story.get(col.upper()) or story.get(col.title()) or story.get(col.capitalize())
                            if value and isinstance(value, str):
                                value = value.replace('•', '').strip()
                            clean_story[col] = value if value is not None else ""
                        clean_stories_data.append(clean_story)
                    
                    stories_df = pd.DataFrame(clean_stories_data)
                    stories_df.to_excel(writer, sheet_name='User Stories', index=False)
                    
                    # FORMATTING: Use column letters (A, B, C...) instead of names
                    worksheet = writer.sheets['User Stories']
                    column_widths = [15, 40, 25, 60, 60, 15, 20, 25, 15] # Widths matching expected_columns order
                    
                    for i, width in enumerate(column_widths):
                        col_letter = get_column_letter(i + 1)
                        worksheet.column_dimensions[col_letter].width = width

                # --- SHEET 2: EPICS ---
                if 'epics' in user_stories_data:
                    epics_df = pd.DataFrame(user_stories_data['epics'])
                    # Convert list values (like related_stories) to strings to prevent Excel errors
                    for col in epics_df.columns:
                        epics_df[col] = epics_df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
                    
                    epics_df.to_excel(writer, sheet_name='Epics', index=False)
                    
                    # Simple auto-width for Epics
                    worksheet = writer.sheets['Epics']
                    for i in range(len(epics_df.columns)):
                        worksheet.column_dimensions[get_column_letter(i + 1)].width = 30

                # --- SHEET 3: DEPENDENCIES ---
                if 'dependencies' in user_stories_data:
                    deps_df = pd.DataFrame(user_stories_data['dependencies'])
                    deps_df.to_excel(writer, sheet_name='Dependencies', index=False)
                    
            print(f"DEBUG: Excel successfully saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"ERROR: Failed to export to Excel: {e}")
            import traceback
            traceback.print_exc()
            # Ensure the potentially corrupt file is removed if it failed
            if os.path.exists(output_path):
                try: os.remove(output_path)
                except: pass
            raise RuntimeError(f"Excel export failed: {e}")

    # ------------------------------------------------------------------
    # ORCHESTRATOR
    # ------------------------------------------------------------------

    def generate_user_stories_excel(
        self,
        brd_content: str,
        version: str,
        schema_json: str,
        output_path: str
    ) -> str:
        """Generate user stories from BRD content and export to Excel."""
        data = self.generate_user_stories_json(brd_content, version, schema_json)
        return self.export_to_excel(data, output_path)