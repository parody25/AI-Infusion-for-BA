import os
import json
from datetime import datetime
from typing import Dict
from copy import deepcopy
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document
from docx.table import Table
from brd_schema import BRD_SCHEMA_JSON_STRING

load_dotenv()


class OpenAIBRDService:
    """
    Enterprise-grade BRD generator.
    Pipeline:
    Documents → Embeddings → GPT-5.1 (JSON) → Word Template
    """

    def __init__(
        self,
        model: str = "gpt-5.1",
        max_output_tokens: int = 15000,
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

    # ------------------------------------------------------------------
    # GPT RESPONSE HANDLING
    # ------------------------------------------------------------------

    def _extract_text(self, response) -> str:
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
        return (
            "You are a Senior Business Analyst with deep BFSI experience.\n"
            "You produce regulatory-compliant Business Requirement Specifications.\n"
            "You respond ONLY in valid JSON following the provided schema."
        )

    # ------------------------------------------------------------------
    # JSON GENERATION (SOURCE OF TRUTH)
    # ------------------------------------------------------------------

    def generate_brd_json(self, requirements: str, context: str, schema_json: str) -> Dict:
        user_prompt = f"""
You MUST return ONLY valid JSON matching the schema below.

CRITICAL RULES:
- No markdown
- No explanations
- No extra keys
- No missing keys
- Arrays must contain multiple objects where applicable
- Use BFSI / CBUAE terminology
- Use clear "shall" statements
- Do NOT invent information

=== REQUIREMENTS ===
{requirements}

=== CONTEXT ===
{context}

=== JSON SCHEMA ===
{schema_json}
"""

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

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from GPT: {e}\n\n{raw}")

        # Persist JSON for audit/debug
        os.makedirs("llm_responses", exist_ok=True)
        with open(
            f"llm_responses/brd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return data

    # ------------------------------------------------------------------
    # WORD TEMPLATE POPULATION
    # ------------------------------------------------------------------

    def fill_word_template(self, data: Dict, template_path: str, output_path: str) -> str:
        doc = Document(template_path)

        # 1. POPULATE DYNAMIC TABLES FIRST
        # This ensures we use the "clean" template with placeholders still intact.

        # ---------- BUSINESS REQUIREMENTS TABLE ----------
        try:
            # Search for the table containing the specific placeholder
            br_table = self._find_table_by_placeholder(doc, "{req_id_bs}")
            if "business_requirements" in data and isinstance(data["business_requirements"], list):
                self._populate_business_requirements(doc, br_table, data["business_requirements"])
        except RuntimeError as e:
            print(f"WARNING: Business requirements table not found: {e}")

        # ---------- TRACEABILITY MATRIX ----------
        try:
            tm_table = self._find_table_by_placeholder(doc, "{req_id_tm}")
            if "traceability_matrix" in data and isinstance(data["traceability_matrix"], list):
                self._populate_traceability(doc, tm_table, data["traceability_matrix"])
        except RuntimeError as e:
            print(f"WARNING: Traceability matrix table not found: {e}")

        # ---------- TABLE OF CONTENTS ----------
        try:
            # Identify TOC by checking for "Table of Content" heading or a specific text
            toc_table = self._find_table_by_placeholder(doc, "Table Of Content")
            toc_items = data.get("document", {}).get("table_of_contents", [])
            if toc_items:
                self._populate_table_of_contents(toc_table, toc_items)
        except RuntimeError:
            # Fallback if TOC doesn't have the string inside the table
            print("WARNING: TOC table not found via placeholder.")

        # ---------- NON-FUNCTIONAL REQUIREMENTS ----------
        try:
            nfr_table = self._find_table_by_placeholder(doc, "{no_of_users}")
            self._populate_nfr(nfr_table, data)
        except RuntimeError as e:
            print(f"WARNING: Non-functional requirements table not found: {e}")

        # 2. PERFORM GLOBAL PLACEHOLDER REPLACEMENT LAST
        # Now we replace document-level fields (title, id, overview) everywhere else.
        flattened_data = self._get_flattened_data(data)

        def replace_text(text: str) -> str:
            for k, v in flattened_data.items():
                placeholder = f"{{{k}}}"
                if placeholder in text:
                    text = text.replace(placeholder, str(v) if v is not None else "")
            return text

        for p in doc.paragraphs:
            p.text = replace_text(p.text)

        # Handle remaining tables (TOC and NFR may have already been processed above)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        para.text = replace_text(para.text)

        doc.save(output_path)
        return output_path

    # ------------------------------------------------------------------
    # TABLE HELPERS
    # ------------------------------------------------------------------

    def _find_table(self, doc, keyword: str):
        for table in doc.tables:
            if keyword.lower() in table.rows[0].cells[0].text.lower():
                return table
        raise RuntimeError(f"Table not found: {keyword}")

    def _find_table_by_placeholder(self, doc, placeholder: str):
        """Finds a table that contains the specific placeholder string."""
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if placeholder in cell.text:
                        return table
        raise RuntimeError(f"Table with placeholder '{placeholder}' not found")

    def _duplicate_table_after(self, base_table, parent_doc):
        """
        Creates a deep copy of a table, inserts a spacer paragraph after the original,
        and then inserts the new table after the spacer.
        Returns the new Table object.
        """
        new_tbl_xml = deepcopy(base_table._tbl)
        parent = base_table._tbl.getparent()

        # Create spacer
        spacer_p = parent_doc.add_paragraph()
        spacer_p._p.getparent().remove(spacer_p._p)

        # Insert after base table
        base_table._tbl.addnext(spacer_p._p)
        spacer_p._p.addnext(new_tbl_xml)

        # Return new Table object wrapped in the same parent as the original
        return Table(new_tbl_xml, base_table._parent)

    def _populate_business_requirements(self, doc, base_table, items):
        """Populate business requirements using robust cloning."""
        print(f"DEBUG: Populating business requirements with {len(items)} items")

        # We use 'current_table' to track where we are inserting.
        # Initially, it's the base template table.
        current_table = base_table

        for idx, req in enumerate(items):
            if idx == 0:
                # First item goes into the existing template table
                target_table = base_table
            else:
                # Subsequent items get a new cloned table inserted AFTER the current one
                target_table = self._duplicate_table_after(current_table, doc)
                # Update current_table so the next loop inserts after THIS one
                current_table = target_table

            # Populate the data
            # Note: Ensure your row indices match your specific template structure.
            # Based on your BRD, it's a vertical Key-Value table.
            try:
                target_table.cell(0, 1).text = str(req.get("req_id_bs", ""))
                target_table.cell(1, 1).text = str(req.get("title_bs", ""))
                target_table.cell(2, 1).text = str(req.get("description_bs", ""))
                target_table.cell(3, 1).text = str(req.get("as_is_behaviour", ""))
                target_table.cell(4, 1).text = str(req.get("to_be_behaviour", ""))
                target_table.cell(5, 1).text = str(req.get("pre_requisite", ""))
                target_table.cell(6, 1).text = str(req.get("acceptance_criteria", ""))
                target_table.cell(7, 1).text = str(req.get("alternate_flows", ""))
            except IndexError:
                print(f"WARNING: Table structure mismatch for item {idx}")

        print(f"DEBUG: Successfully created {len(items)} business requirements tables")

    def _populate_traceability(self, doc, base_table, items):
        """Populate traceability matrix using robust cloning."""
        print(f"DEBUG: Populating traceability matrix with {len(items)} items")

        current_table = base_table

        for idx, trace_item in enumerate(items):
            if idx == 0:
                target_table = base_table
            else:
                target_table = self._duplicate_table_after(current_table, doc)
                current_table = target_table

            try:
                target_table.cell(0, 1).text = str(trace_item.get("req_id_tm", ""))
                target_table.cell(1, 1).text = str(trace_item.get("description_tm", ""))
                target_table.cell(2, 1).text = str(trace_item.get("source_channel", ""))
                target_table.cell(3, 1).text = str(trace_item.get("impacted_system", ""))
                target_table.cell(4, 1).text = str(trace_item.get("outcome", ""))
            except IndexError:
                 print(f"WARNING: Traceability table structure mismatch for item {idx}")

        print(f"DEBUG: Successfully created {len(items)} traceability matrix tables")

    def _populate_table_of_contents(self, table, toc_items):
        """Populate Table of Contents as a horizontal table with rows (not cloned tables)."""
        print(f"DEBUG: Populating Table of Contents with {len(toc_items)} entries")

        # Clear existing rows but keep header
        self._clear_table_keep_header(table)

        # Add a row for each TOC entry
        for item in toc_items:
            row = table.add_row().cells
            row[0].text = str(item)

        print(f"DEBUG: Table of Contents now has {len(table.rows)} rows (1 header + {len(toc_items)} entries)")

    def _clear_table_keep_header(self, table):
        """Clear all rows except the header row."""
        while len(table.rows) > 1:
            table._tbl.remove(table.rows[1]._tr)

    def _get_flattened_data(self, data: Dict) -> Dict:
        """Helper to collect document-level fields for global replacement."""
        flat = {}
        if "document" in data:
            flat.update(data["document"])

        # Add NFR fields to flat data for replacement in the NFR table/paragraphs
        nfr = data.get("non_functional_requirements", {})
        if isinstance(nfr, list) and nfr: nfr = nfr[0]
        if isinstance(nfr, dict): flat.update(nfr)

        # Add other top-level fields
        for key in ["impact_on_operational_process", "regulatory_impact", "reports_requirement",
                    "access_requirement", "security_requirement", "data_requirement", "training_requirement"]:
            if key in data: flat[key] = data[key]
        return flat

    def _populate_nfr(self, table, data):
        # Handle NFR data - could be array or object
        nfr = data.get("non_functional_requirements", {})
        if isinstance(nfr, list) and nfr:
            nfr = nfr[0] if isinstance(nfr[0], dict) else nfr

        # Populate table cells (first 4 rows)
        table.cell(0, 1).text = str(nfr.get("no_of_users", "")) if isinstance(nfr, dict) else ""
        table.cell(1, 1).text = str(nfr.get("peak_volume", "")) if isinstance(nfr, dict) else ""
        table.cell(2, 1).text = str(nfr.get("monthly_volume", "")) if isinstance(nfr, dict) else ""
        table.cell(3, 1).text = str(nfr.get("availability", "")) if isinstance(nfr, dict) else ""

        # The remaining NFR fields are handled by placeholder replacement in paragraphs
        # (impact_on_operational_process, regulatory_impact, etc.)

    # ------------------------------------------------------------------
    # ORCHESTRATOR
    # ------------------------------------------------------------------

    def generate_brd_word(
        self,
        requirements: str,
        context: str,
        schema_json: str,
        template_path: str,
        output_path: str
    ) -> str:
        data = self.generate_brd_json(requirements, context, schema_json)
        return self.fill_word_template(data, template_path, output_path)
