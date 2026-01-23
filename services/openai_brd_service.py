import os
import json
from datetime import datetime
from typing import Dict
from copy import deepcopy
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document
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

        # Flatten the nested JSON structure for placeholder replacement
        flattened_data = {}

        # Extract document-level fields
        if "document" in data:
            flattened_data.update(data["document"])

        # Extract single values from arrays (first item) for placeholder replacement
        # Business requirements - take first item for placeholders
        if "business_requirements" in data and isinstance(data["business_requirements"], list) and data["business_requirements"]:
            first_req = data["business_requirements"][0]
            flattened_data.update({
                "req_id_bs": first_req.get("req_id_bs", ""),
                "title_bs": first_req.get("title_bs", ""),
                "description_bs": first_req.get("description_bs", ""),
                "as_is_behaviour": first_req.get("as_is_behaviour", ""),
                "to_be_behaviour": first_req.get("to_be_behaviour", ""),
                "pre_requisite": first_req.get("pre_requisite", ""),
                "acceptance_criteria": first_req.get("acceptance_criteria", ""),
                "alternate_flows": first_req.get("alternate_flows", ""),
                "reference_documents": first_req.get("reference_documents", "")
            })

        # Traceability matrix - take first item for placeholders
        if "traceability_matrix" in data and isinstance(data["traceability_matrix"], list) and data["traceability_matrix"]:
            first_trace = data["traceability_matrix"][0]
            flattened_data.update({
                "req_id_tm": first_trace.get("req_id_tm", ""),
                "description_tm": first_trace.get("description_tm", ""),
                "source_channel": first_trace.get("source_channel", ""),
                "impacted_system": first_trace.get("impacted_system", ""),
                "outcome": first_trace.get("outcome", "")
            })

        # Non-functional requirements - handle mixed structure
        if "non_functional_requirements" in data:
            nfr = data["non_functional_requirements"]
            if isinstance(nfr, list) and nfr:
                # If it's an array, take first item
                nfr = nfr[0] if isinstance(nfr[0], dict) else nfr
            if isinstance(nfr, dict):
                flattened_data.update(nfr)
            # Also handle top-level NFR fields
            for key in ["impact_on_operational_process", "regulatory_impact", "reports_requirement",
                       "access_requirement", "security_requirement", "data_requirement", "training_requirement"]:
                if key in data:
                    flattened_data[key] = data[key]

        # Ensure all values are strings (not lists or other types)
        for key, value in flattened_data.items():
            if not isinstance(value, str):
                flattened_data[key] = str(value) if value is not None else ""

        # ---------- SIMPLE PLACEHOLDERS ----------
        def replace(text: str) -> str:
            for k, v in flattened_data.items():
                text = text.replace(f"{{{k}}}", v or "")
            return text

        for p in doc.paragraphs:
            p.text = replace(p.text)

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        p.text = replace(p.text)

        # ---------- BUSINESS REQUIREMENTS TABLE ----------
        try:
            br_table = self._find_table(doc, "Business requirements")
            if "business_requirements" in data and isinstance(data["business_requirements"], list):
                print(f"DEBUG: Found {len(data['business_requirements'])} business requirements to populate")
                self._populate_business_requirements(br_table, data["business_requirements"])
        except RuntimeError as e:
            print(f"WARNING: Business requirements table not found in template: {e}")

        # ---------- TRACEABILITY MATRIX ----------
        try:
            tm_table = self._find_table(doc, "Requirement Traceability Matrix")
            if "traceability_matrix" in data and isinstance(data["traceability_matrix"], list):
                print(f"DEBUG: Found {len(data['traceability_matrix'])} traceability items to populate")
                self._populate_traceability(tm_table, data["traceability_matrix"])
        except RuntimeError as e:
            print(f"WARNING: Traceability matrix table not found in template: {e}")

        # ---------- NON-FUNCTIONAL REQUIREMENTS ----------
        try:
            nfr_table = self._find_table(doc, "Non-functional Requirements")
            self._populate_nfr(nfr_table, data)
        except RuntimeError as e:
            print(f"WARNING: Non-functional requirements table not found in template: {e}")

        # ---------- TABLE OF CONTENTS ----------
        try:
            toc_table = self._find_table(doc, "Table Of Content")
            toc_items = data.get("table_of_contents", [])
            if isinstance(toc_items, list) and toc_items:
                print(f"DEBUG: Populating TOC with {len(toc_items)} entries")
                self._populate_table_of_contents(toc_table, toc_items)
        except RuntimeError as e:
            print(f"WARNING: Table of Contents table not found in template: {e}")

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

    def _clone_table_after(self, doc, table):
        """Clone a table and insert it after the original table."""
        tbl_xml = deepcopy(table._tbl)
        table._tbl.addnext(tbl_xml)

        # Find the newly added table
        new_tables = doc.tables
        for t in reversed(new_tables):
            if t != table:  # Find the last table that isn't the original
                return t
        return None

    def _remove_duplicate_tables(self, doc, keyword):
        """Remove all tables with the given keyword except the first one."""
        tables = [
            (i, t) for i, t in enumerate(doc.tables)
            if keyword.lower() in t.rows[0].cells[0].text.lower()
        ]

        # Keep the first table, remove the rest
        if len(tables) > 1:
            for idx, table in reversed(tables[1:]):  # Remove from last to first
                table._tbl.getparent().remove(table._tbl)
                print(f"DEBUG: Removed duplicate {keyword} table")

    def _populate_business_requirements(self, base_table, items):
        """Populate business requirements as vertical key-value tables (clone approach)."""
        print(f"DEBUG: Populating business requirements with {len(items)} items using table cloning")

        # Remove any previously duplicated tables
        doc = base_table._tbl.getparent().getparent().getparent()  # Get document from table
        self._remove_duplicate_tables(doc, "Business requirements")

        for idx, req in enumerate(items):
            if idx == 0:
                # Use the base table for the first requirement
                table = base_table
            else:
                # Clone the base table for additional requirements
                table = self._clone_table_after(doc, base_table)
                if table is None:
                    print(f"WARNING: Failed to clone business requirements table for item {idx}")
                    continue

            print(f"DEBUG: Populating business requirements table {idx+1} with: {req.get('req_id_bs', 'Unknown')}")

            # Populate the vertical key-value table
            table.cell(0, 1).text = str(req.get("req_id_bs", ""))
            table.cell(1, 1).text = str(req.get("title_bs", ""))
            table.cell(2, 1).text = str(req.get("description_bs", ""))
            table.cell(3, 1).text = str(req.get("as_is_behaviour", ""))
            table.cell(4, 1).text = str(req.get("to_be_behaviour", ""))
            table.cell(5, 1).text = str(req.get("pre_requisite", ""))
            table.cell(6, 1).text = str(req.get("acceptance_criteria", ""))
            table.cell(7, 1).text = str(req.get("alternate_flows", ""))

        print(f"DEBUG: Created {len(items)} business requirements tables")

    def _populate_traceability(self, base_table, items):
        """Populate traceability matrix as vertical key-value tables (clone approach)."""
        print(f"DEBUG: Populating traceability matrix with {len(items)} items using table cloning")

        # Remove any previously duplicated tables
        doc = base_table._tbl.getparent().getparent().getparent()  # Get document from table
        self._remove_duplicate_tables(doc, "Requirement Traceability Matrix")

        for idx, trace_item in enumerate(items):
            if idx == 0:
                # Use the base table for the first requirement
                table = base_table
            else:
                # Clone the base table for additional requirements
                table = self._clone_table_after(doc, base_table)
                if table is None:
                    print(f"WARNING: Failed to clone traceability table for item {idx}")
                    continue

            print(f"DEBUG: Populating traceability table {idx+1} with: {trace_item.get('req_id_tm', 'Unknown')}")

            # Populate the vertical key-value table
            table.cell(0, 1).text = str(trace_item.get("req_id_tm", ""))
            table.cell(1, 1).text = str(trace_item.get("description_tm", ""))
            table.cell(2, 1).text = str(trace_item.get("source_channel", ""))
            table.cell(3, 1).text = str(trace_item.get("impacted_system", ""))
            table.cell(4, 1).text = str(trace_item.get("outcome", ""))

        print(f"DEBUG: Created {len(items)} traceability matrix tables")

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
