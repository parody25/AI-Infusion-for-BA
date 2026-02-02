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
        max_output_tokens: int = 12000,
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
        """Load system prompt from external file."""
        prompt_file = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'system_prompt.txt')
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            return (
                "You are a Senior Business Analyst with deep BFSI experience.\n"
                "You produce regulatory-compliant Business Requirement Specifications.\n"
                "You respond ONLY in valid JSON following the provided schema."
            )

    # ------------------------------------------------------------------
    # JSON GENERATION
    # ------------------------------------------------------------------

    def _load_user_prompt_template(self) -> str:
        """Load user prompt template from external file."""
        template_file = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'user_prompt_template.txt')
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            return """
You MUST return ONLY valid JSON that follows the structure of the JSON SCHEMA provided below.

CRITICAL RULES:
- The "description" fields in the JSON SCHEMA are your specific instructions for what to generate for that key.
- Do NOT include the "description" keys in your JSON output; only return the data keys and their generated values.
- No markdown, no explanations.
- Use BFSI / CBUAE terminology.
- Format all text content using dot bullet points (starting with "• ") instead of paragraphs.

=== REQUIREMENTS ===
{requirements}

=== CONTEXT ===
{context}

=== JSON SCHEMA ===
{schema_json}
"""

    def generate_brd_json(self, requirements: str, context: str, schema_json: str) -> Dict:
        user_prompt = self._load_user_prompt_template().format(
            requirements=requirements,
            context=context,
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

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from GPT: {e}\n\n{raw}")

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

    def _format_content(self, text):
        """Format text content with proper bullet points and numbered lists."""
        if not text:
            return ""
        
        lines = text.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with numbered list pattern (e.g., "1.", "2.", etc.)
            if line and line[0].isdigit() and len(line) > 1 and line[1] == '.':
                formatted_lines.append(line)
            # Check if line starts with bullet point pattern (e.g., "- ", "* ")
            elif line.startswith(('- ', '* ', '• ')):
                formatted_lines.append(line)
            # For single lines or paragraphs, convert to bullet points
            elif '\n' not in line and not line.startswith(('- ', '* ', '• ')) and not (line[0].isdigit() and len(line) > 1 and line[1] == '.'):
                formatted_lines.append(f"- {line}")
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    def fill_word_template(self, data: Dict, template_path: str, output_path: str) -> str:
        doc = Document(template_path)

        # ---------- DYNAMIC TABLES ----------
        try:
            br_table = self._find_table_by_placeholder(doc, "{req_id_bs}")
            if isinstance(data.get("business_requirements"), list):
                self._populate_business_requirements(doc, br_table, data["business_requirements"])
        except RuntimeError as e:
            print(f"WARNING: Business requirements table not found: {e}")

        try:
            tm_table = self._find_table_by_placeholder(doc, "{req_id_tm}")
            if isinstance(data.get("traceability_matrix"), list):
                self._populate_traceability(doc, tm_table, data["traceability_matrix"])
        except RuntimeError as e:
            print(f"WARNING: Traceability matrix table not found: {e}")

        try:
            try:
                toc_table = self._find_table_by_placeholder(doc, "{serial_number}")
            except RuntimeError:
                toc_table = self._find_table(doc, "Table Of Content")

            self._populate_table_of_contents(toc_table)
        except RuntimeError:
            print("WARNING: TOC table not found.")

        try:
            nfr_table = self._find_table_by_placeholder(doc, "{no_of_users}")
            self._populate_nfr(nfr_table, data)
        except RuntimeError as e:
            print(f"WARNING: NFR table not found: {e}")

        # ---------- SAFE PLACEHOLDER REPLACEMENT ----------
        flattened_data = self._get_flattened_data(data)

        def replace_text(text: str) -> str:
            for k, v in flattened_data.items():
                placeholder = f"{{{k}}}"
                if placeholder in text:
                    # Format the content if it's text content
                    formatted_value = self._format_content(str(v) if v is not None else "")
                    text = text.replace(placeholder, formatted_value)
            return text

        for p in doc.paragraphs:
            self._replace_text_in_runs(p, replace_text)

        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        self._replace_text_in_runs(para, replace_text)

        doc.save(output_path)
        return output_path

    # ------------------------------------------------------------------
    # FORMAT-SAFE RUN REPLACEMENT (FIX)
    # ------------------------------------------------------------------

    def _replace_text_in_runs(self, paragraph, replace_fn):
        for run in paragraph.runs:
            if run.text:
                run.text = replace_fn(run.text)

    # ------------------------------------------------------------------
    # TABLE HELPERS
    # ------------------------------------------------------------------

    def _find_table(self, doc, keyword: str):
        for table in doc.tables:
            if table.rows and keyword.lower() in table.rows[0].cells[0].text.lower():
                return table
        raise RuntimeError(f"Table not found: {keyword}")

    def _find_table_by_placeholder(self, doc, placeholder: str):
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if placeholder in cell.text:
                        return table
        raise RuntimeError(f"Table with placeholder '{placeholder}' not found")

    def _duplicate_table_after(self, base_table, parent_doc):
        new_tbl_xml = deepcopy(base_table._tbl)
        parent = base_table._tbl.getparent()

        spacer_p = parent_doc.add_paragraph()
        spacer_p._p.getparent().remove(spacer_p._p)

        base_table._tbl.addnext(spacer_p._p)
        spacer_p._p.addnext(new_tbl_xml)

        return Table(new_tbl_xml, base_table._parent)

    def _populate_business_requirements(self, doc, base_table, items):
        current_table = base_table

        for idx, req in enumerate(items):
            target_table = base_table if idx == 0 else self._duplicate_table_after(current_table, doc)
            current_table = target_table

            target_table.cell(0, 1).text = str(req.get("req_id_bs", ""))
            target_table.cell(1, 1).text = str(req.get("title_bs", ""))
            target_table.cell(2, 1).text = self._format_content(str(req.get("description_bs", "")))
            target_table.cell(3, 1).text = self._format_content(str(req.get("as_is_behaviour", "")))
            target_table.cell(4, 1).text = self._format_content(str(req.get("to_be_behaviour", "")))
            target_table.cell(5, 1).text = self._format_content(str(req.get("pre_requisite", "")))
            target_table.cell(6, 1).text = self._format_content(str(req.get("acceptance_criteria", "")))
            target_table.cell(7, 1).text = self._format_content(str(req.get("alternate_flows", "")))
            target_table.cell(8, 1).text = self._format_content(str(req.get("reference_documents", "")))

    def _populate_traceability(self, doc, base_table, items):
        current_table = base_table

        for idx, trace_item in enumerate(items):
            target_table = base_table if idx == 0 else self._duplicate_table_after(current_table, doc)
            current_table = target_table

            target_table.cell(0, 1).text = str(trace_item.get("req_id_tm", ""))
            target_table.cell(1, 1).text = self._format_content(str(trace_item.get("description_tm", "")))
            target_table.cell(2, 1).text = self._format_content(str(trace_item.get("source_channel", "")))
            target_table.cell(3, 1).text = self._format_content(str(trace_item.get("impacted_system", "")))
            target_table.cell(4, 1).text = self._format_content(str(trace_item.get("outcome", "")))

    def _populate_table_of_contents(self, table):
        toc_items = [
            "Document Sign off", "Document History", "Overview", "Current constraints",
            "Objective", "In scope", "Out of scope", "Description",
            "Business Requirements", "Requirement Traceability Matrix",
            "Non-Functional Requirements", "Impact on Operational Process",
            "Regulatory Impact", "Reports Requirement", "Access Requirement",
            "Security Requirement", "Data Requirement", "Training Requirement"
        ]

        self._clear_table_keep_header(table)

        for i, section in enumerate(toc_items, 1):
            row = table.add_row().cells
            row[0].text = str(i)
            row[1].text = section

    def _clear_table_keep_header(self, table):
        while len(table.rows) > 1:
            table._tbl.remove(table.rows[1]._tr)

    def _get_flattened_data(self, data: Dict) -> Dict:
        flat = {}
        if "document" in data:
            flat.update(data["document"])

        nfr = data.get("non_functional_requirements", {})
        if isinstance(nfr, list) and nfr:
            flat.update(nfr[0])

        for key in [
            "impact_on_operational_process", "regulatory_impact",
            "reports_requirement", "access_requirement",
            "security_requirement", "data_requirement",
            "training_requirement", "open_questions", 
            "contradictions_found_in_input_documents"
        ]:
            if key in data:
                flat[key] = data[key]
        
        flat["date_today"] = datetime.today().strftime("%d-%b-%Y")
        return flat

    def _populate_nfr(self, table, data):
        nfr = data.get("non_functional_requirements", {})
        if isinstance(nfr, list) and nfr:
            nfr = nfr[0]

        table.cell(0, 1).text = self._format_content(str(nfr.get("no_of_users", "")))
        table.cell(1, 1).text = self._format_content(str(nfr.get("peak_volume", "")))
        table.cell(2, 1).text = self._format_content(str(nfr.get("monthly_volume", "")))
        table.cell(3, 1).text = self._format_content(str(nfr.get("availability", "")))

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