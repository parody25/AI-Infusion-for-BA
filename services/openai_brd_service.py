import os
import shutil
from typing import Optional, List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document

load_dotenv()


class OpenAIBRDService:
    """
    Service class for generating Business Requirement Documents (BRD)
    using GPT-5 via OpenAI Responses API.
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
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.timeout = timeout

    def _extract_text_from_response(self, response) -> str:
        """
        Robustly extract text from GPT-5 / GPT-5.1 Responses API output.
        """
        # Preferred shortcut (may be empty)
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()

        # Fallback: walk structured output
        texts = []
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))

        return "\n".join(texts).strip()

    def _build_system_prompt(self) -> str:
        return (
            "You are a Senior Business Analyst with over 15 years of experience "
            "in enterprise and BFSI systems. You write high-quality, clear, and "
            "testable Business Requirement Documents (BRDs).\n\n"
            "Rules:\n"
            "- Use clear 'shall' statements\n"
            "- Do NOT assume missing information\n"
            "- Clearly flag ambiguities as Open Questions\n"
            "- Follow standard BRD best practices\n"
        )

    def generate_brd_sections(
        self,
        requirements: str,
        context: str
    ) -> Dict[str, str]:
        """
        Generate BRD content organized by sections.
        """
        print(f"DEBUG BRD SERVICE: Generating sections for requirements: {requirements[:100]}...")
        print(f"DEBUG BRD SERVICE: Context length: {len(context)} characters")

        user_prompt = f"""
Generate content for each BRD section below. For each section, provide clear, concise content based on the requirements and context.

Requirements:
{requirements}

Context from Business Documents, MOMs, Technical Docs, and Transcripts:
{context}

Format your response as:
SECTION_NAME: Content for this section

Sections to generate:
1. introduction
2. business_objectives
3. in_scope
4. out_of_scope
5. business_requirements
6. functional_requirements
7. non_functional_requirements
8. assumptions_constraints
9. risks_dependencies
10. acceptance_criteria
11. open_questions

Ensure each section has relevant, specific content. Use 'shall' statements where appropriate.
"""

        print(f"DEBUG BRD SERVICE: Calling GPT-5 API with prompt length: {len(user_prompt)}")

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            max_output_tokens=self.max_output_tokens,
            timeout=self.timeout
        )

        raw_text = self._extract_text_from_response(response)
        print(f"DEBUG BRD SERVICE: GPT-5 response extracted, length: {len(raw_text)}")
        print(f"DEBUG BRD SERVICE: Response preview: {raw_text[:500]}...")

        if not raw_text:
            print("ERROR: GPT-5 returned empty output - aborting BRD generation")
            raise RuntimeError("GPT returned empty output — aborting BRD generation")

        # Parse the response into sections
        sections = self._parse_sections_from_response(raw_text)
        print(f"DEBUG BRD SERVICE: Parsed sections: {list(sections.keys())}")

        # Debug: Show content preview for each section
        for section_name, content in sections.items():
            content_preview = content[:100] + "..." if len(content) > 100 else content
            print(f"DEBUG BRD SERVICE: {section_name}: {content_preview}")

        return sections

    def _parse_sections_from_response(self, response_text: str) -> Dict[str, str]:
        """Parse the AI response into structured sections."""
        sections = {}
        current_section = None
        current_content = []

        lines = response_text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a section header (handle both numbered and unnumbered formats)
            section_mappings = {
                "introduction": ["introduction:", "1. introduction:"],
                "business_objectives": ["business_objectives:", "2. business_objectives:"],
                "in_scope": ["in_scope:", "3. in_scope:", "in-scope:"],
                "out_of_scope": ["out_of_scope:", "4. out_of_scope:", "out-of-scope:"],
                "business_requirements": ["business_requirements:", "5. business_requirements:"],
                "functional_requirements": ["functional_requirements:", "6. functional_requirements:"],
                "non_functional_requirements": ["non_functional_requirements:", "7. non_functional_requirements:"],
                "assumptions_constraints": ["assumptions_constraints:", "8. assumptions_constraints:", "assumptions & constraints:"],
                "risks_dependencies": ["risks_dependencies:", "9. risks_dependencies:", "risks & dependencies:"],
                "acceptance_criteria": ["acceptance_criteria:", "10. acceptance_criteria:"],
                "open_questions": ["open_questions:", "11. open_questions:"]
            }

            found_section = None
            for section_key, patterns in section_mappings.items():
                for pattern in patterns:
                    if line.lower().startswith(pattern):
                        found_section = section_key
                        section_pattern = pattern
                        break
                if found_section:
                    break

            if found_section:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()

                # Start new section
                current_section = found_section
                current_content = []
                # Add the rest of the line if it has content
                content_part = line[len(section_pattern):].strip()
                if content_part:
                    current_content.append(content_part)
            else:
                # This is content for current section
                if current_section:
                    current_content.append(line)

        # Save the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()

        # Fill in any missing sections with defaults
        default_sections = {
            "introduction": "This BRD outlines the requirements for the specified enhancement.",
            "business_objectives": "To enhance the system capabilities as specified in requirements.",
            "in_scope": "Features and functionality specified in requirements.",
            "out_of_scope": "Features not mentioned in the requirements.",
            "business_requirements": "Business requirements derived from the provided context.",
            "functional_requirements": "Functional requirements to be implemented.",
            "non_functional_requirements": "Performance, security, and other quality requirements.",
            "assumptions_constraints": "Project assumptions and technical constraints.",
            "risks_dependencies": "Potential risks and system dependencies.",
            "acceptance_criteria": "Criteria for project acceptance and completion.",
            "open_questions": "Questions requiring clarification from stakeholders."
        }

        for section, default in default_sections.items():
            if section not in sections or not sections[section]:
                sections[section] = default

        return sections

    def generate_brd(
        self,
        requirements: str,
        context: str,
        brd_template: Optional[str] = None
    ) -> str:
        """
        Generate a full BRD using requirements + retrieved context.
        Returns formatted BRD text.
        """
        sections = self.generate_brd_sections(requirements, context)

        if brd_template:
            return brd_template.format(**sections)
        else:
            # Use default template
            template = """
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
            return template.format(**sections)

    def generate_brd_section(
        self,
        section_name: str,
        requirements: str,
        context: str
    ) -> str:
        """
        Generate a specific BRD section (useful for section-wise generation).
        """

        prompt = f"""
Generate ONLY the following BRD section: {section_name}

Requirements:
{requirements}

Context:
{context}

Ensure:
- Clear and testable language
- No assumptions
- Professional BA tone
"""

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            max_output_tokens=1000,
            timeout=self.timeout
        )

        raw_text = self._extract_text_from_response(response)
        return raw_text

    def fill_word_template(self, sections: Dict[str, str], template_path: str = "BRD_Template.docx", output_path: str = "filled_brd.docx") -> str:
        """
        Fill a Word template with generated BRD sections.
        Returns the path to the filled document.
        """
        # Load the template
        doc = Document(template_path)

        # Replace placeholders in all paragraphs
        for paragraph in doc.paragraphs:
            for run in paragraph.runs:
                for placeholder, content in sections.items():
                    placeholder_text = f"{{{placeholder}}}"
                    if placeholder_text in run.text:
                        run.text = run.text.replace(placeholder_text, content)

        # Save the filled document
        doc.save(output_path)
        return output_path

    def generate_brd_word(
        self,
        requirements: str,
        context: str,
        template_path: str = "BRD_Template.docx",
        output_path: str = "filled_brd.docx"
    ) -> str:
        """
        Generate BRD and fill Word template.
        Returns the path to the filled Word document.
        """
        print(f"DEBUG BRD SERVICE: Starting Word BRD generation")

        # Generate sections
        sections = self.generate_brd_sections(requirements, context)
        print(f"DEBUG BRD SERVICE: Generated sections for Word template: {list(sections.keys())}")

        # Fill template
        filled_path = self.fill_word_template(sections, template_path, output_path)
        print(f"DEBUG BRD SERVICE: Word template filled and saved to: {filled_path}")

        return filled_path

    def quality_check(
        self,
        brd_content: str,
        brd_template: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Perform quality analysis on an existing BRD.
        """

        prompt = f"""
Review the following Business Requirement Document for quality.

BRD Content:
{brd_content}
"""

        if brd_template:
            prompt += f"""

Expected BRD Template:
{brd_template}
"""

        prompt += """

Identify:
- Missing sections
- Ambiguous requirements
- Conflicting statements
- Overall quality score (0–100)
Provide improvement suggestions.
"""

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            max_output_tokens=1200,
            timeout=self.timeout
        )

        analysis_text = self._extract_text_from_response(response)
        return {
            "analysis": analysis_text
        }
