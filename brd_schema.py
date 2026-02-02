BRD_SCHEMA_JSON_STRING = """{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "document": {
      "type": "object",
      "properties": {
        "title_main": { "type": "string" },
        "id": { "type": "string" },
        "program": { "type": "string" },
        "type": { "type": "string" },
        "overview": { 
          "type": "string",
          "description": "Provide a high-level summary of the project goals and scope."
        },
        "current_constraint": { 
          "type": "string",
          "description": "Describe existing limitations or constraints that impact the project."
        },
        "objective": { 
          "type": "string",
          "description": "State the primary objectives and expected outcomes of the project."
        },
        "in_scope": { 
          "type": "string",
          "description": "List all features, functions, and deliverables that are included in this project."
        },
        "out_of_scope": { 
          "type": "string",
          "description": "List all features, functions, and deliverables that are explicitly excluded from this project."
        }
      },
      "required": ["title_main", "id", "program", "type", "overview", "current_constraint", "objective", "in_scope", "out_of_scope"]
    },

    "business_requirements": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "req_id_bs": { "type": "string" },
          "title_bs": { "type": "string" },
          "description_bs": { 
            "type": "string",
            "description": "Describe the business requirement in detail, focusing on what needs to be achieved."
          },
          "as_is_behaviour": { 
            "type": "string",
            "description": "Document the current state or existing process before the change."
          },
          "to_be_behaviour": { 
            "type": "string",
            "description": "Document the desired future state or process after implementation."
          },
          "pre_requisite": { 
            "type": "string",
            "description": "List any conditions or requirements that must be met before this requirement can be implemented."
          },
          "acceptance_criteria": { 
            "type": "string",
            "description": "Define the measurable criteria that will be used to determine when the requirement has been successfully fulfilled."
          },
          "alternate_flows": { 
            "type": "string",
            "description": "Describe alternative scenarios or exception handling for this requirement."
          },
          "reference_documents": { 
            "type": "string",
            "description": "List name of the documents, standards, or specifications that are referenced for this requirement."
          }
        },
        "required": ["req_id_bs", "title_bs", "description_bs", "as_is_behaviour", "to_be_behaviour", "pre_requisite", "acceptance_criteria", "alternate_flows", "reference_documents"]
      }
    },

    "traceability_matrix": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "req_id_tm": { "type": "string" },
          "description_tm": { 
            "type": "string",
            "description": "Provide a very brief, single-sentence description of the requirement's purpose. Keep it crisp."
          },
          "source_channel": { 
            "type": "string",
            "description": "Identify the origin (e.g., Mobile App, Regulatory Mandate)."
          },
          "impacted_system": { 
            "type": "string",
            "description": "List affected systems (e.g., Core Banking, CRM)."
          },
          "outcome": { 
            "type": "string",
            "description": "Provide a very brief, single-sentence expected result. Keep it crisp."
          }
        },
        "required": ["req_id_tm", "description_tm", "source_channel", "impacted_system", "outcome"]
      }
    },

"non_functional_requirements": {
      "type": "array",
      "description": "Performance and capacity metrics",
      "items": {
        "type": "object",
        "properties": {
          "no_of_users": { "type": "string" },
          "peak_volume": { "type": "string" },
          "monthly_volume": { "type": "string" },
          "availability": { "type": "string" }
        },
        "required": ["no_of_users", "peak_volume", "monthly_volume", "availability"]
      },
      "minItems": 1,
      "maxItems": 1
    },
    "impact_on_operational_process": { 
      "type": "string",
      "description": "Describe how this project will affect existing operational processes, workflows, and procedures."
    },
    "regulatory_impact": { 
      "type": "string",
      "description": "Identify any regulatory requirements, compliance issues, or legal considerations that impact this project."
    },
    "reports_requirement": { 
      "type": "string",
      "description": "Specify all reporting requirements, including frequency, format, and distribution of reports."
    },
    "access_requirement": { 
      "type": "string",
      "description": "Define user access levels, permissions, authentication, and authorization requirements."
    },
    "security_requirement": { 
      "type": "string",
      "description": "Document all security requirements including data protection, encryption, and access controls."
    },
    "data_requirement": { 
      "type": "string",
      "description": "Specify data requirements including data sources, data quality, data retention, and data governance."
    },
    "training_requirement": { 
      "type": "string",
      "description": "Identify training needs for users, administrators, and support staff."
    },
    "open_questions": { 
      "type": "string",
      "description": "List any unresolved questions or areas that require further clarification."
    },
    "contradictions_found_in_input_documents": { 
      "type": "string",
      "description": "Identify and list any conflicting information between the provided source documents. You MUST explicitly mention the original source document name where the contradiction was identified(not context 1, 2 etc)."
    }
  },
  "required": [
    "document",
    "business_requirements",
    "traceability_matrix",
    "non_functional_requirements",
    "impact_on_operational_process",
    "regulatory_impact",
    "reports_requirement",
    "access_requirement",
    "security_requirement",
    "data_requirement",
    "training_requirement",
    "open_questions",
    "contradictions_found_in_input_documents"
  ]
}"""
