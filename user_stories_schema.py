USER_STORIES_SCHEMA_JSON_STRING = """{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "user_stories": {
      "type": "array",
      "description": "Generate user stories based on the BRD content for the specified version",
      "items": {
        "type": "object",
        "properties": {
          "story_id": { 
            "type": "string",
            "description": "Unique identifier for the user story (e.g., US-001)"
          },
          "title": { 
            "type": "string",
            "description": "Brief, descriptive title of the user story"
          },
          "user_role": { 
            "type": "string",
            "description": "The role of the user who will benefit from this story"
          },
          "description": { 
            "type": "string",
            "description": "Detailed description of what the user wants to accomplish"
          },
          "acceptance_criteria": { 
            "type": "string",
            "description": "Specific, testable conditions that must be met for the story to be accepted"
          },
          "priority": { 
            "type": "string",
            "description": "Priority level (e.g., High, Medium, Low)"
          },
          "effort_estimate": { 
            "type": "string",
            "description": "Estimated effort required (e.g., Story Points, T-Shirt Size)"
          },
          "brd_reference": { 
            "type": "string",
            "description": "Reference to the specific BRD requirement this story is derived from"
          },
          "version": { 
            "type": "string",
            "description": "Version of the BRD this user story is associated with"
          }
        },
        "required": ["story_id", "title", "user_role", "description", "acceptance_criteria", "priority", "effort_estimate", "brd_reference", "version"]
      }
    },
    "epics": {
      "type": "array",
      "description": "Group related user stories into epics for better organization",
      "items": {
        "type": "object",
        "properties": {
          "epic_id": { 
            "type": "string",
            "description": "Unique identifier for the epic (e.g., EP-001)"
          },
          "title": { 
            "type": "string",
            "description": "Title of the epic"
          },
          "description": { 
            "type": "string",
            "description": "Brief description of the epic"
          },
          "related_stories": { 
            "type": "array",
            "description": "List of story IDs that belong to this epic",
            "items": {
              "type": "string"
            }
          }
        },
        "required": ["epic_id", "title", "description", "related_stories"]
      }
    },
    "dependencies": {
      "type": "array",
      "description": "List of dependencies between user stories",
      "items": {
        "type": "object",
        "properties": {
          "story_id": { 
            "type": "string",
            "description": "The story that has a dependency"
          },
          "depends_on": { 
            "type": "string",
            "description": "The story that must be completed first"
          },
          "dependency_type": { 
            "type": "string",
            "description": "Type of dependency (e.g., Technical, Business, Data)"
          }
        },
        "required": ["story_id", "depends_on", "dependency_type"]
      }
    }
  },
  "required": ["user_stories", "epics", "dependencies"]
}"""