import os
import json
import requests
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from jira import JIRA, JIRAError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class JiraConfig:
    """Jira configuration data class."""
    server: str
    email: str
    api_token: str
    default_project: str = ""
    issue_type_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        if self.issue_type_mapping is None:
            self.issue_type_mapping = {
                "user_story": "Story",
                "epic": "Epic",
                "task": "Task"
            }

class JiraService:
    """Service for interacting with Jira API."""
    
    def __init__(self, config: JiraConfig):
        self.config = config
        self.jira_client = None
        # Don't connect automatically in constructor to avoid async issues
        # Connection will be established when needed
    
    def _connect(self) -> bool:
        """Establish connection to Jira."""
        try:
            # For Jira Cloud, use email + API token
            # For Jira Server/Data Center, use username + API token
            self.jira_client = JIRA(
                server=self.config.server,
                basic_auth=(self.config.email, self.config.api_token),
                timeout=30  # Add timeout to prevent hanging
            )
            # Test connection
            server_info = self.jira_client.server_info()
            logger.info(f"Successfully connected to Jira at {self.config.server}")
            logger.info(f"Server version: {server_info.get('version', 'Unknown')}")
            return True
        except JIRAError as e:
            logger.error(f"Failed to connect to Jira: {e}")
            self.jira_client = None  # Reset client on failure
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Jira: {e}")
            self.jira_client = None  # Reset client on failure
            return False
    
    def _ensure_connected(self) -> bool:
        """Ensure Jira client is connected."""
        if not self.jira_client:
            return self._connect()
        try:
            # Test existing connection
            server_info = self.jira_client.server_info()
            return True
        except Exception as e:
            logger.warning(f"Connection test failed: {e}. Attempting to reconnect...")
            # If connection test fails, try to reconnect
            self.jira_client = None  # Reset client before reconnecting
            return self._connect()

    def test_connection(self) -> Dict[str, Any]:
        """Test Jira connection and return connection status."""
        try:
            connected = self._ensure_connected()
            
            if connected:
                # Get user info
                user = self.jira_client.current_user()
                # Get server info
                server_info = self.jira_client.server_info()
                
                return {
                    "success": True,
                    "message": "Connection successful",
                    "user": user,
                    "server_version": server_info.get('version', 'Unknown'),
                    "connected_at": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to connect to Jira"
                }
        except JIRAError as e:
            return {
                "success": False,
                "message": f"Jira error: {str(e)}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Connection error: {str(e)}"
            }
    
    def get_projects(self) -> List[Dict[str, Any]]:
        """Get list of available projects."""
        try:
            if not self._ensure_connected():
                return []
            projects = self.jira_client.projects()
            return [
                {
                    "key": project.key,
                    "name": project.name,
                    "id": project.id,
                    "description": getattr(project, 'description', '')
                }
                for project in projects
            ]
        except JIRAError as e:
            logger.error(f"Failed to get projects: {e}")
            return []
    
    def get_issue_types(self) -> List[Dict[str, Any]]:
        """Get list of available issue types."""
        try:
            if not self._ensure_connected():
                return []
            issue_types = self.jira_client.issue_types()
            return [
                {
                    "id": issue_type.id,
                    "name": issue_type.name,
                    "description": getattr(issue_type, 'description', ''),
                    "subtask": issue_type.subtask
                }
                for issue_type in issue_types
            ]
        except JIRAError as e:
            logger.error(f"Failed to get issue types: {e}")
            return []
    
    def get_fields(self) -> List[Dict[str, Any]]:
        """Get list of available custom fields."""
        try:
            if not self._ensure_connected():
                return []
            fields = self.jira_client.fields()
            return [
                {
                    "id": field['id'],
                    "name": field['name'],
                    "schema": field.get('schema', {}),
                    "required": field.get('required', False)
                }
                for field in fields
            ]
        except JIRAError as e:
            logger.error(f"Failed to get fields: {e}")
            return []
    
    def create_epic(self, project_key: str, title: str, description: str, 
                   priority: str = "Medium") -> Optional[Dict[str, Any]]:
        """Create an Epic in Jira."""
        try:
            if not self._ensure_connected():
                return None
            epic_issue_type = self._get_issue_type_id("Epic", project_key)
            if not epic_issue_type:
                logger.error("Epic issue type not found")
                return None
            
            issue_dict = {
                'project': {'key': project_key},
                'summary': title,
                'description': description,
                'issuetype': {'name': 'Epic'},
                'priority': {'name': priority}
            }
            
            new_issue = self.jira_client.create_issue(fields=issue_dict)
            logger.info(f"Created Epic: {new_issue.key}")
            
            return {
                "issue_key": new_issue.key,
                "issue_id": new_issue.id,
                "url": f"{self.config.server}/browse/{new_issue.key}",
                "type": "Epic"
            }
        except JIRAError as e:
            logger.error(f"Failed to create Epic: {e}")
            return None
    
    def create_user_story(self, project_key: str, title: str, description: str,
                         acceptance_criteria: str, priority: str = "Medium",
                         story_points: Optional[int] = None,
                         epic_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create a User Story in Jira."""
        try:
            if not self._ensure_connected():
                return None
            story_issue_type = self._get_issue_type_id("Story", project_key)
            if not story_issue_type:
                logger.error("Story issue type not found")
                return None
            
            # Combine description and acceptance criteria
            full_description = f"{description}\n\nh3. Acceptance Criteria\n{acceptance_criteria}"
            
            issue_dict = {
                'project': {'key': project_key},
                'summary': title,
                'description': full_description,
                'issuetype': {'name': 'Story'},
                'priority': {'name': priority}
            }
            
            # Add story points if provided
            if story_points is not None:
                issue_dict['customfield_10016'] = story_points  # Story Points field
            
            new_issue = self.jira_client.create_issue(fields=issue_dict)
            logger.info(f"Created User Story: {new_issue.key}")
            
            # Link to Epic if provided
            if epic_key:
                self._link_to_epic(new_issue.key, epic_key)
            
            return {
                "issue_key": new_issue.key,
                "issue_id": new_issue.id,
                "url": f"{self.config.server}/browse/{new_issue.key}",
                "type": "Story"
            }
        except JIRAError as e:
            logger.error(f"Failed to create User Story: {e}")
            return None
    
    def create_task(self, project_key: str, title: str, description: str,
                   priority: str = "Medium", parent_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create a Task in Jira."""
        try:
            if not self._ensure_connected():
                return None
            task_issue_type = self._get_issue_type_id("Task", project_key)
            if not task_issue_type:
                logger.error("Task issue type not found")
                return None
            
            issue_dict = {
                'project': {'key': project_key},
                'summary': title,
                'description': description,
                'issuetype': {'name': 'Task'},
                'priority': {'name': priority}
            }
            
            new_issue = self.jira_client.create_issue(fields=issue_dict)
            logger.info(f"Created Task: {new_issue.key}")
            
            # Link to parent if provided
            if parent_key:
                self._link_parent_child(parent_key, new_issue.key)
            
            return {
                "issue_key": new_issue.key,
                "issue_id": new_issue.id,
                "url": f"{self.config.server}/browse/{new_issue.key}",
                "type": "Task"
            }
        except JIRAError as e:
            logger.error(f"Failed to create Task: {e}")
            return None
    
    def _link_to_epic(self, story_key: str, epic_key: str):
        """Link a story to an epic."""
        try:
            # Set the Epic Link custom field
            epic_link_field = self._get_epic_link_field_id()
            if epic_link_field:
                self.jira_client.issue(story_key).update(fields={epic_link_field: epic_key})
                logger.info(f"Linked {story_key} to Epic {epic_key}")
        except JIRAError as e:
            logger.error(f"Failed to link {story_key} to Epic {epic_key}: {e}")
    
    def _link_parent_child(self, parent_key: str, child_key: str):
        """Link a child issue to a parent issue."""
        try:
            self.jira_client.create_issue_link(
                type="Subtask",
                inwardIssue=parent_key,
                outwardIssue=child_key
            )
            logger.info(f"Linked {child_key} as subtask of {parent_key}")
        except JIRAError as e:
            logger.error(f"Failed to link {child_key} to {parent_key}: {e}")
    
    def _link_issues(self, issue1_key: str, issue2_key: str, link_type: str = "Relates"):
        """Link two issues together."""
        try:
            self.jira_client.create_issue_link(
                type=link_type,
                inwardIssue=issue1_key,
                outwardIssue=issue2_key
            )
            logger.info(f"Linked {issue1_key} and {issue2_key}")
        except JIRAError as e:
            logger.error(f"Failed to link {issue1_key} and {issue2_key}: {e}")
    
    def _get_issue_type_id(self, issue_type_name: str, project_key: str) -> Optional[str]:
        """Get issue type ID for a given project."""
        try:
            if not self._ensure_connected():
                return None
            issue_types = self.jira_client.issue_types()
            for issue_type in issue_types:
                if issue_type.name.lower() == issue_type_name.lower():
                    return issue_type.id
            return None
        except JIRAError:
            return None
    
    def _get_epic_link_field_id(self) -> Optional[str]:
        """Get the Epic Link custom field ID."""
        try:
            if not self._ensure_connected():
                return None
            fields = self.jira_client.fields()
            for field in fields:
                if field['name'] == 'Epic Link':
                    return field['id']
            return None
        except JIRAError:
            return None
    
    def bulk_create_issues(self, project_key: str, issues_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk create issues from User Stories data."""
        try:
            if not self._ensure_connected():
                return {
                    "success": [],
                    "failed": [{"error": "Failed to connect to Jira"}],
                    "total": len(issues_data)
                }
        except:
            return {
                "success": [],
                "failed": [{"error": "Failed to connect to Jira"}],
                "total": len(issues_data)
            }
        
        results = {
            "success": [],
            "failed": [],
            "total": len(issues_data)
        }
        
        # First pass: create epics
        epic_mapping = {}
        for issue_data in issues_data:
            if issue_data.get("type") == "epic":
                epic = self.create_epic(
                    project_key=project_key,
                    title=issue_data["title"],
                    description=issue_data.get("description", ""),
                    priority=issue_data.get("priority", "Medium")
                )
                if epic:
                    results["success"].append(epic)
                    epic_mapping[issue_data["epic_id"]] = epic["issue_key"]
                else:
                    results["failed"].append({
                        "type": "epic",
                        "title": issue_data["title"],
                        "error": "Failed to create epic"
                    })
        
        # Second pass: create stories and tasks
        story_mapping = {}
        for issue_data in issues_data:
            if issue_data.get("type") == "user_story":
                # Find associated epic
                epic_key = None
                if "epic_id" in issue_data and issue_data["epic_id"] in epic_mapping:
                    epic_key = epic_mapping[issue_data["epic_id"]]
                
                story = self.create_user_story(
                    project_key=project_key,
                    title=issue_data["title"],
                    description=issue_data.get("description", ""),
                    acceptance_criteria=issue_data.get("acceptance_criteria", ""),
                    priority=issue_data.get("priority", "Medium"),
                    story_points=issue_data.get("effort_estimate"),
                    epic_key=epic_key
                )
                if story:
                    results["success"].append(story)
                    story_mapping[issue_data["story_id"]] = story["issue_key"]
                else:
                    results["failed"].append({
                        "type": "user_story",
                        "title": issue_data["title"],
                        "error": "Failed to create user story"
                    })
        
        # Third pass: create dependencies
        for issue_data in issues_data:
            if issue_data.get("type") == "dependency":
                story_key = story_mapping.get(issue_data["story_id"])
                depends_on_key = story_mapping.get(issue_data["depends_on"])
                
                if story_key and depends_on_key:
                    self._link_issues(story_key, depends_on_key, "Blocks")
        
        return results

def get_jira_config() -> Optional[JiraConfig]:
    """Get Jira configuration from environment variables."""
    server = os.getenv("JIRA_SERVER")
    email = os.getenv("JIRA_EMAIL")
    api_token = os.getenv("JIRA_API_TOKEN")
    default_project = os.getenv("JIRA_DEFAULT_PROJECT", "")
    
    if not all([server, email, api_token]):
        logger.warning("Jira configuration incomplete. Set JIRA_SERVER, JIRA_EMAIL, and JIRA_API_TOKEN environment variables.")
        return None
    
    return JiraConfig(
        server=server,
        email=email,
        api_token=api_token,
        default_project=default_project
    )