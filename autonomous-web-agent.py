import os
from multion.client import MultiOn

from utils import (
    get_multi_on_api_key,
    visualizeSession,
    MultiOnDemo,
    SessionManager,
    ImageUtils,
    display_step_header,
)

multion_api_key = get_multi_on_api_key()
multion = MultiOn(api_key=multion_api_key, base_url=os.getenv("DLAI_MULTION_BASE_URL"))


class MultiOnClient:
    """A simplified client for the MultiOn API"""

    def __init__(self, multion):
        """Initialize the MultiOn client with the API key"""
        # Use the actual MultiOn class from the imported module
        self.client = multion
        self.session_id = None
        self.current_url = None
        self.screenshot = None

    def create_session(self, url):
        """Create a new agent session"""
        session = self.client.sessions.create(url=url, include_screenshot=True)
        self.session_id = session.session_id
        self.current_url = session.url
        self.screenshot = session.screenshot
        return session

    def close_session(self):
        """Close the current session"""
        if self.session_id:
            self.client.sessions.close(self.session_id)
            self.session_id = None

    def list_sessions(self):
        """List all active sessions"""
        return self.client.sessions.list()

    def close_all_sessions(self):
        """Close all open sessions"""
        sessions = self.list_sessions()
        for session in sessions.session_ids:
            self.client.sessions.close(session)

    def navigate_to_url(self, url):
        """Navigate to a URL in the current session"""
        if not self.session_id:
            return self.create_session(url)

        response = self.client.sessions.step(
            session_id=self.session_id,
            cmd=f"GO TO URL {url}",
            include_screenshot=True,
            mode="standard",
        )

        self.current_url = response.url
        self.screenshot = response.screenshot
        return response

    def execute_task(self, task):
        """Execute a task in the current agent session"""
        if not self.session_id:
            raise ValueError("No active session. Call create_session first.")

        response = self.client.sessions.step(
            session_id=self.session_id,
            cmd=(
                f"""IMPORTANT: DO NOT ASK THE USER ANY QUESTIONS. 
                All the necessary information is already provided 
                and is on the current Page.\n
                Complete the task to the best of your abilities.\n\n
                Task:\n\n{task}"""
            ),
            include_screenshot=True,
        )

        self.current_url = response.url
        self.screenshot = response.screenshot

        return response


multionClient = MultiOnClient(multion)


def example1():
    instruction = "get list all the courses"
    url = "https://deeplearning.ai/courses"
    MAX_STEPS = 10
    MAX_IMAGE_WIDTH = 500
    session = multionClient.create_session(url)
    step = 0
    while session.status == "CONTINUE" and step < MAX_STEPS:
        display_step_header(step)
        visualizeSession(session, max_image_width=MAX_IMAGE_WIDTH)
        session = multionClient.execute_task(instruction)
        step += 1
    visualizeSession(session, max_image_width=MAX_IMAGE_WIDTH)


def mutltiOnBrowserUI():
    url = "https://deeplearning.ai/courses"
    sessionManager = SessionManager(url, multionClient)
    subject = "RAG"
    name = "Div Garg"
    email = "info@theagi.company"
    action_engine = None
    instructions = [
        f"Find the course on {subject} and open it",
        f"Summarize the course",
        f"Detailed course lessons",
        f"""Go to the deeplearning ai home page and subscribe 
                  to the batch newsletter use the name {name}, 
                  {email} choose the other required fields
                  as you see fit. Make sure to select the proper dropdown
                  values. Finally once you see 
                  the subscribe button click it""",
    ]
    demo = MultiOnDemo(url, sessionManager, multionClient, instructions, action_engine)
    demo.create_demo()
