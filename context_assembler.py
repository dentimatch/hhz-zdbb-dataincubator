import os
import json

class VibeContextAssembler:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.context = {}

    def extract_readme_summary(self):
        readme_path = os.path.join(self.project_root, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # Example: extract first 20 lines as summary context
            summary = "".join(lines[:20]).strip()
            self.context['readme_summary'] = summary
        else:
            self.context['readme_summary'] = "No README found."

    def extract_requirements(self):
        req_path = os.path.join(self.project_root, "requirements.txt")
        if os.path.exists(req_path):
            with open(req_path, "r", encoding="utf-8") as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            self.context['requirements'] = requirements
        else:
            self.context['requirements'] = []

    def add_custom_context(self, key, value):
        self.context[key] = value

    def get_context_payload(self):
        return json.dumps(self.context, indent=2)

# Example usage
if __name__ == "__main__":
    assembler = VibeContextAssembler(project_root=".")
    assembler.extract_readme_summary()
    assembler.extract_requirements()
    # Add any user or session-specific info required:
    assembler.add_custom_context("user_level", "non-technical")
    # Get JSON context for passing to AI agents
    context_payload = assembler.get_context_payload()
    print(context_payload)
