import yaml
import json
import os
import sys

# Wir verwenden hier ein Mockup für den Skill Loader, da das Google GenAI SDK keine nativen "Skills" in diesem Format direkt parst
class GemmaSkillLoader:
    def __init__(self, skill_dir):
        self.skill_dir = skill_dir
        self.instructions = ""
        self.metadata = {}
        self.schemas = []
        self.load_skill()
        
    def load_skill(self):
        # 1. Parse YAML frontmatter and Markdown body from SKILL.md
        skill_md_path = os.path.join(self.skill_dir, "SKILL.md")
        with open(skill_md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        parts = content.split("---")
        if len(parts) >= 3:
            self.metadata = yaml.safe_load(parts[1])
            self.instructions = parts[2].strip()
            
        # 2. Load JSON schemas from the tools/ directory
        tools_dir = os.path.join(self.skill_dir, "tools")
        for filename in os.listdir(tools_dir):
            if filename.endswith(".json"):
                with open(os.path.join(tools_dir, filename), 'r', encoding='utf-8') as f:
                    self.schemas.append(json.load(f))
                    
        print(f"Loaded Skill: {self.metadata.get('name')} - {self.metadata.get('description')}")
        print(f"Loaded {len(self.schemas)} tool schemas.")

if __name__ == "__main__":
    print("Loading Gemma-Skill...")
    skill_dir = os.path.join(os.path.dirname(__file__), "..", "skills", "mandelbrot_explorer")
    loader = GemmaSkillLoader(skill_dir)
    print("\nSkill Bootstrap successful. Instructions:")
    print(loader.instructions[:100] + "...")
