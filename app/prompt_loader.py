from pathlib import Path
from typing import Optional

import yaml


class PromptLoader:
    def __init__(self, prompts_dir: Optional[Path] = None):
        self._dir = prompts_dir or Path(__file__).parent / "prompts"

    def load(self, name: str, language: Optional[str] = None) -> dict:
        if language:
            filename = f"{name}_{language}_v1.yaml"
        else:
            filename = f"{name}_v1.yaml"

        path = self._dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")

        with open(path) as f:
            return yaml.safe_load(f)

    def render(self, template: str, variables: dict) -> str:
        rendered = template
        for key, value in variables.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        # Unescape doubled braces used to protect literal braces from substitution
        rendered = rendered.replace("{{", "{").replace("}}", "}")
        return rendered
