import json
from langchain_core.documents import Document

def load_cv_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    # Handle work experience separately to keep entries intact
    if "work_experience" in data:
        for experience in data["work_experience"]:
            content = format_work_experience(experience)
            metadata = {
                "is_most_recent": experience.get("is_most_recent", False),
                "type": "work_experience",
                "company": experience.get("company", ""),
                "start_date": experience.get("start_date", "")
            }
            docs.append(Document(page_content=content, metadata=metadata))

    # Handle education separately
    if "education" in data:
        for edu in data["education"]:
            content = format_education(edu)
            metadata = {"type": "education"}
            docs.append(Document(page_content=content, metadata=metadata))

    # Handle other sections with flattening
    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}{k}: ")
        elif isinstance(obj, list):
            for item in obj:
                walk(item, prefix)
        else:
            docs.append(
                Document(page_content=f"{prefix}{obj}")
            )

    # Skip work_experience and education as they're already processed
    for key, value in data.items():
        if key not in ["work_experience", "education"]:
            walk(value, f"{key}: ")

    return docs


def format_work_experience(exp: dict) -> str:
    """Format work experience entry as readable text."""
    lines = [
        f"Company: {exp.get('company', 'N/A')}",
        f"Role: {exp.get('role', 'N/A')}",
        f"Location: {exp.get('location', 'N/A')}",
        f"Employment Type: {exp.get('employment_type', 'N/A')}",
        f"Duration: {exp.get('start_date', 'N/A')} to {exp.get('end_date', 'N/A')}",
    ]
    
    if "responsibilities" in exp:
        lines.append("Responsibilities:")
        for resp in exp["responsibilities"]:
            lines.append(f"- {resp}")
    
    if "achievements" in exp:
        lines.append("Achievements:")
        for ach in exp["achievements"]:
            lines.append(f"- {ach}")
    
    if "skills_used" in exp:
        lines.append(f"Skills Used: {', '.join(exp['skills_used'])}")
    
    return "\n".join(lines)


def format_education(edu: dict) -> str:
    """Format education entry as readable text."""
    lines = [
        f"Degree: {edu.get('degree', 'N/A')}",
        f"Field: {edu.get('field_of_study', 'N/A')}",
        f"Institution: {edu.get('institution', 'N/A')}",
        f"Location: {edu.get('city', 'N/A')}, {edu.get('country', 'N/A')}",
        f"Duration: {edu.get('start_date', 'N/A')} to {edu.get('end_date', 'N/A')}",
    ]
    
    if "gpa" in edu:
        lines.append(f"GPA: {edu['gpa']}")
    
    if "eqf_level" in edu:
        lines.append(f"EQF Level: {edu['eqf_level']}")
    
    return "\n".join(lines)
