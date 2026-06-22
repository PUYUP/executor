import re

def clean_academic_text(text: str) -> str:
    """
    Cleans academic text extracted from PDFs.
    - Fixes hyphenated words broken across lines
    - Replaces single newlines with spaces (to unwrap paragraphs)
    - Replaces multiple whitespace characters with a single space
    """
    if not text:
        return ""
        
    # 1. Fix hyphenated words broken across lines (e.g., "exam-\nple" -> "example")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # 2. Replace single newlines with a space to merge broken lines in the same paragraph
    # (assuming double newlines separate actual paragraphs)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # 3. Collapse multiple spaces and tabs into a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 4. Strip leading/trailing whitespace
    return text.strip()
