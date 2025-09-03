import config
import logging
from typing import List, Dict, Any
import os
import json
import hashlib
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

source_dir = os.path.join(config.DATA_DIR, "md")
docs_dir = os.path.join(config.DATA_DIR, "docs")

os.makedirs(docs_dir, exist_ok=True)

def process_markdown_content(content: str) -> Dict[str, Any]:
    """
    Process markdown content and extract metadata including source and crumbs.
    
    Args:
        content: Raw markdown content
        
    Returns:
        Dictionary containing processed content and extracted metadata
    """
    result_dict = {"content": content}
    
    # Extract Source, Crumbs, and Description from markdown content
    # Look for patterns like:
    # Source: https://docs-ai.alarislabs.com/HTML-SMS/agreements-defaults-tab.html
    # Crumbs: Reference books > Contract companies > Agreements defaults tab
    # Description: Administration Account manager history: the permission must be enabled...
    
    source_pattern = r'Source:\s*(.+?)(?:\n|$)'
    crumbs_pattern = r'Crumbs:\s*(.+?)(?:\n|$)'
    description_pattern = r'Description:\s*(.+?)(?:\n|$)'
    
    # Search for source URL
    source_match = re.search(source_pattern, content, re.IGNORECASE | re.MULTILINE)
    if source_match:
        result_dict['source'] = source_match.group(1).strip()
        logger.debug(f"Extracted source: {result_dict['source']}")
    
    # Search for breadcrumbs
    crumbs_match = re.search(crumbs_pattern, content, re.IGNORECASE | re.MULTILINE)
    if crumbs_match:
        result_dict['crumbs'] = crumbs_match.group(1).strip()
        logger.debug(f"Extracted crumbs: {result_dict['crumbs']}")
    
    # Search for description
    description_match = re.search(description_pattern, content, re.IGNORECASE | re.MULTILINE)
    if description_match:
        result_dict['description'] = description_match.group(1).strip()
        logger.debug(f"Extracted description: {result_dict['description']}")
    
    # Clean content by removing the extracted metadata lines
    cleaned_content = content
    if source_match:
        cleaned_content = re.sub(source_pattern, '', cleaned_content, flags=re.IGNORECASE | re.MULTILINE)
    if crumbs_match:
        cleaned_content = re.sub(crumbs_pattern, '', cleaned_content, flags=re.IGNORECASE | re.MULTILINE)
    if description_match:
        cleaned_content = re.sub(description_pattern, '', cleaned_content, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove extra whitespace and empty lines at the beginning
    cleaned_content = re.sub(r'^\s*\n+', '', cleaned_content.strip())
    result_dict['content'] = cleaned_content
    
    return result_dict

def process_file(path: str):
    global docs_dir
    logger.info(f"Processing file {path}")
    doc_dict = {}
    if path.endswith(".md"):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Process markdown content to extract metadata
            processed_result = process_markdown_content(content)
            
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            doc_id = os.path.basename(path).replace(".md", "")+'-'+content_hash[0:8]
            logger.info(f"Doc ID: {doc_id}")
            
            # Build document dictionary with extracted metadata
            doc_dict["id"] = doc_id
            doc_dict["source_path"] = path
            doc_dict["content"] = processed_result["content"]
            
            # Add extracted metadata if available
            if "source" in processed_result:
                doc_dict["source"] = processed_result["source"]
            if "crumbs" in processed_result:
                doc_dict["crumbs"] = processed_result["crumbs"]
            if "description" in processed_result:
                doc_dict["description"] = processed_result["description"]
            
            # Save processed document
            output_path = os.path.join(docs_dir, doc_id + ".json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved processed document to {output_path}")


    else:
        logger.info(f"File {path} is not a Markdown file")

def parse_dir(path: str) -> List[str]:
    logger.info(f"Parsing directory {path}")
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)):
            parse_dir(os.path.join(path, f))
        elif os.path.isfile(os.path.join(path, f)):
            process_file(os.path.join(path, f))

if __name__ == '__main__':
    parse_dir(source_dir)