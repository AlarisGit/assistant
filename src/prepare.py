import config
import logging
from typing import List, Dict, Any
import os
import json
import hashlib
import re
from llm import get_image_description

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
    
    # Augment images with LLM-generated descriptions
    augmented_content = _augment_images_with_descriptions(cleaned_content)
    result_dict['content'] = augmented_content
    result_dict['original_content'] = cleaned_content
    
    return result_dict

def _augment_images_with_descriptions(content: str) -> str:
    """
    Find all image links in markdown content and augment them with LLM-generated descriptions.
    
    Args:
        content: Cleaned markdown content
        
    Returns:
        Markdown content with image descriptions added
    """
    # Pattern to match markdown image syntax: ![alt text](image_url)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def replace_image_with_description(match):
        alt_text = match.group(1)
        image_url = match.group(2)
        original_image = match.group(0)
        
        try:
            logger.info(f"Generating description for image: {image_url}")
            # Call LLM to get image description
            description_result = get_image_description(image_url)
            
            # Extract description from the result
            if isinstance(description_result, dict) and 'description' in description_result:
                image_description = description_result['description']
            elif isinstance(description_result, str):
                # Try to parse as JSON first
                try:
                    import json as json_module
                    # Remove markdown code block formatting if present
                    clean_result = description_result.strip()
                    if clean_result.startswith('```json'):
                        clean_result = clean_result[7:]  # Remove ```json
                    if clean_result.endswith('```'):
                        clean_result = clean_result[:-3]  # Remove ```
                    clean_result = clean_result.strip()
                    
                    parsed_json = json_module.loads(clean_result)
                    if isinstance(parsed_json, dict) and 'description' in parsed_json:
                        image_description = parsed_json['description']
                    else:
                        # Use the original string if no description field found
                        image_description = description_result
                except (json_module.JSONDecodeError, ValueError):
                    # If JSON parsing fails, use the string as-is
                    image_description = description_result
            else:
                logger.warning(f"Unexpected result format from get_image_description: {type(description_result)}")
                return original_image
            
            # Add description right after the image
            augmented_image = f"{original_image}\n\n*Image Description: {image_description}*"
            logger.debug(f"Augmented image with description: {image_url}")
            return augmented_image
            
        except Exception as e:
            logger.error(f"Failed to generate description for image {image_url}: {e}")
            # Return original image if description generation fails
            return original_image
    
    # Replace all images with augmented versions
    augmented_content = re.sub(image_pattern, replace_image_with_description, content)
    
    return augmented_content

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