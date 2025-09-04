import config
import logging
from typing import List, Dict, Any, Tuple
import os
import json
import hashlib
import logging
import re
import tiktoken
from llm import get_image_description, get_summarization
from config import MIN_CHUNK_TOKENS, MAX_CHUNK_TOKENS, OVERLAP_CHUNK_TOKENS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

source_dir = os.path.join(config.DATA_DIR, "md")
docs_dir = os.path.join(config.DATA_DIR, "docs")

os.makedirs(docs_dir, exist_ok=True)

# Initialize tokenizer for token counting
try:
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
except Exception:
    # Fallback to simple word-based estimation
    encoding = None
    
def count_tokens(text: str) -> int:
    if encoding:
        return len(encoding.encode(text))
    else:
        # Rough estimation: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)
 
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

    result_dict['content_tokens'] = count_tokens(augmented_content)
    result_dict['original_content_tokens'] = count_tokens(cleaned_content)
    
    # Generate summary for the augmented content
    try:
        logger.info("Generating summary for augmented content")
        summary_result = get_summarization(augmented_content)
        
        # Handle both dict and JSON string return formats
        if isinstance(summary_result, dict):
            result_dict.update(summary_result)
            logger.debug(f"Merged summary keys: {list(summary_result.keys())}")
        elif isinstance(summary_result, str):
            # Try to parse as JSON first
            try:
                import json as json_module
                # Remove markdown code block formatting if present
                clean_result = summary_result.strip()
                if clean_result.startswith('```json'):
                    clean_result = clean_result[7:]  # Remove ```json
                if clean_result.endswith('```'):
                    clean_result = clean_result[:-3]  # Remove ```
                clean_result = clean_result.strip()
                
                parsed_json = json_module.loads(clean_result)
                if isinstance(parsed_json, dict):
                    result_dict.update(parsed_json)
                    logger.debug(f"Parsed and merged JSON summary keys: {list(parsed_json.keys())}")
                else:
                    # If parsed JSON is not a dict, store as 'summary' key
                    result_dict['summary'] = parsed_json
                    logger.debug("Stored parsed JSON summary in 'summary' key")
            except (json_module.JSONDecodeError, ValueError):
                # If JSON parsing fails, use the string as-is
                result_dict['summary'] = summary_result
                logger.debug("Stored string summary in 'summary' key (JSON parsing failed)")
        else:
            logger.warning(f"Unexpected summary result format: {type(summary_result)}")
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        # Continue without summary if it fails
    
    # Generate smart chunks from the processed content
    chunks = smart_chunk_content(augmented_content, result_dict)
    result_dict['chunks'] = chunks
    logger.info(f"Generated {len(chunks)} chunks for content of {result_dict['content_tokens']}/{result_dict['original_content_tokens']} tokens")
    
    return result_dict

def smart_chunk_content(content: str, doc_metadata: Dict[str, Any]) -> List[str]:
    """
    Split content into smart chunks following sentence and paragraph boundaries.
    
    Args:
        content: The processed content with image descriptions
        doc_metadata: Document metadata to include in each chunk
        
    Returns:
        List of content strings (chunks)
    """
   
    # Split content into sentences while preserving image descriptions
    sentences = _split_into_sentences_with_images(content)
    
    # Group sentences into paragraphs
    paragraphs = _group_sentences_into_paragraphs(sentences)
    
    # Create chunks from paragraphs
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = sum(count_tokens(sentence) for sentence in paragraph)
        
        # If current chunk + this paragraph exceeds max tokens, finalize current chunk
        if current_chunk_sentences and (current_chunk_tokens + paragraph_tokens) > MAX_CHUNK_TOKENS:
            # Create chunk from current sentences
            chunk_content = ' '.join(current_chunk_sentences).strip()
            if chunk_content and count_tokens(chunk_content) >= MIN_CHUNK_TOKENS:
                chunks.append(chunk_content)
            
            # Start new chunk with overlap
            overlap_sentences = _get_overlap_sentences(current_chunk_sentences, count_tokens)
            current_chunk_sentences = overlap_sentences
            current_chunk_tokens = sum(count_tokens(sentence) for sentence in overlap_sentences)
        
        # Add paragraph to current chunk
        current_chunk_sentences.extend(paragraph)
        current_chunk_tokens += paragraph_tokens
        
        # If paragraph alone exceeds max tokens, split it
        if paragraph_tokens > MAX_CHUNK_TOKENS:
            current_chunk_sentences = current_chunk_sentences[:-len(paragraph)]  # Remove the paragraph we just added
            current_chunk_tokens -= paragraph_tokens
            
            # Split large paragraph into smaller chunks
            paragraph_chunks = _split_large_paragraph(paragraph, count_tokens)
            chunks.extend(paragraph_chunks)
            
            # Reset current chunk
            current_chunk_sentences = []
            current_chunk_tokens = 0
    
    # Add final chunk if it has content
    if current_chunk_sentences:
        chunk_content = ' '.join(current_chunk_sentences).strip()
        if chunk_content and count_tokens(chunk_content) >= MIN_CHUNK_TOKENS:
            chunks.append(chunk_content)
    
    return chunks

def _split_into_sentences_with_images(content: str) -> List[str]:
    """
    Split content into sentences while keeping image descriptions intact.
    """
    sentences = []
    
    # Find all image description blocks (marked with special formatting from LLM)
    image_pattern = r'\*Image Description: [^*]+\*'
    
    # Split content by image descriptions to handle them separately
    parts = re.split(f'({image_pattern})', content)
    
    for part in parts:
        if re.match(image_pattern, part):
            # This is an image description - keep it as a single "sentence"
            sentences.append(part.strip())
        else:
            # Regular text - split into sentences
            if part.strip():
                # Enhanced sentence splitting pattern that handles:
                # - Standard sentence endings (.!?)
                # - Abbreviations (e.g., "Mr.", "etc.")
                # - Numbers and decimals
                # - Multiple spaces/newlines between sentences
                text_sentences = re.split(
                    r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+(?=[A-Z])|(?<=\.)\s+(?=[A-Z][a-z])',
                    part.strip()
                )
                
                # Clean and filter sentences
                for sentence in text_sentences:
                    cleaned = sentence.strip()
                    if cleaned and len(cleaned) > 3:  # Avoid very short fragments
                        # Ensure sentence ends properly
                        if not cleaned.endswith(('.', '!', '?', '*', ':')):
                            # Look for natural sentence ending or add period
                            if cleaned.endswith(('etc', 'vs', 'e.g', 'i.e')):
                                cleaned += '.'
                        sentences.append(cleaned)
    
    return [s for s in sentences if s]

def _group_sentences_into_paragraphs(sentences: List[str]) -> List[List[str]]:
    """
    Group sentences into paragraphs based on content structure.
    """
    paragraphs = []
    current_paragraph = []
    
    for sentence in sentences:
        # Image descriptions always start a new paragraph
        if sentence.startswith('[Image Description:'):
            if current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = []
            paragraphs.append([sentence])  # Image description is its own paragraph
        else:
            current_paragraph.append(sentence)
    
    # Add final paragraph
    if current_paragraph:
        paragraphs.append(current_paragraph)
    
    return paragraphs

def _get_overlap_sentences(sentences: List[str], count_tokens) -> List[str]:
    """
    Get sentences for overlap, ensuring we don't exceed overlap token limit.
    """
    overlap_sentences = []
    overlap_tokens = 0
    
    # Take sentences from the end, working backwards
    for sentence in reversed(sentences):
        sentence_tokens = count_tokens(sentence)
        if overlap_tokens + sentence_tokens <= OVERLAP_CHUNK_TOKENS:
            overlap_sentences.insert(0, sentence)
            overlap_tokens += sentence_tokens
        else:
            break
    
    return overlap_sentences

def _split_large_paragraph(paragraph: List[str], count_tokens) -> List[str]:
    """
    Split a large paragraph into multiple chunks while preserving sentence boundaries.
    """
    chunks = []
    current_sentences = []
    current_tokens = 0
    
    for sentence in paragraph:
        sentence_tokens = count_tokens(sentence)
        
        # If adding this sentence would exceed max tokens, create a chunk
        if current_sentences and (current_tokens + sentence_tokens) > MAX_CHUNK_TOKENS:
            chunk_content = ' '.join(current_sentences).strip()
            chunks.append(chunk_content)
            
            # Start new chunk with overlap
            overlap_sentences = _get_overlap_sentences(current_sentences, count_tokens)
            current_sentences = overlap_sentences
            current_tokens = sum(count_tokens(s) for s in overlap_sentences)
        
        current_sentences.append(sentence)
        current_tokens += sentence_tokens
    
    # Add final chunk
    if current_sentences:
        chunk_content = ' '.join(current_sentences).strip()
        chunks.append(chunk_content)
    
    return chunks


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
            doc_dict.update(processed_result)
            
            # Save processed document with chunks included
            output_path = os.path.join(docs_dir, doc_id + ".json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, indent=4, ensure_ascii=False)
            logger.info(f"Saved processed document to {output_path}")
            logger.info(f"Document contains {len(doc_dict.get('chunks', []))} chunks")


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