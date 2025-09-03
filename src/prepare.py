import config
import logging
from typing import List
import os
import json
import hashlib

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

source_dir = os.path.join(config.DATA_DIR, "md")
docs_dir = os.path.join(config.DATA_DIR, "docs")

os.makedirs(docs_dir, exist_ok=True)

def process_markdown_content(content: str) -> str:
    result_dict = {"content": content}
    return result_dict

def process_file(path: str):
    global docs_dir
    logger.info(f"Processing file {path}")
    doc_dict = {}
    if path.endswith(".md"):
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            doc_id = os.path.basename(path).replace(".md", "")+'-'+content_hash[0:8]
            logger.info(f"Doc ID: {doc_id}")
            doc_dict["id"] = doc_id
            doc_dict["source_path"] = path
            json.dump(doc_dict, open(os.path.join(docs_dir, doc_id + ".json"), 'w', encoding='utf-8'), indent=4)


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