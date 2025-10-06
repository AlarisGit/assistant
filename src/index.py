import config
import logging
import os
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

source_dir = os.path.join(config.DATA_DIR, 'docs')

_data = dict()

def _process_path(path: str=source_dir):
    global _data
    if os.path.isdir(path):
        logger.debug(f"Processing dir: {path}")
        for root, dirs, files in os.walk(path):
            for dir in dirs:
                _process_path(os.path.join(root, dir))
            for file in files:
                _process_path(os.path.join(root, file))
    elif os.path.isfile(path) and path.endswith('.json'):
        logger.debug(f"Processing file: {path}")
        with open(path, 'r') as f:
            doc = json.load(f)
            doc_ref = _data
            doc_id = doc.get('id', '')
            doc_source = doc.get('source', '')
            doc_summary = doc.get('summary', '')
            doc_description = doc.get('description', '')
            doc_keypoints = doc.get('keypoints', [])
            doc_keywords = doc.get('keywords', [])
            doc_tree_items = []
            for crumb in ['', *doc.get('crumbs', [])]:
                doc_tree_items.append(crumb)
                if crumb not in doc_ref:
                    doc_ref[crumb] = {}
                doc_ref = doc_ref[crumb]
            
            if doc_id in doc_ref:
                logger.warning(f"Duplicate doc_id: {doc_id} {doc_source} / {doc_ref['source']}")
            doc_ref['id'] = doc_id
            doc_ref['source'] = doc_source
            doc_ref['summary'] = doc_summary
            doc_ref['description'] = doc_description
            doc_ref['keypoints'] = doc_keypoints
            doc_ref['keywords'] = doc_keywords
    

if __name__ == '__main__':
    print(f"LOG_LEVEL: {config.LOG_LEVEL}")
    _process_path()
    print(f"{json.dumps(_data, indent=2, ensure_ascii=False, sort_keys=True)}")