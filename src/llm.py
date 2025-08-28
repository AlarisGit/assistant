import config
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _parse_model(model_provider: str) -> tuple[str, str]:
    model = ''
    provider = 'ollama'
    if '@' in model_provider:
        model, provider = model_provider.lower().split('@')
    else:
        model = model_provider.lower()
        if 'gpt-oss' in model or 'gemma' in model:
            provider = 'ollama'
        elif 'gpt' in model:
            provider = 'openai'
        elif 'gemini' in model:
            provider = 'google'
    if provider not in ['ollama', 'openai', 'google']:
        raise ValueError(f"Unknown provider: {provider}")
    return model, provider

def _get_prompt(action:str, model: str, provider: str) -> str:
    if os.path.exists(config.PROMPTS_DIR):
        for name in [
            f'{action}_{provider}_{model}', 
            f'{action}_{provider}', 
            f'{action}_{model}', 
            f'{action}', 
            f'{model}'
        ]:
            prompt_path = os.path.join(config.PROMPTS_DIR, f'prompt_{name}.md')
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    return f.read()
    return ''

def get_embedding(text: str, model_provider: str = config.EMB_MODEL) -> list[float]:
    model, provider = _parse_model(model_provider)
    return []

def get_response(prompt: str, image: str, model_provider: str = config.RSP_MODEL) -> str:
    model, provider = _parse_model(model_provider)
    return ''

def get_summarization(text: str, model_provider: str = config.SUM_MODEL) -> str:
    model, provider = _parse_model(model_provider)
    prompt = _get_prompt('sum', model, provider)
    return get_response(prompt, '', model_provider)

def get_image_description(image: str, model_provider: str = config.RSP_MODEL) -> str:
    model, provider = _parse_model(model_provider)
    prompt = _get_prompt('vsn', model, provider)
    return get_response(prompt, image, model_provider)



if __name__ == '__main__':
    test_text_models = [
        'gpt-5-nano@openai',
        'gemini-2.5-flash@google',
        'gemma3:27b@ollama',
        'gpt-oss:20b@ollama'
    ]
    test_image_models = [
        'llava:7b@ollama',
        'gemma3:27b@ollama',
        'gemini-2.5-flash@google'
    ]
    test_embedding_models = [
        'text-embedding-3-large@openai',
        'mxbai-embed-large@ollama',
        'gemini-embedding-001@google'
    ]
    test_text = """
    As it flew an idea formed itself in the Procurator's mind, which was
now bright and clear. It was thus : the hegemon had examined the case of the
vagrant philosopher Yeshua, surnamed Ha-Notsri, and could not substantiate
the criminal charge made against him. In particular he could not find the
slightest connection between Yeshua's actions and the recent disorders in
Jerusalem. The vagrant philosopher was mentally ill, as a result of which
the sentence of death pronounced on Ha-Notsri by the Lesser Sanhedrin would
not be confirmed. But in view of the danger of unrest liable to be caused by
Yeshua's mad, Utopian preaching, the Procurator would remove the man from
Jerusalem and sentence him to imprisonment in Caesarea Stratonova on the
Mediterranean--the place of the Procurator's own residence. It only remained
to dictate this to the secretary. 
    """
    test_images = [
        'https://docs-ai.alarislabs.com/HTML-SMS/hmfile_hash_d43cc7e4.png',
        '~/Downloads/clip0419.png'
    ]
    for model in test_text_models:
        logger.debug(f"Testing model: {model}")
        response = get_response(test_text, model=model)
        logger.debug(f"Response: {response}")
    for model in test_image_models:
        for image in test_images:
            logger.debug(f"Testing model {model} with image: {image}")
            response = get_response(test_text, image=image, model=model)
            logger.debug(f"Response: {response}")
    for model in test_embedding_models:
        logger.debug(f"Testing model: {model}")
        response = get_embedding(test_text, model=model)
        logger.debug(f"Response: {response[0:10]}")

    logger.debug("Hello")
    config.print_config()