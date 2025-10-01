import config
import logging
import os
import base64
import requests
import mimetypes
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from openai import OpenAI
import google.generativeai as genai
import sys
import io
import time
from PIL import Image as PILImage
import json
import hashlib
import string
from ratelimit_metrics import RateLimitedMetricsProvider

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_cache_path(key: dict) -> Tuple[str, str]:
    if key and isinstance(key, dict) and os.path.isdir(config.CACHE_DIR):
        key_str = json.dumps(key, sort_keys=True, ensure_ascii=False, indent=4)
        key_hash = hashlib.md5(key_str.encode('utf-8')).hexdigest()
        cache_subdir = os.path.join(config.CACHE_DIR, 'llm', key_hash[0:2], key_hash[2:4])

        return cache_subdir, key_hash
    else:
        return None, None
   
def load_cached_value(key: dict) -> Optional[dict]:
    value_dict = None
    if key and isinstance(key, dict):
        cache_subdir, key_hash = get_cache_path(key)
        logger.debug(f"Looking for cached value for key {key_hash}")
        if cache_subdir and os.path.isdir(cache_subdir):
            cache_file = os.path.join(cache_subdir, key_hash + '.json')
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    value_str = f.read()
                    try:
                        value_dict = json.loads(value_str)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to load cached value for key {key_hash}")
                        value_dict = None
    return value_dict


def save_cached_value(key: dict, value: dict):
    if key and isinstance(key, dict) and value and isinstance(value, dict):
        cache_subdir, key_hash = get_cache_path(key)
        if cache_subdir:
            if not os.path.exists(cache_subdir):
                os.makedirs(cache_subdir)
            cache_file = os.path.join(cache_subdir, key_hash + '.json')
            logger.debug(f"Saving cached value for key {key_hash}")
            with open(cache_file, 'w') as f:
                f.write(json.dumps({**key, **value}, sort_keys=True, ensure_ascii=False, indent=4))

class BaseProvider(ABC):
    """Base class for LLM providers with unified interface"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def generate_text(self, prompt: str, system_prompt: str = '', history: List[Tuple[str, str]] = [], 
                     image: str = '', **kwargs) -> str:
        """Generate text response (supports both text and image inputs)"""
        pass
    
    @abstractmethod
    def generate_embedding(self, text: str, model: str, **kwargs) -> List[float]:
        """Generate text embedding"""
        pass
    
    @abstractmethod
    async def generate_text_async(self, prompt: str, system_prompt: str = '', history: List[Tuple[str, str]] = [], 
                                 image: str = '', **kwargs) -> str:
        """Generate text response asynchronously (supports both text and image inputs)"""
        pass
    
    @abstractmethod
    async def generate_embedding_async(self, text: str, model: str, **kwargs) -> List[float]:
        """Generate text embedding asynchronously"""
        pass


class OpenAIProvider(BaseProvider):
    """OpenAI provider using latest responses API with reasoning"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Configure proxy for OpenAI client
        import httpx
        proxy_settings = config.get_proxy_settings()
        if proxy_settings is not None:
            # Custom proxy configuration
            if proxy_settings.get('http') is None:
                # Proxy disabled - use default httpx client without proxy
                http_client = httpx.Client()
            else:
                # Custom proxy specified
                http_client = httpx.Client(proxies=proxy_settings)
            self.client = OpenAI(api_key=api_key, http_client=http_client)
        else:
            # PROXY=SYSTEM - use default client (respects environment variables)
            self.client = OpenAI(api_key=api_key)
    
    def _is_reasoning_model(self, model: str) -> bool:
        """Check if model is a reasoning model"""
        reasoning_models = ['o1', 'o3', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano']
        return any(rm in model.lower() for rm in reasoning_models)
    
    def _is_gpt5_model(self, model: str) -> bool:
        """Check if model is GPT-5"""
        return 'gpt-5' in model.lower()
    
    def generate_text(self, prompt: str, system_prompt: str = '', history: List[Tuple[str, str]] = [], 
                     image: str = '', model: str = 'gpt-5-nano', reasoning_effort: str = 'medium', 
                     max_output_tokens: Optional[int] = None, use_responses_api: bool = None, **kwargs) -> str:
        """Generate text using OpenAI API with proper reasoning support"""
        
        is_reasoning = self._is_reasoning_model(model)
        is_gpt5 = self._is_gpt5_model(model)
        
        # Auto-detect responses API usage - disable for vision tasks for now
        if use_responses_api is None:
            use_responses_api = is_reasoning and not image
        
        # Validate reasoning_effort for GPT-5
        if is_gpt5 and reasoning_effort not in ['minimal', 'low', 'medium', 'high']:
            reasoning_effort = 'medium'
        elif not is_gpt5 and reasoning_effort == 'minimal':
            reasoning_effort = 'medium'  # Fallback for non-GPT-5 models
        
        try:
            if use_responses_api:
                return self._generate_with_responses_api(
                    prompt, system_prompt, history, image, model, 
                    reasoning_effort, max_output_tokens, **kwargs
                )
            else:
                return self._generate_with_chat_completions(
                    prompt, system_prompt, history, image, model, 
                    max_output_tokens, **kwargs
                )
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    def _generate_with_responses_api(self, prompt: str, system_prompt: str, 
                                   history: List[Tuple[str, str]], image: str, model: str,
                                   reasoning_effort: str, max_output_tokens: Optional[int], **kwargs) -> str:
        """Generate using responses API for reasoning models"""
        input_messages = self._build_input_messages(prompt, system_prompt, history, image)
        
        params = {
            "model": model,
            "input": input_messages,
            "reasoning": {
                "effort": reasoning_effort
            }
        }
        
        # Note: verbosity parameter not yet supported in OpenAI API
        # Will be added when available
        
        if max_output_tokens:
            params["max_output_tokens"] = max_output_tokens
        
        # Add tools if specified
        if 'tools' in kwargs:
            params["tools"] = kwargs['tools']
        if 'tool_choice' in kwargs:
            params["tool_choice"] = kwargs['tool_choice']
        
        response = self.client.responses.create(**params)
        
        # Extract text from response output
        for output in response.output:
            if output.type == 'message' and hasattr(output, 'content'):
                for content in output.content:
                    if hasattr(content, 'text'):
                        return content.text
        
        return "No response content found"
    
    def _generate_with_chat_completions(self, prompt: str, system_prompt: str,
                                      history: List[Tuple[str, str]], image: str, model: str,
                                      max_output_tokens: Optional[int], **kwargs) -> str:
        """Generate using chat completions API for non-reasoning models"""
        messages = self._build_messages(prompt, system_prompt, history, image)
        
        params = {
            "model": model,
            "messages": messages
        }
        
        # Only add supported parameters for non-reasoning models
        if max_output_tokens:
            params["max_tokens"] = max_output_tokens
        
        # Add optional parameters that are supported
        supported_params = ['temperature', 'top_p', 'presence_penalty', 'frequency_penalty', 
                          'logprobs', 'top_logprobs', 'logit_bias', 'tools', 'tool_choice']
        for param in supported_params:
            if param in kwargs:
                params[param] = kwargs[param]
        
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    def generate_embedding(self, text: str, model: str = 'text-embedding-3-large', **kwargs) -> List[float]:
        """Generate embedding using OpenAI API"""
        try:
            response = self.client.embeddings.create(input=text, model=model)
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"OpenAI embedding error: {e}")
            raise
    
    async def generate_text_async(self, prompt: str, system_prompt: str = '', history: List[Tuple[str, str]] = [], 
                                 image: str = '', model: str = 'gpt-5-nano', reasoning_effort: str = 'medium', 
                                 max_output_tokens: Optional[int] = None, use_responses_api: bool = None, **kwargs) -> str:
        """Generate text asynchronously using OpenAI API with proper reasoning support"""
        
        # Run the synchronous method in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.generate_text(
                prompt, system_prompt, history, image, model, 
                reasoning_effort, max_output_tokens, use_responses_api, **kwargs
            )
        )
    
    async def generate_embedding_async(self, text: str, model: str = 'text-embedding-3-large', **kwargs) -> List[float]:
        """Generate embedding asynchronously using OpenAI API"""
        
        # Run the synchronous method in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.generate_embedding(text, model, **kwargs)
        )
    
    
    def _build_messages(self, prompt: str, system_prompt: str, history: List[Tuple[str, str]], image: str) -> List[Dict[str, Any]]:
        """Build OpenAI message format for chat completions"""
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # Build current user message with optional image
        content = [{"type": "text", "text": prompt}]
        
        if image:
            if image.startswith(('http://', 'https://')):
                content.append({"type": "image_url", "image_url": {"url": image}})
            else:
                # Expand tilde and resolve path
                image_path = os.path.expanduser(image)
                mime_type, _ = mimetypes.guess_type(image_path)
                mime_type = mime_type or 'image/jpeg'
                with open(image_path, 'rb') as f:
                    base64_image = base64.b64encode(f.read()).decode('utf-8')
                content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                })
        
        messages.append({"role": "user", "content": content})
        return messages
    
    def _build_input_messages(self, prompt: str, system_prompt: str, history: List[Tuple[str, str]], image: str) -> List[Dict[str, Any]]:
        """Build input messages for responses API"""
        messages = []
        
        # For reasoning models, use developer messages instead of system messages
        if system_prompt:
            messages.append({"role": "developer", "content": system_prompt})
        
        # Add conversation history
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # Build current user message with optional image
        if image:
            # For responses API with images, use simple text format for now
            messages.append({"role": "user", "content": f"{prompt}\n[Image provided but not supported in responses API yet]"})
        else:
            messages.append({"role": "user", "content": prompt})
        
        return messages


class GoogleProvider(BaseProvider):
    """Google Gemini provider with thinking budget configuration
    
    Note: Google's genai SDK does not support direct proxy configuration.
    Use PROXY=SYSTEM to respect environment variables, as the SDK will use them automatically.
    """
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Note: genai.configure() respects environment proxy variables (http_proxy, https_proxy)
        # For custom proxy, set PROXY=SYSTEM and configure via environment variables
        genai.configure(api_key=api_key)
    
    def generate_text(self, prompt: str, system_prompt: str = '', history: List[Tuple[str, str]] = [], 
                     image: str = '', model: str = 'gemini-2.5-flash-lite', 
                     thinking_budget: int = 20000, temperature: float = 1.0, 
                     max_output_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate text using Gemini API with thinking budget"""
        
        # Configure generation settings
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        
        # Add thinking budget for thinking models
        if 'thinking' in model:
            generation_config.thinking_budget = thinking_budget
        
        # Configure safety settings (permissive for development)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        try:
            gen_model = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                safety_settings=safety_settings,
                system_instruction=system_prompt if system_prompt else None
            )
            
            # Build conversation history
            chat_history = []
            for user_msg, assistant_msg in history:
                chat_history.extend([
                    {"role": "user", "parts": [user_msg]},
                    {"role": "model", "parts": [assistant_msg]}
                ])
            
            # Build content with optional image
            contents = []
            if image:
                if image.startswith(('http://', 'https://')):
                    # For URLs, use the image directly
                    contents.append(image)
                else:
                    # For local files, expand path and read with PIL Image
                    image_path = os.path.expanduser(image)
                    from PIL import Image as PILImage
                    img = PILImage.open(image_path)
                    contents.append(img)
            
            contents.append(prompt)
            
            # Start chat if we have history, otherwise generate directly
            if chat_history:
                chat = gen_model.start_chat(history=chat_history)
                response = chat.send_message(contents)
            else:
                response = gen_model.generate_content(contents)
            
            return response.text
        except Exception as e:
            self.logger.error(f"Google API error: {e}")
            raise
    
    def generate_embedding(self, text: str, model: str = 'text-embedding-004', **kwargs) -> List[float]:
        """Generate embedding using Google API"""
        try:
            result = genai.embed_content(model=f"models/{model}", content=text)
            return result['embedding']
        except Exception as e:
            self.logger.error(f"Google embedding error: {e}")
            raise
    
    async def generate_text_async(self, prompt: str, system_prompt: str = '', history: List[Tuple[str, str]] = [], 
                                 image: str = '', model: str = 'gemini-2.5-flash-lite', 
                                 thinking_budget: int = 20000, temperature: float = 1.0, 
                                 max_output_tokens: Optional[int] = None, **kwargs) -> str:
        """Generate text asynchronously using Gemini API with thinking budget"""
        
        # Run the synchronous method in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.generate_text(
                prompt, system_prompt, history, image, model, 
                thinking_budget, temperature, max_output_tokens, **kwargs
            )
        )
    
    async def generate_embedding_async(self, text: str, model: str = 'text-embedding-004', **kwargs) -> List[float]:
        """Generate embedding asynchronously using Google API"""
        
        # Run the synchronous method in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.generate_embedding(text, model, **kwargs)
        )
    


class OllamaProvider(BaseProvider):
    """Ollama provider for local models"""
    
    def __init__(self, api_key: str = '', base_url: str = 'http://localhost:11434'):
        super().__init__(api_key)
        self.base_url = base_url

    """
    1. create Modelfile:
    FROM gemma3:27b
    PARAMETER num_ctx 32768
    2. build model:
    ollama create gemma3:27b-32k -f Modelfile
    """
    
    def generate_text(self, prompt: str, system_prompt: str = '', history: List[Tuple[str, str]] = [], 
                     image: str = '', model: str = 'gpt-oss:20b', temperature: float = 0.7, 
                     max_tokens: Optional[int] = None, num_ctx: Optional[int] = None, 
                     request_timeout: Optional[Tuple[float, float]] = None,
                     max_image_dim: Optional[int] = None, jpeg_quality: int = 90,
                     **kwargs) -> str:
        """Generate text using Ollama API"""
        # Auto-tune image limits for known models if not provided
        if image and max_image_dim is None:
            mdl = (model or '').lower()
            if 'gemma3' in mdl:
                max_image_dim = 896
            elif 'llava' in mdl:
                max_image_dim = 336
            else:
                max_image_dim = 2048
        
        messages = self._build_messages(prompt, system_prompt, history, image,
                                        max_image_dim=max_image_dim, jpeg_quality=jpeg_quality)
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                # Ensure large context window when supported by the model
                "num_ctx": num_ctx or int(os.getenv("OLLAMA_NUM_CTX", 32768)),
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        # Timeouts: (connect, read). Defaults tuned for large-generation reads
        timeout = request_timeout or (10.0, 300.0)
        url = f"{self.base_url}/api/chat"
        
        # Simple retry for transient 5xx
        last_exc = None
        proxy_settings = config.get_proxy_settings()
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=timeout, proxies=proxy_settings)
                if resp.status_code >= 500:
                    # Capture body for diagnostics, but avoid logging large payloads
                    body_preview = resp.text[:1000]
                    self.logger.warning(
                        f"Ollama server {resp.status_code} on attempt {attempt+1}/3. Body preview: {body_preview}"
                    )
                    # Backoff then retry for transient errors
                    if resp.status_code in (502, 503, 504) and attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    resp.raise_for_status()
                resp.raise_for_status()
                data = resp.json()
                return data['message']['content']
            except Exception as e:
                last_exc = e
                # On network/timeout/transient errors, retry
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
        
        # Final failure: log details and raise
        self.logger.error(f"Ollama API error after retries: {last_exc}")
        raise last_exc
    
    def generate_embedding(self, text: str, model: str = 'mxbai-embed-large', **kwargs) -> List[float]:
        """Generate embedding using Ollama API"""
        payload = {"model": model, "input": text}
        proxy_settings = config.get_proxy_settings()
        
        try:
            resp = requests.post(f"{self.base_url}/api/embed", json=payload, proxies=proxy_settings)
            resp.raise_for_status()
            return resp.json()['embeddings'][0]
        except Exception as e:
            self.logger.error(f"Ollama embedding error: {e}")
            raise
    
    async def generate_text_async(self, prompt: str, system_prompt: str = '', history: List[Tuple[str, str]] = [], 
                                 image: str = '', model: str = 'gpt-oss:20b', temperature: float = 0.7, 
                                 max_tokens: Optional[int] = None, num_ctx: Optional[int] = None, 
                                 request_timeout: Optional[Tuple[float, float]] = None,
                                 max_image_dim: Optional[int] = None, jpeg_quality: int = 90,
                                 **kwargs) -> str:
        """Generate text asynchronously using Ollama API"""
        
        # Run the synchronous method in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.generate_text(
                prompt, system_prompt, history, image, model, temperature, 
                max_tokens, num_ctx, request_timeout, max_image_dim, jpeg_quality, **kwargs
            )
        )
    
    async def generate_embedding_async(self, text: str, model: str = 'mxbai-embed-large', **kwargs) -> List[float]:
        """Generate embedding asynchronously using Ollama API"""
        
        # Run the synchronous method in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.generate_embedding(text, model, **kwargs)
        )
    
    
    def _build_messages(self, prompt: str, system_prompt: str, history: List[Tuple[str, str]], image: str,
                        max_image_dim: Optional[int] = None, jpeg_quality: int = 95) -> List[Dict[str, Any]]:
        """Build Ollama message format"""
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        
        # Build current user message
        user_message = {"role": "user", "content": prompt}
        
        # Add image if provided
        if image:
            try:
                # Load image (URL or local)
                if image.startswith(('http://', 'https://')):
                    proxy_settings = config.get_proxy_settings()
                    resp = requests.get(image, timeout=(5.0, 20.0), proxies=proxy_settings)
                    resp.raise_for_status()
                    img_bytes = resp.content
                else:
                    image_path = os.path.expanduser(image)
                    with open(image_path, 'rb') as f:
                        img_bytes = f.read()

                # Process with PIL: resize and re-encode to JPEG
                try:
                    with PILImage.open(io.BytesIO(img_bytes)) as im:
                        # Convert to RGB (drop alpha)
                        if im.mode != 'RGB':
                            im = im.convert('RGB')
                        if max_image_dim and (im.width > max_image_dim or im.height > max_image_dim):
                            im.thumbnail((max_image_dim, max_image_dim), resample=PILImage.LANCZOS)
                        buf = io.BytesIO()
                        im.save(buf, format='JPEG', quality=jpeg_quality, optimize=True)
                        processed = buf.getvalue()
                except Exception:
                    # Fallback to original bytes if PIL fails
                    processed = img_bytes

                base64_image = base64.b64encode(processed).decode('utf-8')
                user_message["images"] = [base64_image]
            except Exception as e:
                # If image processing fails, still send text-only to avoid request failure
                logging.getLogger(f"{__name__}.{self.__class__.__name__}").warning(
                    f"Image preprocessing failed ({e}); sending text-only message.")
        
        messages.append(user_message)
        return messages


# Caching wrapper
class CachedProvider(BaseProvider):
    """Provider proxy that adds caching for text and embedding calls"""
    def __init__(self, inner: BaseProvider, provider_name: str):
        super().__init__(getattr(inner, 'api_key', ''))
        self.inner = inner
        self.provider_name = provider_name
        self.logger = logging.getLogger(f"{__name__}.CachedProvider[{provider_name}]")

    def _sanitize_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # Remove transport-only and non-deterministic args
        filtered = dict(kwargs)
        filtered.pop('request_timeout', None)
        filtered.pop('use_cache', None)
        filtered.pop('force_refresh', None)
        filtered.pop('action', None)
        return filtered

    def _image_fingerprint(self, image: str) -> Optional[str]:
        try:
            if not image:
                return None
            if image.startswith(('http://', 'https://')):
                # Avoid network fetch for key; use URL string
                return f"url:{image}"
            # Local file: hash contents
            image_path = os.path.expanduser(image)
            with open(image_path, 'rb') as f:
                data = f.read()
            return 'sha256:' + hashlib.sha256(data).hexdigest()
        except Exception:
            return None

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = '',
        history: List[Tuple[str, str]] = [],
        image: str = '',
        **kwargs,
    ) -> str:
        use_cache: bool = kwargs.pop('use_cache', True)
        force_refresh: bool = kwargs.pop('force_refresh', False)
        model = kwargs.get('model', '')
        sanitized_kwargs = self._sanitize_kwargs(kwargs)

        key = {
            'cache_schema': 1,
            'op': 'generate_text',
            'provider': self.provider_name,
            'model': model,
            'system_prompt': system_prompt,
            'prompt': prompt,
            'history': history,
            'image': image,
            'image_fp': self._image_fingerprint(image),
            'kwargs': sanitized_kwargs,
        }

        if use_cache and not force_refresh:
            cached = load_cached_value(key)
            if cached and isinstance(cached, dict) and 'response' in cached:
                return cached['response']

        # Miss: delegate to inner
        response = self.inner.generate_text(prompt, system_prompt, history, image, **kwargs)

        try:
            save_cached_value(key, {'response': response, 'ts': time.time()})
        except Exception as e:
            self.logger.warning(f"Failed to save cache (text): {e}")
        return response

    def generate_embedding(self, text: str, model: str, **kwargs) -> List[float]:
        use_cache: bool = kwargs.pop('use_cache', True)
        force_refresh: bool = kwargs.pop('force_refresh', False)
        sanitized_kwargs = self._sanitize_kwargs(kwargs)

        key = {
            'cache_schema': 1,
            'op': 'generate_embedding',
            'provider': self.provider_name,
            'model': model,
            'text': text,
            'kwargs': sanitized_kwargs,
        }

        if use_cache and not force_refresh:
            cached = load_cached_value(key)
            if cached and isinstance(cached, dict) and 'embedding' in cached:
                return cached['embedding']

        emb = self.inner.generate_embedding(text, model, **kwargs)
        try:
            save_cached_value(key, {'embedding': emb, 'ts': time.time()})
        except Exception as e:
            self.logger.warning(f"Failed to save cache (embedding): {e}")
        return emb
    
    async def generate_text_async(self, prompt: str, system_prompt: str = '', history: List[Tuple[str, str]] = [], 
                                 image: str = '', **kwargs) -> str:
        """Generate text asynchronously with caching"""
        use_cache: bool = kwargs.pop('use_cache', True)
        force_refresh: bool = kwargs.pop('force_refresh', False)
        model = kwargs.get('model', '')
        sanitized_kwargs = self._sanitize_kwargs(kwargs)

        key = {
            'cache_schema': 1,
            'op': 'generate_text',
            'provider': self.provider_name,
            'model': model,
            'system_prompt': system_prompt,
            'prompt': prompt,
            'history': history,
            'image': image,
            'image_fp': self._image_fingerprint(image),
            'kwargs': sanitized_kwargs,
        }

        if use_cache and not force_refresh:
            cached = load_cached_value(key)
            if cached and isinstance(cached, dict) and 'response' in cached:
                return cached['response']

        # Miss: delegate to inner async method
        response = await self.inner.generate_text_async(prompt, system_prompt, history, image, **kwargs)

        try:
            save_cached_value(key, {'response': response, 'ts': time.time()})
        except Exception as e:
            self.logger.warning(f"Failed to save cache (text): {e}")
        return response
    
    async def generate_embedding_async(self, text: str, model: str, **kwargs) -> List[float]:
        """Generate embedding asynchronously with caching"""
        use_cache: bool = kwargs.pop('use_cache', True)
        force_refresh: bool = kwargs.pop('force_refresh', False)
        sanitized_kwargs = self._sanitize_kwargs(kwargs)

        key = {
            'cache_schema': 1,
            'op': 'generate_embedding',
            'provider': self.provider_name,
            'model': model,
            'text': text,
            'kwargs': sanitized_kwargs,
        }

        if use_cache and not force_refresh:
            cached = load_cached_value(key)
            if cached and isinstance(cached, dict) and 'embedding' in cached:
                return cached['embedding']

        emb = await self.inner.generate_embedding_async(text, model, **kwargs)
        try:
            save_cached_value(key, {'embedding': emb, 'ts': time.time()})
        except Exception as e:
            self.logger.warning(f"Failed to save cache (embedding): {e}")
        return emb


# Provider factory and cache
_provider_cache = {}

def get_provider(provider: str, api_key: str = None, **kwargs) -> BaseProvider:
    """Factory function to get provider instances with caching"""
    cache_key = f"{provider}_{api_key or 'default'}"
    
    if cache_key not in _provider_cache:
        if provider == 'openai':
            inner = OpenAIProvider(api_key or config.OPENAI_API_KEY)
        elif provider == 'google':
            inner = GoogleProvider(api_key or config.GOOGLE_API_KEY)
        elif provider == 'ollama':
            base_url = kwargs.get('base_url', config.OLLAMA_URL)
            inner = OllamaProvider(api_key or '', base_url)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        # Wrap with rate limiting + metrics, then caching proxy
        rate_limited = RateLimitedMetricsProvider(inner, provider)
        _provider_cache[cache_key] = CachedProvider(rate_limited, provider)
    
    return _provider_cache[cache_key]

def _parse_model(model_provider: str) -> tuple[str, str]:
    """Parse model@provider format"""
    model_provider = model_provider.lower()
    if '@' in model_provider:
        # Split only on the last @ to handle cases like model:version@provider
        parts = model_provider.rsplit('@', 1)
        if len(parts) == 2:
            model, provider = parts
        else:
            # Fallback if rsplit doesn't work as expected
            model = model_provider
            provider = 'ollama'
    else:
        model = model_provider
        if 'gpt' in model or 'o1' in model or 'o3' in model:
            provider = 'openai'
        elif 'gemini' in model:
            provider = 'google'
        else:
            provider = 'ollama'
    
    if provider not in ['ollama', 'openai', 'google']:
        raise ValueError(f"Unknown provider: {provider}")
    return model, provider

class _SafeDict(dict):
    def __missing__(self, key):
        # Leave unknown placeholders intact
        return '{' + key + '}'

def _extract_placeholders(s: str) -> set:
    names = set()
    for literal_text, field_name, format_spec, conversion in string.Formatter().parse(s or ''):
        if field_name:
            # Handle nested or attribute style if ever used
            names.add(field_name.split('.')[0].split('[')[0])
    return names

def _process_includes(template: str, base_dir: str) -> str:
    """
    Process {inc:filename} includes in template.
    
    - {inc:persona.md} - loads from same directory as template
    - {inc:common/persona.md} - loads from relative path within prompts dir
    
    Args:
        template: Template string with potential includes
        base_dir: Directory of the template file for resolving relative paths
        
    Returns:
        Template with includes resolved
    """
    import re
    
    # Pattern to match {inc:path/to/file.md}
    include_pattern = r'\{inc:([^}]+)\}'
    
    def replace_include(match):
        include_path = match.group(1).strip()
        
        # Resolve the file path
        if os.path.isabs(include_path):
            # Absolute path
            file_path = include_path
        else:
            # Relative to base_dir (template directory)
            file_path = os.path.join(base_dir, include_path)
        
        # Read the included file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Recursively process includes in the included file
                return _process_includes(content, os.path.dirname(file_path))
        except FileNotFoundError:
            logger.warning(f"Include file not found: {file_path}")
            return f"[Include not found: {include_path}]"
        except Exception as e:
            logger.warning(f"Error loading include {include_path}: {e}")
            return f"[Include error: {include_path}]"
    
    return re.sub(include_pattern, replace_include, template)

def _process_macros(template: str, prompt_options: Optional[Dict[str, Any]]) -> str:
    """
    Process Python macro expressions in template.
    
    Supports:
    - {util.get_current_time()} - calls Python functions
    - {datetime.now().strftime('%Y-%m-%d')} - any Python expression
    - {variable_name} - regular variable substitution (falls back to prompt_options)
    
    Args:
        template: Template string with potential macros
        prompt_options: Dictionary of variables for substitution
        
    Returns:
        Template with macros evaluated
    """
    import re
    import util
    from datetime import datetime as dt_class
    
    # Pattern to match {expression} - but we'll be smart about it
    macro_pattern = r'\{([^}]+)\}'
    
    def replace_macro(match):
        expression = match.group(1).strip()
        
        # Check if it looks like a Python expression (has function call or method access)
        # Examples: util.get_current_time(), datetime.now(), config.VERSION
        is_expression = (
            '(' in expression or  # Function call
            ('.' in expression and not expression.split('.')[0] in (prompt_options or {}))  # Attribute/method access not in variables
        )
        
        if is_expression:
            # This looks like a Python expression - evaluate it
            try:
                # Create a safe evaluation context with limited builtins
                safe_globals = {
                    '__builtins__': {
                        # Allow safe builtins for common operations
                        'len': len,
                        'str': str,
                        'int': int,
                        'float': float,
                        'bool': bool,
                        'list': list,
                        'dict': dict,
                        'tuple': tuple,
                    },
                    'util': util,
                    'datetime': dt_class,
                    'config': config,
                }
                # Add prompt_options as variables
                safe_locals = dict(prompt_options or {})
                
                result = eval(expression, safe_globals, safe_locals)
                return str(result)
            except Exception as e:
                logger.warning(f"Error evaluating macro '{expression}': {e}")
                return f"{{{{Error: {expression}}}}}"
        else:
            # Simple variable - return placeholder for standard format_map
            return f"{{{expression}}}"
    
    return re.sub(macro_pattern, replace_macro, template)

def _format_prompt(template: str, prompt_options: Optional[Dict[str, Any]]) -> str:
    """
    Format prompt template with variable substitution.
    Uses _SafeDict to avoid KeyError on missing variables.
    
    Note: Macros and includes should be processed before calling this function.
    """
    try:
        return (template or '').format_map(_SafeDict(prompt_options or {}))
    except Exception:
        # Fallback to raw if formatting fails
        return template or ''

def _get_prompt(type: str, action: str, model: str, provider: str, 
                prompt_options: Optional[Dict[str, Any]] = None) -> str:
    """
    Get prompt template with hierarchical fallback and optional formatting.
    
    Supports:
    1. File includes: {inc:filename.md} or {inc:path/to/file.md}
    2. Python macros: {util.get_current_time()} or {datetime.now().strftime('%Y-%m-%d')}
    3. Variable substitution: {variable_name}
    
    Processing order: includes → macros → variables
    """
    if os.path.exists(config.PROMPTS_DIR):
        for name in [
            f'{type}_{action}_{provider}_{model}',
            f'{type}_{action}_{provider}',
            f'{type}_{action}_{model}',
            f'{type}_{action}',
            f'{type}_{model}',
            f'{action}_{provider}_{model}',
            f'{action}_{provider}',
            f'{action}_{model}',
            f'{action}'
        ]:
            prompt_path = os.path.join(config.PROMPTS_DIR, f'{name}.md')
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    raw = f.read()
                    
                    # Process template in order: includes → macros → variables
                    template = _process_includes(raw, os.path.dirname(prompt_path))
                    template = _process_macros(template, prompt_options)
                    return _format_prompt(template, prompt_options)
    return ''

def parse_response(response_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Parse LLM response to extract both plain text and JSON dictionary if present.
    
    This function handles cases where LLM responses contain JSON wrapped with extra text.
    It extracts the first valid JSON object found in the response and removes it from the text.
    
    Args:
        response_text: Raw LLM response text
        
    Returns:
        Tuple of (plain_text, json_dict):
        - plain_text: The response text with JSON removed (only extra explanatory text)
        - json_dict: Parsed JSON dictionary if found, None otherwise
    """
    if not response_text or not response_text.strip():
        return response_text, None
    
    try:
        # Look for JSON object in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            clean_json = response_text[json_start:json_end]
            response_dict = json.loads(clean_json)
            
            # Extract text before and after JSON, then combine
            text_before = response_text[:json_start].strip()
            text_after = response_text[json_end:].strip()
            
            # Combine non-JSON text parts
            remaining_text = ""
            if text_before:
                remaining_text += text_before
            if text_after:
                if remaining_text:
                    remaining_text += "\n\n" + text_after
                else:
                    remaining_text = text_after
            
            return remaining_text, response_dict
        else:
            # No JSON found
            return response_text, None
            
    except json.JSONDecodeError:
        # JSON parsing failed
        return response_text, None
    except Exception:
        # Any other error
        return response_text, None

def get_embedding(text: str, model_provider: str = config.EMB_MODEL, prompt_options: Dict[str, Any] = {}) -> List[float]:
    """Generate text embedding using specified model and provider"""
    model, provider_name = _parse_model(model_provider)
    prompt = _get_prompt('usr', 'emb', model, provider_name, prompt_options)
    sys_prompt = _get_prompt('sys', 'emb', model, provider_name, prompt_options)

    emb_text = f"{sys_prompt + '\n\n' if sys_prompt else ''}{prompt + '\n\n' if prompt else ''}{text}"

    provider = get_provider(provider_name)
    return provider.generate_embedding(emb_text, model)

def generate_text(action: str, text: str = '', history: List[Tuple[str, str]] = [], image: str = '',
                model_provider: str = config.GEN_MODEL, prompt_options: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    """Generate text response using specified model and provider"""
    model, provider_name = _parse_model(model_provider)
    
    usr_prompt = _get_prompt('usr', action, model, provider_name, prompt_options)
    sys_prompt = _get_prompt('sys', action, model, provider_name, prompt_options)

    prompt = f"{usr_prompt + '\n\n' if usr_prompt else ''}{text}"    

    provider = get_provider(provider_name)
    return provider.generate_text(prompt, sys_prompt, history, image, model=model, action=action, **kwargs)

def get_summarization(text: str, model_provider: str = config.SUM_MODEL, prompt_options: Dict[str, Any] = {}) -> str:
    """Generate text summarization"""
    return generate_text('sum', text, model_provider=model_provider, prompt_options=prompt_options)

def get_image_description(image: str, model_provider: str = config.VSN_MODEL, prompt_options: Dict[str, Any] = {}) -> str:
    """Generate image description using vision capabilities"""
    return generate_text('vsn', text='', image=image, model_provider=model_provider, prompt_options=prompt_options)

# Async wrapper functions for assistant.py
async def get_embedding_async(text: str, model_provider: str = config.EMB_MODEL, prompt_options: Dict[str, Any] = {}) -> List[float]:
    """Generate text embedding asynchronously using specified model and provider"""
    model, provider_name = _parse_model(model_provider)
    prompt = _get_prompt('usr', 'emb', model, provider_name, prompt_options)
    sys_prompt = _get_prompt('sys', 'emb', model, provider_name, prompt_options)

    emb_text = f"{sys_prompt + '\n\n' if sys_prompt else ''}{prompt + '\n\n' if prompt else ''}{text}"

    provider = get_provider(provider_name)
    return await provider.generate_embedding_async(emb_text, model)

async def generate_text_async(action: str, text: str = '', history: List[Tuple[str, str]] = [], image: str = '',
                            model_provider: str = config.GEN_MODEL, prompt_options: Optional[Dict[str, Any]] = None, **kwargs) -> str:
    """Generate text response asynchronously using specified model and provider"""
    model, provider_name = _parse_model(model_provider)
    
    usr_prompt = _get_prompt('usr', action, model, provider_name, prompt_options)
    sys_prompt = _get_prompt('sys', action, model, provider_name, prompt_options)

    prompt = f"{usr_prompt + '\n\n' if usr_prompt else ''}{text}"    

    provider = get_provider(provider_name)
    return await provider.generate_text_async(prompt, sys_prompt, history, image, model=model, action=action, **kwargs)

async def get_summarization_async(text: str, model_provider: str = config.SUM_MODEL, prompt_options: Dict[str, Any] = {}) -> str:
    """Generate text summarization asynchronously"""
    return await generate_text_async('sum', text, model_provider=model_provider, prompt_options=prompt_options)

async def get_image_description_async(image: str, model_provider: str = config.VSN_MODEL, prompt_options: Dict[str, Any] = {}) -> str:
    """Generate image description asynchronously using vision capabilities"""
    return await generate_text_async('vsn', text='', image=image, model_provider=model_provider, prompt_options=prompt_options)

if __name__ == '__main__':
    test_text_models = [
        'gpt-5-nano@openai',
        'gemini-2.5-flash@google',
        'gpt-oss:20b@ollama',
        'gemma3:27b-32k@ollama'
    ]
    test_vision_models = [
        'gpt-5-nano@openai',
        'gemini-2.5-flash@google',
        'gemma3:27b-32k@ollama'
    ]
    test_embedding_models = [
        'text-embedding-3-large@openai',
        'gemini-embedding-001@google',
        'mxbai-embed-large@ollama'
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
        'test/clip0419.png',
        'test/clip0783.png'
    ]
    
    # Test image description
    #test_vision_models = ['gemma3:27b-32k@ollama']
    #test_vision_models = ['llava:34b@ollama']
    for image in test_images:
        for model in test_vision_models:
            try:
                logger.info(f"Testing image {image} description with model {model}")
                response = get_image_description(image, model_provider=model)
                logger.info(f"Image description: {response}")
            except Exception as e:
                logger.error(f"Error testing {model} with image {image}: {e}")

    # Test text generation
    for model in test_text_models:
        try:
            logger.info(f"Testing text model: {model}")
            response = get_summarization(test_text, model_provider=model)
            logger.info(f"Response: {response}")
        except Exception as e:
            logger.error(f"Error testing {model}: {e}")
    
    # Test embeddings
    for model in test_embedding_models:
        try:
            logger.info(f"Testing embedding model: {model}")
            response = get_embedding(test_text, model_provider=model)
            logger.info(f"Embedding dimensions: {len(response)}")
        except Exception as e:
            logger.error(f"Error testing {model}: {e}")
    
    logger.info("Testing completed")
