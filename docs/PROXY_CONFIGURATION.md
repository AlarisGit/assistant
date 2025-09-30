# Proxy Configuration Guide

The assistant supports flexible proxy configuration for HTTP/HTTPS requests, particularly useful for connecting to local Ollama instances or other services.

## Configuration Options

Add the `PROXY` setting to your `.env` file with one of three values:

### 1. System Default (`PROXY=SYSTEM`)

Uses environment variables for proxy configuration. This is the traditional approach that respects system-wide proxy settings.

```bash
# In .env file
PROXY=SYSTEM
```

**Behavior:**
- Reads `http_proxy`, `https_proxy`, `HTTP_PROXY`, `HTTPS_PROXY` environment variables
- Follows system proxy configuration
- Standard behavior for most applications

**Use when:**
- You have system-wide proxy configuration
- You want consistent proxy behavior across all applications
- Your network requires proxy for external connections

---

### 2. No Proxy (`PROXY=NO`)

Disables all proxy usage, even if environment variables are set. Forces direct connections.

```bash
# In .env file
PROXY=NO
```

**Behavior:**
- Ignores all proxy environment variables
- Makes direct connections to all services
- Explicitly disables proxy for all HTTP/HTTPS requests

**Use when:**
- Connecting to local services (e.g., Ollama on LAN)
- Environment has proxy variables set but you want direct connection
- Testing or debugging network issues
- **This is the recommended default for local Ollama usage**

---

### 3. Custom Proxy (`PROXY=ip:port`)

Specifies a custom proxy server regardless of environment variables.

```bash
# In .env file
PROXY=127.0.0.1:10809
# or
PROXY=192.168.1.100:8080
```

**Behavior:**
- Uses specified proxy for all HTTP/HTTPS requests
- Ignores environment variables
- Can specify IP:port or hostname:port

**Use when:**
- Need to use specific proxy server
- Testing with different proxy configurations
- Proxy address differs from system settings

---

## Common Scenarios

### Scenario 1: Local Ollama on LAN

```bash
OLLAMA_HOST=192.168.1.42
OLLAMA_PORT=11434
PROXY=NO
```

This ensures direct connection to your local Ollama instance, bypassing any system proxy settings.

### Scenario 2: Corporate Network with Proxy

```bash
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
PROXY=SYSTEM
```

Uses corporate proxy settings from environment for external connections.

### Scenario 3: Development with Proxy Testing

```bash
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
PROXY=127.0.0.1:10809
```

Routes all requests through specific local proxy for testing/debugging.

---

## Technical Details

### How It Works

The `config.get_proxy_settings()` function returns proxy configuration for the `requests` library:

- **`PROXY=SYSTEM`**: Returns `None` → requests uses environment variables
- **`PROXY=NO`**: Returns `{'http': None, 'https': None}` → requests disables proxy
- **`PROXY=custom`**: Returns `{'http': 'http://custom', 'https': 'http://custom'}` → requests uses custom proxy

### Affected Services

Proxy configuration applies to all HTTP/HTTPS requests in the system:

**Fully Configurable (respects all three modes):**
- **Ollama API calls** (text generation, embeddings, image loading)
- **Web crawling** (HTML fetching, image downloading in crawl.py)
- **HTTP image loading** (when processing image URLs)
- **OpenAI API calls** (via httpx client configuration)

**Environment-Variable Only (use PROXY=SYSTEM or PROXY=NO+env vars):**
- **Google Gemini API calls** (SDK respects http_proxy/https_proxy environment variables)

### Code Locations

Proxy configuration is implemented across multiple files:

- **Configuration**: `src/config.py`
  - `PROXY` environment variable
  - `get_proxy_settings()` helper function
  
- **LLM Providers**: `src/llm.py`
  - `OllamaProvider` - Full proxy support via requests library
  - `OpenAIProvider` - Full proxy support via httpx client
  - `GoogleProvider` - Environment variable support only
  
- **Web Crawling**: `src/crawl.py`
  - HTML fetching with proxy support
  - Image downloading with proxy support

---

## Troubleshooting

### Problem: "ProxyError: Unable to connect to proxy"

**Symptoms:**
```
ProxyError(MaxRetryError("HTTPConnectionPool(host='127.0.0.1', port=10809): 
Max retries exceeded with url: http://192.168.1.42:11434/api/chat 
(Caused by ProxyError('Unable to connect to proxy', ...))"))
```

**Cause:** System has proxy environment variables set, but proxy server is not accessible.

**Solution:** Add `PROXY=NO` to `.env` file to bypass proxy.

---

### Problem: Cannot connect to local Ollama

**Symptoms:**
- Connection timeout to local Ollama instance
- Works when running commands directly but fails from assistant

**Solution:**
1. Check your environment variables: `env | grep -i proxy`
2. If proxy variables are set, add `PROXY=NO` to `.env`
3. Restart the assistant service

---

### Problem: Need different proxy for different services

**Current Limitation:** The `PROXY` setting applies globally to most HTTP/HTTPS requests.

**Workaround:** 
- Use `PROXY=NO` for local services (Ollama)
- Configure external services (OpenAI) use the same global setting
- **For Google Gemini**: Use `PROXY=SYSTEM` and set environment variables as the SDK only supports env vars
- Consider network-level proxy rules if fine-grained control is needed

---

### Problem: Google Gemini not respecting PROXY=NO

**Cause:** Google's genai SDK does not support direct proxy configuration through the `PROXY` setting.

**Solution:**
1. Set `PROXY=SYSTEM` in `.env`
2. Unset proxy environment variables: 
   ```bash
   unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
   ```
3. Restart the assistant

**Note:** For Google Gemini, the only proxy control is through environment variables.

---

## Checking Current Configuration

To verify your proxy settings:

```python
import sys
import os
sys.path.append('src')
import config

print(f"PROXY setting: {config.PROXY}")
print(f"Proxy config: {config.get_proxy_settings()}")
```

Or check environment:
```bash
env | grep -i proxy
```

---

## Best Practices

1. **For local Ollama**: Always use `PROXY=NO`
2. **For corporate networks**: Start with `PROXY=SYSTEM` and adjust if needed
3. **Document your setup**: Comment in `.env` why specific proxy setting is used
4. **Test after changes**: Verify connections work after changing proxy settings

---

## Example .env Configurations

### Development (Local Ollama)
```bash
OLLAMA_HOST=192.168.1.42
OLLAMA_PORT=11434
PROXY=NO  # Direct connection to local Ollama
```

### Production (Corporate Network)
```bash
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
PROXY=SYSTEM  # Use corporate proxy settings
```

### Testing (Custom Proxy)
```bash
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
PROXY=127.0.0.1:8888  # Route through local debugging proxy
```
