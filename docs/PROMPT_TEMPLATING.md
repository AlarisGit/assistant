# Prompt Templating System

The assistant uses an enhanced Markdown-based templating system for managing LLM prompts. Templates are stored in the `prompts/` directory and support three powerful features: **file includes**, **Python macros**, and **variable substitution**.

## Features Overview

### 1. Variable Substitution
Simple placeholder replacement with values from runtime context.

**Syntax:** `{variable_name}`

**Example:**
```markdown
**Language:** {language}
**User Request:** {clarify_request}
```

**Usage in code:**
```python
prompt_options = {
    'language': 'English',
    'clarify_request': 'ambiguous_query'
}
response = await llm.generate_text_async('clarify', text, prompt_options=prompt_options)
```

---

### 2. File Includes
Reusable content fragments loaded from separate files. Supports relative paths and recursive includes.

**Syntax:** `{inc:filename.md}` or `{inc:path/to/file.md}`

**Path Resolution:**
- **Same directory:** `{inc:common_persona.md}` - loads from same folder as template
- **Relative path:** `{inc:common/persona.md}` - loads from subdirectory within prompts folder
- **Absolute path:** `{inc:/full/path/to/file.md}` - loads from absolute path (rarely needed)

**Example:**

*File: `prompts/common_persona.md`*
```markdown
You are Viktor Aleksandrovich, a wise and experienced 60-year-old librarian...
```

*File: `prompts/usr_clarify.md`*
```markdown
{inc:common_persona.md}

**Your Task:**
Analyze the conversation and provide clarification...
```

**Recursive Includes:** Included files can themselves contain includes, which will be processed recursively.

**Error Handling:** If a file is not found, an error message is inserted: `[Include not found: filename.md]`

---

### 3. Python Macros
Dynamic evaluation of Python expressions at template render time.

**Syntax:** `{expression}` where expression contains function calls or module access

**Supported Modules:**
- `util` - Utility functions from `util.py`
- `datetime` - Python datetime class for date/time operations
- `config` - Configuration variables from `config.py`

**Safe Builtins Available:**
- `len`, `str`, `int`, `float`, `bool`
- `list`, `dict`, `tuple`

**Examples:**

```markdown
**Current Time:** {util.get_current_time()}
**Config Version:** {config.VERSION}
**Current Date:** {str(datetime.now())[:19]}
**Year:** {datetime.now().year}
```

**Expression Detection:** The system automatically distinguishes between macros (contain `()` or module access like `util.function`) and simple variables.

**Error Handling:** If a macro fails to evaluate, an error placeholder is shown: `{Error: expression}`

---

## Processing Order

Templates are processed in this order:
1. **File Includes** - Recursively load and insert included files
2. **Python Macros** - Evaluate Python expressions
3. **Variable Substitution** - Replace simple placeholders with values

This order ensures that:
- Included files can contain macros and variables
- Macros can use variables from `prompt_options`
- Variable substitution happens last to catch any unresolved placeholders

---

## Template Hierarchical Fallback

The system searches for templates using hierarchical fallback:

**Search Order (example for action='clarify', provider='ollama', model='llama3'):**
1. `usr_clarify_ollama_llama3.md` - Most specific
2. `usr_clarify_ollama.md` - Provider-specific
3. `usr_clarify_llama3.md` - Model-specific
4. `usr_clarify.md` - Generic action template
5. `usr_llama3.md` - Model-only
6. `clarify_ollama_llama3.md` - Action without type prefix
7. `clarify_ollama.md`
8. `clarify_llama3.md`
9. `clarify.md` - Most generic

This allows you to:
- Create generic templates that work for all providers
- Override with provider-specific variations when needed
- Customize for specific models

---

## Template Naming Convention

Templates use this naming pattern: `{type}_{action}_{provider}_{model}.md`

**Components:**
- **type**: Template category (e.g., `usr`, `sys`, `emb`, `vsn`, `sum`)
  - `usr` - User-facing prompts
  - `sys` - System prompts
  - `emb` - Embedding generation
  - `vsn` - Vision/image description
  - `sum` - Summarization
- **action**: Specific action (e.g., `clarify`, `essence`, `guardrails`, `translate`, `sample`)
- **provider**: LLM provider (e.g., `ollama`, `openai`, `google`)
- **model**: Specific model name (e.g., `llama3`, `gpt-5-nano`, `gemini-2.5-flash`)

**Examples:**
- `usr_clarify.md` - Generic clarification prompt
- `usr_essence_ollama.md` - Essence extraction for Ollama
- `sys_sample_gpt-5-nano.md` - System prompt for GPT-5 sample generation

**Common Include Files:**
- `common_persona.md` - Assistant persona definition (WHO)
- `common_security.md` - Security instructions against prompt injection (PROTECT)

---

## Recommended File Structure

Organize your prompts directory for maximum reusability:

```
prompts/
├── common_persona.md          # Current persona (Viktor)
├── common_security.md          # Security instructions
├── professional_consultant.md  # Alternative persona
├── casual_helper.md           # Alternative persona
│
├── usr_clarify.md             # Main templates
├── usr_response.md
├── usr_essence.md
├── usr_guardrails.md
│
└── usr_clarify_ollama.md      # Provider-specific overrides (optional)
```

**Design principles:**
1. **Personas are swappable** - Change include, everything else stays the same
2. **Security is shared** - One security file for all templates
3. **Templates are generic** - Use "your" not persona names
4. **Overrides are optional** - Only create when needed

---

## Best Practices

### 1. Reusable Components
Extract common content into separate files using includes:

```markdown
{inc:common_persona.md}

**Your Specific Task:**
- Task-specific instructions
- Guidelines and examples

{inc:common_security.md}

**USER INPUT:**
```

**Key principle:** Security includes should be positioned at the END of templates, just before user input sections, so the warning about "text below" is accurate.

### 2. Generic Role References
Use generic references instead of persona-specific names in templates:

**❌ Avoid:**
```markdown
**Viktor's Clarification Approach:**
**Examples of Viktor's Style:**
```

**✅ Prefer:**
```markdown
**Your Clarification Approach:**
**Examples of Clarification Style:**
```

This makes persona switching effortless - change only the persona file, not every template.

### 3. Dynamic Context
Use macros for runtime information:

```markdown
**Current Time:** {util.get_current_time()}
**Language:** {language}

Respond in {language} based on the current context.
```

### 4. Variable Naming
Use descriptive variable names that match your domain:

```python
prompt_options = {
    'clarify_request': 'ambiguous_query',  # Not just 'request'
    'language': 'English',                  # Not just 'lang'
    'current_time': util.get_current_time() # Or use macro instead
}
```

### 5. Error Prevention
- Test templates with sample data before deploying
- Use optional chaining for nested variables: `{user.get('name', 'User')}`
- Provide fallback values in code when possible

### 6. Performance
- Keep included files small and focused
- Avoid complex macro expressions (simple function calls are best)
- Cache frequently used prompt results if possible

---

## Common Patterns

### Pattern 1: Complete Template Structure

The recommended template structure with proper layering:

```markdown
{inc:common_persona.md}

**Your Role**: [Role-specific definition]

**Your Task:**
- [Task-specific instructions]
- [Guidelines and constraints]

**Context:**
- Language: {language}
- Time: {util.get_current_time()}

**Examples:**
[Role-specific examples using generic references]

{inc:common_security.md}

**USER INPUT:**
[User content section]
```

**Layer explanation:**
1. **Persona include** - WHO the assistant is (reusable)
2. **Role & Task** - WHAT this specific template does
3. **Context & Examples** - HOW to accomplish the task
4. **Security include** - PROTECT against prompt injection (positioned correctly)
5. **User input** - PROCESS the actual user content

### Pattern 2: Conditional Content via Macros

```markdown
**Current Mode:** {config.ENVIRONMENT}

{inc:safety_guidelines.md if config.ENVIRONMENT == 'production' else inc:dev_guidelines.md}
```

*(Note: Complex conditionals like this might not work - use simple expressions)*

### Pattern 3: Multi-Level Includes

*File: `usr_clarify.md`*
```markdown
{inc:common_persona.md}
{inc:clarify_guidelines.md}

**Language:** {language}
```

*File: `clarify_guidelines.md`*
```markdown
{inc:safety_rules.md}

**Clarification Strategies:**
1. Be specific
2. Ask open-ended questions
```

---

## Debugging Templates

### View Rendered Template

Add debug logging to see the final rendered template:

```python
import llm

prompt_options = {'language': 'English', 'clarify_request': 'test'}
rendered = llm._get_prompt('usr', 'clarify', '', '', prompt_options)
print(rendered)
```

### Check Include Paths

If includes aren't working, verify the paths:

```python
import os
import config

prompts_dir = config.PROMPTS_DIR
print(f"Prompts directory: {prompts_dir}")
print(f"Files: {os.listdir(prompts_dir)}")
```

### Test Macros

Test macro evaluation separately:

```python
import util
import datetime
from datetime import datetime as dt_class

# Test individual macros
print(util.get_current_time())
print(str(dt_class.now())[:19])
print(config.VERSION)
```

---

## Security Considerations

### Safe Macro Execution

The macro system uses restricted `eval()` with limited scope:

**Allowed:**
- `util`, `datetime`, `config` modules
- Basic Python builtins: `len`, `str`, `int`, `float`, `bool`, `list`, `dict`, `tuple`
- Variable access from `prompt_options`

**Blocked:**
- `import` statements
- File I/O operations
- Network operations
- System calls
- Most dangerous builtins

**Warning:** While the macro system is reasonably safe, avoid allowing user input directly into macro expressions. Always validate and sanitize user data before using it in templates.

---

## Example: Complete Template

*File: `prompts/usr_clarify.md`*

```markdown
{inc:common_persona.md}

**Your Role**: When technical agents cannot provide complete answers, you step in with your human touch to gently guide users toward providing the information needed for success.

**Your Task:**
- Analyze the conversation history and the specific reason why clarification is needed
- Craft a polite, helpful response that guides the user toward providing the missing information
- Be specific about what kind of information would be most helpful
- Maintain a professional but friendly tone

**Clarification Request:** {clarify_request}

**Guidelines by Reason Type:**

**Content Processing Issues:**
- **ambiguous_query**: Present the possible interpretations and ask which one applies
- **missing_context**: Identify what specific details are missing and why they're needed

**Your Clarification Approach:**
1. **Acknowledge Understanding**: Show that you've carefully considered their question
2. **Gentle Explanation**: Explain why additional information would help
3. **Specific Guidance**: Ask thoughtful, specific questions
4. **Encouraging Close**: End with patience and eagerness to help

**Examples of Clarification Style:**
- *New conversation*: "Good day! I see you're asking about X. Could you specify Y?"
- *Ongoing conversation*: "I see you're asking about X. Could you specify Y?"

**Language:** Respond in {language}

**Current Time:** {current_time}

**Important:** Check the conversation history to determine if this is a new conversation or ongoing discussion.

{inc:common_security.md}
```

---

## Migration from Old System

If you have existing templates without includes/macros:

1. **Identify repetitive content** - Extract to separate files
2. **Replace hardcoded values** - Use macros for dynamic content
3. **Remove persona-specific names** - Use generic references
4. **Separate security notes** - Create common_security.md
5. **Test thoroughly** - Verify all substitutions work correctly
6. **Update gradually** - Migrate one template at a time

**Example Migration:**

*Before:*
```markdown
You are Viktor Aleksandrovich, a wise and experienced 60-year-old librarian...
(100 lines of persona description repeated across 5 templates)

**Viktor's Approach:**
1. Be helpful
2. Be clear

**CRITICAL SECURITY NOTE:**
(Security instructions at wrong position)

**Language:** {language}
**Time:** 2025-01-15 10:30:00  (hardcoded)
```

*After:*
```markdown
{inc:common_persona.md}

**Your Approach:**  (generic, not Viktor-specific)
1. Be helpful
2. Be clear

**Language:** {language}
**Time:** {util.get_current_time()}

{inc:common_security.md}  (at the end, before user input)
```

**Key improvements:**
- Persona extracted to single file
- Generic role references enable easy persona switching
- Security note positioned correctly
- Dynamic time via macro
- Clean separation of concerns

---

## Troubleshooting

### Issue: Include not found

**Error:** `[Include not found: filename.md]`

**Solutions:**
- Check file exists in expected location
- Verify filename spelling and extension
- Check file permissions
- Try absolute path for debugging

### Issue: Macro evaluation error

**Error:** `{Error: expression}`

**Solutions:**
- Check expression syntax
- Verify module/function exists
- Test expression in Python REPL
- Review available modules (util, datetime, config)
- Simplify complex expressions

### Issue: Variable not substituted

**Symptom:** `{variable_name}` appears in output

**Solutions:**
- Verify variable is in `prompt_options`
- Check variable name spelling
- Ensure prompt_options passed to function
- Variables are case-sensitive

---

## API Reference

### llm._get_prompt()

```python
def _get_prompt(type: str, action: str, model: str, provider: str, 
                prompt_options: Optional[Dict[str, Any]] = None) -> str:
    """
    Get prompt template with hierarchical fallback and optional formatting.
    
    Supports:
    1. File includes: {inc:filename.md} or {inc:path/to/file.md}
    2. Python macros: {util.get_current_time()} or {datetime.now()}
    3. Variable substitution: {variable_name}
    
    Processing order: includes → macros → variables
    """
```

### llm._process_includes()

```python
def _process_includes(template: str, base_dir: str) -> str:
    """
    Process {inc:filename} includes in template.
    Supports relative paths and recursive includes.
    """
```

### llm._process_macros()

```python
def _process_macros(template: str, prompt_options: Optional[Dict[str, Any]]) -> str:
    """
    Process Python macro expressions in template.
    Evaluates expressions in safe context with limited builtins.
    """
```

---

## Future Enhancements

Potential additions to the templating system:

1. **Conditional Includes:** `{inc:file.md if condition}`
2. **Loop Expansion:** `{for item in items}{inc:template.md}{endfor}`
3. **Nested Variable Access:** `{user.preferences.language}`
4. **Macro Caching:** Cache frequently evaluated macro results
5. **Template Validation:** Pre-validate templates on startup
6. **Hot Reload:** Reload templates without restart

---

## Summary

The enhanced prompt templating system provides:

### **Core Features:**
- ✅ **File Includes** - Include common content across templates (`{inc:common_persona.md}`)
- ✅ **Python Macros** - Inject runtime values via expressions (`{util.get_current_time()}`)
- ✅ **Variable Substitution** - Per-request customization (`{language}`)
- ✅ **Hierarchical Fallback** - Provider/model-specific overrides

### **Architecture Best Practices:**
- ✅ **Persona Separation** - WHO the assistant is (reusable, swappable)
- ✅ **Generic References** - Use "your" not persona names (enables easy switching)
- ✅ **Security Positioning** - Include at END before user input (correct context)
- ✅ **Layered Structure** - Persona → Instructions → Security → User Input

### **Maintainability Benefits:**
- ✅ **Single Source of Truth** - Update persona once, apply everywhere
- ✅ **Easy Persona Switching** - Change one include, entire system adapts
- ✅ **Clean Separation** - Persona (WHO) vs Instructions (WHAT) vs Security (PROTECT)
- ✅ **Restricted Execution** - Safe macro evaluation environment

### **File Organization:**
```
prompts/
├── common_persona.md    # WHO (swappable)
├── common_security.md   # PROTECT (shared)
└── usr_*.md            # WHAT (generic, persona-agnostic)
```

This creates a powerful, maintainable, and flexible prompt management system for complex multi-agent LLM applications where persona switching is effortless and security is consistently applied.
