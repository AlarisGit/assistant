Analyze this image from an SMS platform user manual and return a JSON object with structured information for RAG-based search and context replacement.

**Analysis Guidelines**:
1. **Extract all visible text** - buttons, labels, headers, field names, values, messages
2. **Identify the SMS platform feature** - what functionality does this relate to?
3. **Describe key UI elements** - only mention layout/positioning for complex interfaces
4. **Focus on searchable content** - use terminology users would search for
5. **Stay factual** - only describe what is explicitly visible, do not infer or assume functionality

**Required JSON Output Format**:
```json
{
  "complexity": "simple|medium|complex",
  "feature_name": "Main SMS platform feature or UI component name",
  "description": "Factual description of what is visible (length based on complexity)",
  "visible_text": ["array", "of", "all", "visible", "text", "elements"],
  "ui_elements": ["list", "of", "interactive", "elements", "if", "applicable"],
  "text_for_search": "Optimized text combining feature name and key terms for RAG search",
  "assumptions": "Any inferred functionality or context (clearly marked as non-factual)"
}
```

**Complexity Levels**:
- **Simple**: Single button, icon, small dialog → brief description
- **Medium**: Form with multiple fields, interface section → moderate detail  
- **Complex**: Full screen, multi-panel dashboard, detailed diagram → comprehensive description

**Example Output**:
```json
{
  "complexity": "simple",
  "feature_name": "Stop Button",
  "description": "Red button with white text labeled 'Stop'",
  "visible_text": ["Stop"],
  "ui_elements": ["button"],
  "text_for_search": "Stop button SMS campaign control",
  "assumptions": "Likely used to halt or cancel an ongoing process"
}
```
