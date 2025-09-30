"""
================================================================================
Assistant Application Layer - Agent Implementations
================================================================================

This module contains the application-level agent implementations that define
the business logic for the conversational AI assistant. It demonstrates the
separation of concerns between system infrastructure (agent.py) and application
logic (assistant.py).

Agent Architecture:
- CommandAgent: Handles action-based commands (1 instance - fast processing)
- ManagerAgent: Orchestrates FSM-based message pipeline (1 instance - routing only)
- LangAgent: Detects language from conversation history (1 instance - fast detection)
- TranslationAgent: Translates non-English queries (2 instances - LLM translation)
- EssenceAgent: Extracts canonical questions (2 instances - LLM context analysis)
- GuardrailsAgent: LLM-powered documentation scope enforcement (2 instances)
- SearchAgent: Multi-level search across documentation (2 instances - semantic search)
- ClarificationAgent: Detects ambiguity and generates questions (2 instances - LLM analysis)
- AugmentationAgent: Crafts enhanced LLM prompts (2 instances - LLM prompt engineering)
- ResponseAgent: Generates final responses using flagship LLM (2 instances)
- QualityAgent: Final response validation with return capability (2 instances - LLM validation)

Key Features:
- Clean separation between command handling and FSM orchestration
- Distributed conversation memory with automatic cleanup
- Custom logging for agent-specific insights
- Simplified envelope handling with env.final() method
- Zero-parameter constructors with automatic registration

External API:
- process_user_message(): Main entry point for user messages
- clear_conversation_history(): Clear message history only
- clear_all_conversation_data(): Complete conversation cleanup

Usage:
    # Process user message through pipeline
    result = await process_user_message("user123", "Hello world")
    
    # Clear conversation history
    success = await clear_conversation_history("user123")
"""

import logging
from typing import Dict, Any
import asyncio
import time
from datetime import datetime
import json

import config
from agent import process_request, broadcast_command, BaseAgent, Envelope
import llm
import util

logger = logging.getLogger(__name__)

class CommandAgent(BaseAgent):
    """Handles action-based command requests.
    
    This agent processes administrative commands that don't require FSM routing:
    - clear_history: Clear conversation message history only
    - clear_all: Complete conversation data cleanup
    
    The agent automatically routes final results back to the external caller
    using the env.final() convenience method.
    
    Role: "command" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Process action-based command envelope.
        
        Args:
            env: Envelope containing action in payload
            
        Returns:
            Envelope with result or error, configured for final delivery
        """
        action = env.payload.get("action")
        if action == "clear_history":
            # Clear message history only, keep other memory intact
            await self.cleanup_message_history(env.conversation_id)
            logger.info(f"[CommandAgent] Cleared message history for conversation {env.conversation_id}")
            env.payload["result"] = "Message history cleared successfully"
            return env.final()
        
        elif action == "clear_all":
            # Clear ALL conversation data (complete cleanup)
            await self.cleanup_memory(env.conversation_id)
            logger.info(f"[CommandAgent] Cleared ALL data for conversation {env.conversation_id}")
            env.payload["result"] = "All conversation data cleared successfully"
            return env.final()

        elif action == "set_language":
            language = env.payload["language"]
            memory = await self.get_memory(env.conversation_id)
            default_prefs = {
                "language": None
            }
            prefs = await memory.get("user_preferences", default_prefs)
            if language is None:
                # Reset to default language
                prefs["language"] = None
                env.payload["response"] = "Language reset to default"
            elif language in config.SUPPORTED_LANGUAGES:
                # Set to specific language
                prefs["language"] = language
                env.payload["response"] = f"Language set successfully to {language}"
            else:
                # Invalid language code
                env.payload["response"] = "Unsupported language"
                # Don't change the preference for invalid input
                return env.final()
            await memory.set("user_preferences", prefs)
            await self.log(env.conversation_id, f"User preferences: {prefs}")
            return env.final()

        # Unknown action - return error
        logger.error(f"[CommandAgent] Unknown action '{action}' for {env}")
        env.payload.setdefault("errors", []).append({
            "code": "command.action",
            "message": f"Unknown action '{action}'",
        })
        return env

class ManagerAgent(BaseAgent):
    """FSM-based pipeline orchestrator for message processing.
    
    This agent implements a finite state machine that routes messages through
    different processing stages:
    
    State Machine:
    1. start -> lang: Route user message to language detection
    2. lang -> final: Process detected language and return result
    
    Features:
    - Distributed conversation memory for history and preferences
    - Automatic message counting and user preference tracking
    - Custom logging for pipeline decision making and statistics
    - Integration with memory cleanup and statistics reporting
    
    Role: "manager" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Route messages through the pipeline stages.
        
        Enhanced with memory usage for conversation tracking and user preferences.
        
        Returns modified envelope with updated routing information.
        """
        stage = env.payload.get("stage", "start")
        
        # Get conversation memory
        memory = await self.get_memory(env.conversation_id)

        if stage == "start":
            user_text = env.payload.get("text", "")
            messages = await memory.get_messages(limit=10)
            for i, message in enumerate(messages):
                await self.log(env.conversation_id, f"History item {i}: {message}")
            await memory.add_message("user", user_text, {"message_id": env.message_id})
            await self.log(env.conversation_id, f"Added user message to history: {user_text}")
            env.target_role = "lang"
            env.payload["stage"] = "lang"
            return env

#        if stage == "translation": #Temporary block for clarification testing
#            env.payload["needs_clarification"] = True
#            env.payload["clarification_reason"] = "not_enough_info"
#            env.payload["clarification_message"] = "I need more information to translate this text. Please provide more details."
        
        if 'needs_clarification' in env.payload and env.payload['needs_clarification']:
            env.target_role = "clarification"
            env.payload["stage"] = "clarification"
            return env

        if stage == "lang":
            env.target_role = "translation"
            env.payload["stage"] = "translation"
            return env

        if stage == "translation":
            env.target_role = "essence"
            env.payload["stage"] = "essence" #can be overriden by manager if clarification needed
            return env
        
        if stage == "essence":
            env.target_role = "guardrails"
            env.payload["stage"] = "guardrails" #can be overriden by manager if clarification needed
            return env
        
        if stage == "guardrails":
            env.target_role = "response"
            env.payload["stage"] = "response" #can be overriden by manager if clarification needed
            return env
        
        if stage in ["response", "clarification"]:
            env.target_role = "manager"
            env.payload["stage"] = "final"
            return env
            
        if stage == 'final':
            if 'response' not in env.payload or not env.payload['response']:
                env.payload['response'] = f"No response"
            await memory.add_message("assistant", env.payload['response'], {"message_id": env.message_id})
            await self.log(env.conversation_id, f"Added assistant message to history: {env.payload['response']}")
            message_count = await memory.get_message_count()
            await self.log(env.conversation_id, f"Pipeline complete: final_length={len(env.payload.get("response", ""))} chars total_messages={message_count}")
            return env.final()
        
        # Unknown stage - return error
        env.payload.setdefault("errors", []).append({
            "code": "manager.stage",
            "message": f"Unknown stage '{stage}'",
        })
        return env.final()

class LangAgent(BaseAgent):
    """Language detection agent using conversation history.
    
    Analyzes recent conversation messages to detect the primary language
    being used by the user. Supports Cyrillic (Russian) and Chinese detection
    with English as the default fallback.
    
    Detection Logic:
    - Examines up to 6 recent user messages
    - Checks for Cyrillic characters (Russian)
    - Checks for Chinese characters
    - Defaults to English if no specific patterns found
    
    Role: "lang" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        text = env.payload.get("text", "")
        await self.log(env.conversation_id, f"Processing text: {text}")
        memory = await self.get_memory(env.conversation_id)
        
        # Check user preferences first
        prefs = await memory.get("user_preferences", {})
        preferred_language = prefs.get("language")
        
        if preferred_language is not None:
            # User has explicitly set a language preference
            language = preferred_language
            confidence = 1.0
            await self.log(env.conversation_id, f"Using user preferred language: {language}")
        else:
            # Auto-detect language from conversation history
            language = 'en'
            confidence = 0.9
            messages = await memory.get_messages(limit=6)
            
            # Collect all user messages for analysis
            user_texts = []
            for message in messages:
                await self.log(env.conversation_id, f"Message: {message}")
                if message.get("role") == "user":
                    user_texts.append(message.get("content", ""))
            
            # Add current message to analysis
            current_text = env.payload.get("text", "")
            if current_text:
                user_texts.append(current_text)
            
            # Analyze combined text for language detection
            combined_text = " ".join(user_texts)
            await self.log(env.conversation_id, f"Analyzing combined text ({len(combined_text)} chars): {combined_text[:100]}...")
            
            if util._is_cyrillic_text(combined_text):
                language = 'ru'
                confidence = 0.95
                await self.log(env.conversation_id, f"Detected Russian: sufficient Cyrillic content in {len(combined_text)} characters")
            elif util._is_chinese_text(combined_text):
                language = 'zh'
                confidence = 0.95
                await self.log(env.conversation_id, f"Detected Chinese: sufficient Chinese content in {len(combined_text)} characters")
            else:
                await self.log(env.conversation_id, f"Defaulting to English: insufficient non-Latin content in {len(combined_text)} characters")
            
            await self.log(env.conversation_id, f"Auto-detected language: {language} ({config.SUPPORTED_LANGUAGES.get(language, "English")})")

        await memory.set("language", language)
        env.payload["language"] = language
        env.payload["confidence"] = confidence
        await self.log(env.conversation_id, f"Final language: {language} ({config.SUPPORTED_LANGUAGES.get(language, "English")}) (confidence: {confidence})")
 
        return env

class TranslationAgent(BaseAgent):
    """Mandatory translation and text normalization for all queries.
    
    Responsibilities:
    - Always process all queries regardless of detected language
    - Translate non-English queries to English for optimal search performance
    - Spell-check and correct English queries (typos, grammar, clarity)
    - Handle technical terminology correctly
    - Maintain translation/correction confidence scores
    
    Role: "translation" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Process translation/normalization request.
        
        Reads: text, language
        Writes: text_eng
        """
        # Extract input attributes
        text = env.payload.get("text", "")
        language = env.payload.get("language", "en")
        await self.log(env.conversation_id, f"Translation/normalization: text='{text[:50]}...' language={language}")
        
        memory = await self.get_memory(env.conversation_id)
        # Use raw history for translation to avoid circular dependency and preserve original context
        history = await memory.get_history(limit=config.ASSISTANT_HISTORY_LIMIT, normalized=False)
        prompt_options = {'language': config.SUPPORTED_LANGUAGES.get(language, "English")}
        
        # Generate normalized English text
        normalized_text = await llm.generate_text_async('translate', text, history, prompt_options=prompt_options)
        env.payload["text_eng"] = normalized_text
        
        # Store normalized content in the latest user message metadata
        norm_info = {
            "type": "translation" if language != "en" else "correction",
            "source_lang": language,
            "confidence": 0.95,  # TODO: Get actual confidence from LLM
            "model": "llm",  # TODO: Get actual model info
            "ts": time.time()
        }
        
        await memory.update_last_user_message_normalized(normalized_text, norm_info)
        await self.log(env.conversation_id, f"Stored normalized text: '{normalized_text[:50]}...' (type: {norm_info['type']})")
        
        return env


class EssenceAgent(BaseAgent):
    """Extract canonical question from conversation history.
    
    Responsibilities:
    - Extract canonical question from conversation flow
    - Resolve pronouns and references ("it", "that feature", etc.)
    - Combine related questions from conversation history
    - Detect follow-up questions vs new topics
    - Clean and normalize queries for search
    
    Role: "essence" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Process essence extraction request.
        
        Reads: text, text_eng, conversation history from memory
        Writes: canonical_question, context_type, related_history
        + Universal clarification attributes (if uncertain)
        """
        # Extract input attributes
        text = env.payload.get("text", "")
        text_eng = env.payload.get("text_eng", text)
        
        await self.log(env.conversation_id, f"Essence extraction: text='{text[:50]}...' text_eng='{text_eng[:50]}...'")
        
        memory = await self.get_memory(env.conversation_id)
        history = await memory.get_history(limit=config.ASSISTANT_HISTORY_LIMIT, normalized=True, show_timestamps=True)
        text = env.payload.get("canonical_question", env.payload.get("text_eng", env.payload.get("text", "")))
        
        # EssenceAgent works in English as system language - no user language needed
        current_time = util.get_current_time()
        prompt_options = {
            'current_time': current_time
        }
        await self.log(env.conversation_id, f"Requesting essence extraction for: '{text[:100]}...' at {current_time}")
        response_text = await llm.generate_text_async('essence', text, history, prompt_options=prompt_options)
        
        await self.log(env.conversation_id, f"LLM response: {response_text}")
        
        try:
            response_text, response_dict = llm.parse_response(response_text)
            
            if response_dict:
                if 'canonical_question' in response_dict:
                    canonical_q = response_dict['canonical_question']
                    env.payload['canonical_question'] = canonical_q
                    await self.log(env.conversation_id, f"Extracted canonical question: '{canonical_q}'")
                elif 'clarification_reason' in response_dict:
                    reason = response_dict['clarification_reason']
                    message = response_dict.get('clarification_message', "Please provide more details to help me understand your question.")
                    env.payload['needs_clarification'] = True
                    env.payload['clarification_reason'] = reason
                    env.payload['clarification_message'] = message
                    await self.log(env.conversation_id, f"Requesting clarification: {reason} - {message}")
                else:
                    await self.log(env.conversation_id, "Error: LLM response missing required fields")
                    env.payload['needs_clarification'] = True
                    env.payload['clarification_reason'] = 'processing_error'
                    env.payload['clarification_message'] = "I had trouble understanding your request. Could you please rephrase it?"
            else:
                await self.log(env.conversation_id, "Error: No JSON found in LLM response")
                env.payload['needs_clarification'] = True
                env.payload['clarification_reason'] = 'processing_error'
                env.payload['clarification_message'] = "I had trouble understanding your request. Could you please rephrase it?"
                
        except json.JSONDecodeError as e:
            await self.log(env.conversation_id, f"JSON parsing error: {e}")
            env.payload['needs_clarification'] = True
            env.payload['clarification_reason'] = 'processing_error'
            env.payload['clarification_message'] = "I had trouble understanding your request. Could you please rephrase it?"
        return env


class GuardrailsAgent(BaseAgent):
    """LLM-powered documentation scope enforcement.
    
    Responsibilities:
    - Analyze query intent using LLM intelligence
    - Determine if queries are within documentation scope
    - Use universal clarification system for out-of-scope queries (no direct rejection)
    - Prevent access to general knowledge outside documentation
    - Provide analysis for ManagerAgent routing decisions
    
    Role: "guardrails" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Process guardrails analysis request.
        
        Reads: canonical_question, domain_scope (from config)
        Writes: Universal clarification attributes (when out-of-scope or abuse, injection and other misbehaviors detected)
        """
    
        text = env.payload.get("canonical_question", env.payload.get("text_eng", env.payload.get("text", "")))
        
        # EssenceAgent works in English as system language - no user language needed
        current_time = util.get_current_time()
        prompt_options = {
            'current_time': current_time
        }
        
        await self.log(env.conversation_id, f"Guardrails analysis: text='{text[:100]}...'")

        response_text = await llm.generate_text_async('guardrails', text, prompt_options=prompt_options)
        await self.log(env.conversation_id, f"LLM response: {response_text}")
        
        try:
            response_text, response_dict = llm.parse_response(response_text)
            
            if response_dict:
                if 'guardrails_passed' in response_dict:
                    env.payload['guardrails_passed'] = response_dict['guardrails_passed']
                    await self.log(env.conversation_id, f"Guardrails passed: {response_dict['guardrails_passed']}")
                elif 'clarification_reason' in response_dict:
                    reason = response_dict['clarification_reason']
                    message = response_dict.get('clarification_message', "Internal error: incomplete response from guardrails")
                    env.payload['needs_clarification'] = True
                    env.payload['clarification_reason'] = reason
                    env.payload['clarification_message'] = message
                    await self.log(env.conversation_id, f"Requesting clarification: {reason} - {message}")
                else:
                    await self.log(env.conversation_id, "Error: LLM response missing required fields")
                    env.payload['needs_clarification'] = True
                    env.payload['clarification_reason'] = 'processing_error'
                    env.payload['clarification_message'] = "Internal error: malformed response from guardrails"
            else:
                await self.log(env.conversation_id, "Error: No JSON found in LLM response")
                env.payload['needs_clarification'] = True
                env.payload['clarification_reason'] = 'processing_error'
                env.payload['clarification_message'] = "Internal error: no JSON found in guardrails response"
                
        except json.JSONDecodeError as e:
            await self.log(env.conversation_id, f"JSON parsing error: {e}")
            env.payload['needs_clarification'] = True
            env.payload['clarification_reason'] = 'processing_error'
            env.payload['clarification_message'] = "Internal error: JSON parsing error from guardrails"
        return env

class SearchAgent(BaseAgent):
    """Multi-level search across documentation using all metadata fields.
    
    Responsibilities:
    - Level 1: Keyword search using keywords field
    - Level 2: Semantic search using chunks embeddings  
    - Level 3: Hierarchical search using crumbs navigation paths
    - Level 4: Summary search using summary and keypoints
    - Rank and merge results from all search levels
    - Assess search quality and coverage
    
    Role: "search" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Process search request.
        
        Reads: canonical_question, text_eng
        Writes: search_results, search_context, search_stats, search_quality, search_error (if any)
        + Universal clarification attributes (if poor results)
        """
        # Extract input attributes
        canonical_question = env.payload.get("canonical_question", "")
        text_eng = env.payload.get("text_eng", canonical_question)
        
        await self.log(env.conversation_id, f"Search request: question='{canonical_question[:50]}...' text_eng='{text_eng[:50]}...'")
        
        # TODO: Implement actual multi-level search logic
        # For now, return placeholder results with quality assessment
        search_results = [
            {
                "content": f"Sample search result for: {canonical_question}",
                "source": "https://example.com/doc1",
                "score": 0.85,
                "metadata": {"crumbs": ["Documentation", "API"]}
            }
        ]
        
        avg_score = 0.85
        search_quality = "excellent" if avg_score > 0.8 else "good" if avg_score > 0.6 else "poor"
        
        env.payload["search_results"] = search_results
        env.payload["search_context"] = f"Context for query: {canonical_question}"
        env.payload["search_stats"] = {
            "total_results": len(search_results),
            "used_results": len(search_results),
            "context_length": 100,
            "avg_score": avg_score
        }
        env.payload["search_quality"] = search_quality
        
        # Request clarification if search quality is poor
        if search_quality == "poor":
            env.payload["needs_clarification"] = True
            env.payload["clarification_reason"] = "missing_details"
            env.payload["clarification_message"] = "Search results are insufficient - need more specific details"
            await self.log(env.conversation_id, "Requesting clarification: poor search quality")
        else:
            await self.log(env.conversation_id, f"Search completed: {len(search_results)} results, quality: {search_quality}")
        
        return env


class ClarificationAgent(BaseAgent):
    """Compose polite, comprehensive, localized clarification messages.
    
    Responsibilities:
    - Compose clarification messages: Transform agent uncertainty into user-friendly requests
    - Context-aware messaging: Use conversation history and requesting agent context
    - Localization: Translate clarification messages to user's preferred language
    - Professional tone: Maintain appropriate tone for the documentation domain
    - Actionable guidance: Provide specific suggestions on how user can clarify their request
    
    Role: "clarification" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Process clarification message composition request.
        
        Reads: needs_clarification, clarification_reason, clarification_message, language, stage, conversation history
        Writes: response, clarification_type, suggested_actions
        """
        # Extract input attributes
        needs_clarification = env.payload.get("needs_clarification", False)
        clarification_reason = env.payload.get("clarification_reason", "")
        clarification_message = env.payload.get("clarification_message", "")

        if not needs_clarification:
            await self.log(env.conversation_id, "No clarification needed - skipping")
            return env
        
        # Format clarification reason with additional context if available
        reason_text = clarification_reason
        if clarification_message:
            reason_text += f" ({clarification_message})"
            
        await self.log(env.conversation_id, f"Clarification requested: {reason_text}")
        
        memory = await self.get_memory(env.conversation_id)
        history = await memory.get_history(limit=config.ASSISTANT_HISTORY_LIMIT, normalized=False)
        # Get language from payload (set by LangAgent) or default to 'en'
        language = config.SUPPORTED_LANGUAGES.get(env.payload.get("language", "en"), "English")
        text = env.payload.get("canonical_question", env.payload.get("text_eng", env.payload.get("text", "")))
        prompt_options = {'language': language}
        prompt_options['clarification_reason'] = reason_text
        await self.log(env.conversation_id, f"Preparing clarification request in {language} for text: {text}")
        env.payload["response"] = await llm.generate_text_async('clarify', text, history, prompt_options=prompt_options)

        #Flush need_clarification flag to avoid loop
        env.payload["needs_clarification"]= False
        env.payload["clarification_reason"] = ""
        env.payload["clarification_message"] = ""
        #reason_templates = {
        #    "insufficient_context": "I need more context to understand your request better.",
        #    "missing_details": "Could you provide more specific details about what you're looking for?",
        #    "out_of_scope": "This question appears to be outside our documentation scope. Could you rephrase it to focus on our documented features?",
        #    "quality_insufficient": "I couldn't provide a complete answer with the available information. Could you be more specific?",
        #    "ambiguous_query": "Your question could be interpreted in multiple ways. Could you clarify what specifically you're asking about?"
        #}
        
        return env


class AugmentationAgent(BaseAgent):
    """Craft enhanced LLM prompts with search context.
    
    Responsibilities:
    - Build enhanced prompts with documentation context
    - Include source attribution from source URLs
    - Add breadcrumb navigation context from crumbs
    - Structure context for optimal LLM performance
    - Ensure all necessary context is included
    
    Role: "augmentation" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Process prompt augmentation request.
        
        Reads: search_results, search_context, canonical_question, language, text
        Writes: augmented_prompt, source_references, context_structure
        """
        # Extract input attributes
        search_results = env.payload.get("search_results", [])
        search_context = env.payload.get("search_context", "")
        canonical_question = env.payload.get("canonical_question", "")
        language = env.payload.get("language", "en")
        text = env.payload.get("text", canonical_question)  # Original user query
        
        await self.log(env.conversation_id, f"Prompt augmentation: {len(search_results)} results for '{canonical_question[:50]}...' in {language}")
        
        # TODO: Implement actual prompt augmentation logic
        # For now, create basic augmented prompt
        context_parts = []
        source_refs = []
        
        for result in search_results:
            context_parts.append(result.get("content", ""))
            source_refs.append({
                "url": result.get("source", ""),
                "crumbs": result.get("metadata", {}).get("crumbs", [])
            })
        
        env.payload["augmented_prompt"] = f"""
Context: {' '.join(context_parts)}

User Question: {text}

Please provide a comprehensive answer based on the provided context.
"""
        env.payload["source_references"] = source_refs
        env.payload["context_structure"] = {
            "context_length": len(' '.join(context_parts)),
            "source_count": len(source_refs),
            "language": language
        }
        
        await self.log(env.conversation_id, f"Augmented prompt created: {env.payload['context_structure']['context_length']} chars, {len(source_refs)} sources")
        
        return env


class ResponseAgent(BaseAgent):
    """Generate final response using flagship LLM with constraints.
    
    Responsibilities:
    - Generate responses using flagship LLM (GPT-4, Gemini, etc.)
    - Detect uncertainty when documentation context is insufficient for user's specific question
    - Enforce strict documentation-only constraints
    - Match user's preferred language
    - Include source references and navigation hints
    - Format responses appropriately for the domain
    
    Role: "response" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Process response generation request.
        
        Reads: augmented_prompt, language, canonical_question, conversation history from memory
        Writes: response, sources_used, generation_confidence, generation_error (if any)
        + Universal clarification attributes (when documentation insufficient)
        """
        memory = await self.get_memory(env.conversation_id)
        history = await memory.get_history(limit=config.ASSISTANT_HISTORY_LIMIT, normalized=True, show_timestamps=True)
        # Get language from payload (set by LangAgent) or default to 'en'
        language = config.SUPPORTED_LANGUAGES.get(env.payload.get("language", "en"), "English")
        text = env.payload.get("canonical_question", env.payload.get("text_eng", env.payload.get("text", "")))
        prompt_options = {'language': language, 'current_time': util.get_current_time()}
        await self.log(env.conversation_id, f"Preparing response in {language} for text: {text}")
        env.payload["response"] = await llm.generate_text_async('sample', text, history, prompt_options=prompt_options)

        return env

        #keep this code for reference       
        # Extract input attributes
        augmented_prompt = env.payload.get("augmented_prompt", "")
        language = env.payload.get("language", "English")
        canonical_question = env.payload.get("canonical_question", "")
        
        await self.log(env.conversation_id, f"Response generation: prompt_length={len(augmented_prompt)} language={language}")
        
        if len(augmented_prompt.strip()) < 50:
            # Request clarification when documentation context is insufficient
            env.payload["needs_clarification"] = True
            env.payload["clarification_reason"] = "insufficient_context"
            env.payload["clarification_message"] = "The available documentation doesn't contain enough information to answer your specific question"
            await self.log(env.conversation_id, "Requesting clarification: insufficient documentation context")
        else:
            env.payload["response"] = f"This is a placeholder response for: {canonical_question}"
            env.payload["sources_used"] = env.payload.get("source_references", [])
            env.payload["generation_confidence"] = 0.8
            await self.log(env.conversation_id, f"Response generated: {len(env.payload['response'])} chars, confidence: {env.payload['generation_confidence']}")
        
        return env


class QualityAgent(BaseAgent):
    """Final response validation with return capability.
    
    Responsibilities:
    - Validate response accuracy against source documents
    - Check for information outside documentation scope
    - Detect potential hallucinations or knowledge leakage
    - Make three-way routing decisions: approve, pipeline retry, or clarification
    - Ensure response completeness and clarity
    
    Role: "quality" (auto-derived from class name)
    """
    
    async def process(self, env: Envelope) -> Envelope:
        """Process quality validation request.
        
        Reads: response, search_context, canonical_question, source_references
        Writes: quality_score, quality_decision, quality_analysis, validation_result
        + Universal clarification attributes (when insufficient user details)
        """
        # Extract input attributes
        response = env.payload.get("response", "")
        search_context = env.payload.get("search_context", "")
        canonical_question = env.payload.get("canonical_question", "")
        source_references = env.payload.get("source_references", [])
        
        await self.log(env.conversation_id, f"Quality validation: response_length={len(response)} sources={len(source_references)}")
        
        # TODO: Implement actual quality validation logic with three-way routing
        # For now, use simple heuristics
        quality_score = 0.85 if len(response) > 50 and len(source_references) > 0 else 0.6
        
        env.payload["quality_score"] = quality_score
        env.payload["quality_analysis"] = f"Response quality assessment: score={quality_score}, length={len(response)}, sources={len(source_references)}"
        env.payload["validation_result"] = {
            "accuracy_check": "passed" if quality_score > 0.7 else "failed",
            "scope_compliance": "within_scope",
            "completeness": "adequate" if len(response) > 50 else "insufficient"
        }
        
        # Three-way routing decision
        if quality_score > 0.8:
            env.payload["quality_decision"] = "approve"
            await self.log(env.conversation_id, f"Quality approved: score={quality_score}")
        elif quality_score > 0.5:
            # Pipeline issue - let ManagerAgent handle retry
            env.payload["quality_decision"] = "regenerate"  # or "research"
            await self.log(env.conversation_id, f"Quality retry needed: score={quality_score}")
        else:
            # Insufficient user details - request clarification
            env.payload["quality_decision"] = "approve"  # Don't block, let clarification handle it
            env.payload["needs_clarification"] = True
            env.payload["clarification_reason"] = "quality_insufficient"
            env.payload["clarification_message"] = "The response quality is poor, likely due to insufficient details in your question"
            await self.log(env.conversation_id, f"Requesting clarification: poor quality due to insufficient user details")
        
        return env
    


# Create default agent instances - they will auto-register via BaseAgent.__init__
# Single instances (fast, non-LLM agents)
_manager = ManagerAgent()          # 1 instance - routing logic only
_command = CommandAgent()          # 1 instance - simple command processing  
_lang = LangAgent()               # 1 instance - fast language detection

# Multiple instances (LLM-backed agents to prevent blocking)
_translate1 = TranslationAgent()   # 2 instances - LLM translation services
_translate2 = TranslationAgent()
_essence1 = EssenceAgent()        # 2 instances - LLM context extraction
_essence2 = EssenceAgent()
_guardrails1 = GuardrailsAgent()  # 2 instances - LLM scope analysis
_guardrails2 = GuardrailsAgent()
_search1 = SearchAgent()          # 2 instances - semantic search + LLM ranking
_search2 = SearchAgent()
_clarification1 = ClarificationAgent()  # 2 instances - LLM ambiguity detection
_clarification2 = ClarificationAgent()
_augmentation1 = AugmentationAgent()    # 2 instances - LLM prompt engineering
_augmentation2 = AugmentationAgent()
_response1 = ResponseAgent()      # 2 instances - flagship LLM generation
_response2 = ResponseAgent()
_quality1 = QualityAgent()        # 2 instances - LLM validation
_quality2 = QualityAgent()

# Direct function implementations (no proxy class needed)

async def process_user_message(user_id: str, message: str) -> Dict[str, Any]:
    """Process a user message through the agent pipeline.
    
    Args:
        user_id: User identifier (used as conversation_id)
        message: User message text to process
        
    Returns:
        Dict containing the processed message response
    """
    logger.info(f"Processing message from user {user_id}: {message}")
    response = dict()
    payload = {"text": message, "stage": "start"}
    response["message"] = await process_request('manager', user_id, payload)
    return response

async def clear_conversation_history(user_id: str) -> bool:
    """Clear conversation history for a user while keeping other memory intact.
    
    Args:
        user_id: User identifier
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use the distributed memory system to clear message history
        result = await process_request("command", user_id, {
            "action": "clear_history",
            "user_id": user_id
        })
        
        if "error" not in result:
            logger.info(f"Cleared conversation history for user {user_id}")
            return True
        else:
            logger.error(f"Failed to clear history for user {user_id}: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error clearing conversation history for user {user_id}: {e}")
        return False

async def clear_all_conversation_data(user_id: str) -> bool:
    """Clear ALL conversation data for a user (complete cleanup).
    
    Args:
        user_id: User identifier
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Use the distributed memory system to clear all conversation data
        result = await process_request("command", user_id, {
            "action": "clear_all",
            "user_id": user_id
        })
        
        if "error" not in result:
            logger.info(f"Cleared all conversation data for user {user_id}")
            return True
        else:
            logger.error(f"Failed to clear all data for user {user_id}: {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error clearing all conversation data for user {user_id}: {e}")
        return False

async def set_user_language(user_id: str, language: str | None) -> bool:
    try:
        if language is None or language in config.SUPPORTED_LANGUAGES:
            #language = None means reset to default language
            result = await process_request("command", user_id, {"action": "set_language", "language": language})
            # process_request returns a string, check if it indicates success
            success = "successfully" in result.lower() or "reset" in result.lower()
            if success:
                logger.info(f"Language set for user {user_id}: {result}")
            return success
        else:
            logger.error(f"Unsupported language: {language} for user {user_id}")
            return False
    except Exception as e:
        logger.error(f"Error setting user language for user {user_id}: {e}")
        return False

if __name__ == "__main__":
    async def _demo():
        try:
            print("=== Memory Cleanup Demo ===")
            
            # Send some messages to build conversation history
            print("\n1. Building conversation history...")
            for i in range(3):
                message = f"Hello message {i+1}! [{datetime.now().isoformat()}]"
                res = await process_user_message("test_user", message)
                print(f"Message {i+1} result:", res.get("message", "No response"))
                await asyncio.sleep(0.5)
            
            # Test partial cleanup (messages only)
            print("\n2. Testing partial cleanup (messages only)...")
            cleanup_result = await clear_conversation_history("test_user")
            print(f"   Cleanup successful: {cleanup_result}")
            
            # Send one more message to verify system still works
            print("\n3. Sending message after cleanup...")
            new_message = f"New message after cleanup! [{datetime.now().isoformat()}]"
            res = await process_user_message("test_user", new_message)
            print("New message result:", res.get("message", "No response"))
            
            print("\n=== Demo Complete ===")

            # Example: global graceful shutdown (agents finish current tasks)
            await broadcast_command("shutdown")
        finally:
            #await stop_runtime()
            pass

    asyncio.run(_demo())