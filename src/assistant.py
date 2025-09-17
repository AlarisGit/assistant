import logging
from typing import Dict, Any

import config
import agent

logger = logging.getLogger(__name__)

class AlarisAssistant:
    def __init__(self):
        self.conversations = {}  # In-memory conversation storage
        
    async def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        logger.info(f"Processing message from user {user_id}: {message}")
        response = dict()
        response["message"] = await agent.process_request(user_id, message)
        return response
   
    async def clear_conversation_history(self, user_id: str) -> bool:
        if user_id and user_id in self.conversations:
            del self.conversations[user_id]
            logger.info(f"Cleared conversation history for user {user_id}")
            return True
        
        logger.info(f"No conversation history found for user {user_id}")
        return False

assistant = AlarisAssistant()

async def clear_conversation_history(user_id: str) -> bool:
    global assistant
    return await assistant.clear_conversation_history(user_id)

async def process_user_message(user_id: str, message: str) -> Dict[str, Any]:
    global assistant
    return await assistant.process_message(user_id, message)

