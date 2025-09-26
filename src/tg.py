import config
import logging
import asyncio
import re
import html
import time
import signal
from telegram import Update, InputMediaPhoto
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import NetworkError, TimedOut, RetryAfter, BadRequest
import assistant
import atexit
import os
import json
import sys
from datetime import datetime


logging.getLogger('apscheduler').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.WARNING)
logging.getLogger('aiomysql').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

telegram_users_file = os.path.join(config.DATA_DIR, "users.json")
telegram_users = {}

if os.path.exists(telegram_users_file):
    with open(telegram_users_file, "r") as f:
        telegram_users = json.load(f)
else:
    telegram_users = {}
    logger.info("Telegram users file not found, creating new file.")

def save_telegram_users():
    logger.info(f"Saving telegram users to file {telegram_users_file}")
    with open(telegram_users_file, "w") as f:
        json.dump(telegram_users, f)

# Global shutdown event for coordinated shutdown
_global_shutdown_event = None

def get_global_shutdown_event():
    """Get or create the global shutdown event."""
    global _global_shutdown_event
    if _global_shutdown_event is None:
        _global_shutdown_event = asyncio.Event()
    return _global_shutdown_event

def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) and SIGTERM signals."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown of Telegram bot")
    
    # Set the global shutdown event to stop the bot
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            shutdown_event = get_global_shutdown_event()
            loop.call_soon_threadsafe(shutdown_event.set)
    except RuntimeError:
        # No event loop available, that's OK during shutdown
        pass

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

atexit.register(save_telegram_users)


class TelegramBot:
    """Simple Telegram bot that processes user messages through the Alaris Assistant."""
    
    def __init__(self, token: str):
        self.token = token
        self.application = Application.builder().token(token).build()
        self._setup_handlers()
        self.max_retries = 5
        self.base_delay = 1.0  # Base delay for exponential backoff
    
    async def _retry_with_backoff(self, operation, *args, **kwargs):
        """Execute operation with exponential backoff retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await operation(*args, **kwargs)
            except (NetworkError, TimedOut) as e:
                last_exception = e
                if attempt == self.max_retries - 1:
                    logger.error(f"Network operation failed after {self.max_retries} attempts: {str(e)}")
                    raise
                
                # Calculate exponential backoff delay
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"Network error on attempt {attempt + 1}/{self.max_retries}: {str(e)}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
            except RetryAfter as e:
                # Telegram rate limiting - respect the retry_after value
                logger.warning(f"Rate limited by Telegram. Waiting {e.retry_after}s...")
                await asyncio.sleep(e.retry_after)
                # Don't count rate limits as retry attempts
                continue
            except BadRequest as e:
                # Don't retry bad requests - they won't succeed
                logger.error(f"Bad request error: {str(e)}")
                raise
            except Exception as e:
                # For other exceptions, log and re-raise immediately
                logger.error(f"Unexpected error in network operation: {str(e)}")
                raise
        
        # This should never be reached, but just in case
        raise last_exception

    def _setup_handlers(self):
        """Set up message and command handlers."""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("restart", self.stop_command))
        self.application.add_handler(CommandHandler("stop", self.stop_command))
        self.application.add_handler(CommandHandler("reset", self.stop_command))
        self.application.add_handler(CommandHandler("reboot", self.stop_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))

        # Message handler for all text messages
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Clear conversation history for the user."""
        user_id = str(update.effective_user.id)
        logger.info(f"Clear command received from user {user_id}")
        
        if user_id not in telegram_users:
            await self._retry_with_backoff(
                context.bot.send_message,
                chat_id=update.effective_chat.id, 
                text="You are not authorized to use this bot. Please contact the administrator."
            ) 
            return
        
        # Clear conversation history for this user
        cleared = await assistant.clear_conversation_history(user_id)
        
        if cleared:
            await self._retry_with_backoff(
                update.message.reply_text,
                "✅ Conversation history cleared. You can start a fresh conversation now."
            )
        else:
            await self._retry_with_backoff(
                update.message.reply_text,
                "ℹ️ No conversation history found to clear."
            )

    async def stop_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop the bot gracefully for wrapper restart."""
        user_id = str(update.effective_user.id)
        logger.info(f"Stop command received from user {user_id}")
        
        if user_id not in telegram_users:
            await self._retry_with_backoff(
                context.bot.send_message,
                chat_id=update.effective_chat.id, 
                text="You are not authorized to use this bot. Please contact the administrator."
            ) 
            return
        
        await self._retry_with_backoff(
            update.message.reply_text,
            "🔄 Bot is stopping for restart..."
        )
        logger.info("Bot stopping via /stop command")
        
        # Save users before exit
        save_telegram_users()
        
        # Set the global shutdown event to break the main loop
        shutdown_event = get_global_shutdown_event()
        shutdown_event.set()
        
        # Give a moment for the message to be sent
        await asyncio.sleep(0.5)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = str(update.effective_user.id)
        if user_id not in telegram_users:
            text = update.message.text
            if text.startswith("/start"):
                text = text[6:].strip()
                if text and text.lower() in [config.TELEGRAM_INVITE_CODE.lower()]:
                    telegram_users[user_id] = {
                        "first_name": update.effective_user.first_name,
                        "last_name": update.effective_user.last_name,
                        "username": update.effective_user.username,
                        "created_at": datetime.now().isoformat()
                    }
                    save_telegram_users()
                    logger.info(f"New user added: {user_id}")
                    await self._retry_with_backoff(
                        update.message.reply_text,
                        "You have been added to the list of authorized users."
                    )
                else:
                    logger.info(f"User {user_id} is not authorized to use this bot.")
                    await self._retry_with_backoff(
                        update.message.reply_text,
                        "You are not authorized to use this bot. Please contact the administrator."
                    ) 
                    return

        logger.debug(update)

        """Handle /start command."""
        welcome_message = (
            "🤖 Welcome to Alaris SMS Platform Assistant!\n\n"
            "I can help you with questions about the Alaris telecom billing and routing platform.\n"
            "Just send me your question and I'll do my best to help!"
        )
        await self._retry_with_backoff(
            update.message.reply_text,
            welcome_message
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_message = (
            f"🤖 Alaris SMS Platform Assistant v{config.VERSION} Help\n\n"
            "I can answer questions about:\n"
            "• Platform overview & UI navigation\n"
            "• Administration & System settings\n"
            "• Users/Roles/Permissions management\n"
            "• Messaging channels & APIs\n"
            "• Routing rules and features\n"
            "• Rates & Pricing\n"
            "• Billing & Finance\n"
            "• Analytics & Reports\n"
            "• And much more!\n\n"
            "Just type your question and I'll help you find the answer."
        )
        await self._retry_with_backoff(
            update.message.reply_text,
            help_message
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text messages."""
        try:
            user_id = str(update.effective_user.id)
            message_text = update.message.text
            
            logger.info(f"Received message from user {user_id}: {message_text}")

            if user_id not in telegram_users:
                await self._retry_with_backoff(
                    context.bot.send_message,
                    chat_id=update.effective_chat.id, 
                    text="You are not authorized to use this bot. Please contact the administrator."
                ) 
                return
            
            # Send immediate "Thinking..." response with retry logic
            thinking_message = await self._retry_with_backoff(
                update.message.reply_text,
                "🤔 Thinking..."
            )
            
            # 🚀 NON-BLOCKING: Process message in background task
            asyncio.create_task(self._process_message_async(user_id, message_text, thinking_message))
            
        except Exception as e:
            logger.error(f"Error in handle_message: {str(e)}")
            try:
                await self._retry_with_backoff(
                    update.message.reply_text,
                    "❌ Sorry, I encountered an error processing your message."
                )
            except Exception as reply_error:
                logger.error(f"Failed to send error message: {str(reply_error)}")
    
    async def _process_message_async(self, user_id: str, message_text: str, thinking_message):
        """Process message asynchronously and update the thinking message."""
        try:
            response = await assistant.process_user_message(user_id, message_text)
            message = response.get("message", None)
            if not message:
                message = "I'm sorry, but I couldn't generate a response. Please try again later."
            image_urls = None

            # Replace "Thinking..." message with text response
            await self._send_response_message(thinking_message, message, image_urls, user_id)
            
        except Exception as e:
            logger.error(f"Error in _process_message_async: {str(e)}")
            # Try to edit the thinking message with error
            try:
                await self._retry_with_backoff(
                    thinking_message.edit_text,
                    "I apologize, but I encountered an error while processing your message. Please try again."
                )
            except Exception as edit_error:
                logger.error(f"Error editing thinking message with error: {str(edit_error)}")
    
    async def _send_response_message(self, thinking_message, message, image_urls, user_id):
        """Send the response message, handling long messages and images."""
        message_sent = False
        try:
            # Telegram message limit is 4096 characters
            MAX_MESSAGE_LENGTH = 4000  # Leave some buffer for HTML tags
            

            if len(message) <= MAX_MESSAGE_LENGTH:
                # Message fits in one piece
                await self._retry_with_backoff(
                    thinking_message.edit_text,
                    message,
                    parse_mode='HTML'
                )
                message_sent = True
            else:
                # Message is too long, need to split
                await self._retry_with_backoff(thinking_message.delete)
                
                # Split message into chunks
                chunks = []
                current_chunk = ""
                
                # Split by paragraphs first
                paragraphs = message.split('\n\n')
                
                for paragraph in paragraphs:
                    if len(current_chunk + paragraph + '\n\n') <= MAX_MESSAGE_LENGTH:
                        current_chunk += paragraph + '\n\n'
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = paragraph + '\n\n'
                        else:
                            # Single paragraph is too long, split by sentences
                            sentences = paragraph.split('. ')
                            for i, sentence in enumerate(sentences):
                                sentence_with_period = sentence + ('. ' if i < len(sentences) - 1 else '')
                                if len(current_chunk + sentence_with_period) <= MAX_MESSAGE_LENGTH:
                                    current_chunk += sentence_with_period
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk.strip())
                                    current_chunk = sentence_with_period
                            current_chunk += '\n\n'
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Send chunks as separate messages with retry logic
                for i, chunk in enumerate(chunks):
                    chunk_text = f"📄 Part {i+1} of {len(chunks)}:\n\n{chunk}"
                    
                    await self._retry_with_backoff(
                        thinking_message.get_bot().send_message,
                        chat_id=thinking_message.chat_id, 
                        text=chunk_text, 
                        parse_mode='HTML'
                    )
                message_sent = True
                    
        except Exception as html_error:
            logger.error(f"HTML parsing error: {str(html_error)}")
            # Fallback: send as plain text without HTML formatting
            plain_message = message.replace('<a href="', '').replace('">', ' (').replace('</a>', ')')
            
            try:
                # Check length for plain text too
                if len(plain_message) <= MAX_MESSAGE_LENGTH:
                    if not message_sent:
                        await self._retry_with_backoff(
                            thinking_message.edit_text,
                            plain_message
                        )
                    else:
                        await self._retry_with_backoff(
                            thinking_message.get_bot().send_message,
                            chat_id=thinking_message.chat_id, 
                            text=plain_message
                        )
                else:
                    if not message_sent:
                        await self._retry_with_backoff(thinking_message.delete)
                    await self._retry_with_backoff(
                        thinking_message.get_bot().send_message,
                        chat_id=thinking_message.chat_id, 
                        text=plain_message[:MAX_MESSAGE_LENGTH] + "...\n\n[The message was truncated due to length limit]"
                    )
            except Exception as fallback_error:
                logger.error(f"Fallback error: {str(fallback_error)}")
                # Last resort: send a simple error message with retry
                try:
                    await self._retry_with_backoff(
                        thinking_message.get_bot().send_message,
                        chat_id=thinking_message.chat_id, 
                        text="Sorry, an error occurred while sending the message. Please try again."
                    )
                except Exception as final_error:
                    logger.error(f"Final fallback failed: {str(final_error)}")
                    pass  # Give up gracefully
        
        # If we have images, send them as a separate media group without captions
        if image_urls:
            try:
                # Prepare media group with images (no captions)
                media = [InputMediaPhoto(media=url) for url in image_urls[:10]]  # Telegram limit: max 10 images
                
                # Send media group separately with retry logic
                await self._retry_with_backoff(
                    thinking_message.get_bot().send_media_group,
                    chat_id=thinking_message.chat_id,
                    media=media
                )
                
            except Exception as img_error:
                logger.error(f"Error sending images: {str(img_error)}")
                # Images failed to send, but text was already sent successfully
        
        logger.info(f"Sent response to user {user_id}: {message[:100]}...")
    
    async def run(self):
        """Start the bot with automatic reconnection."""
        logger.info("Starting Telegram bot...")
        
        shutdown_event = get_global_shutdown_event()
        
        while not shutdown_event.is_set():
            try:
                await self.application.initialize()
                await self.application.start()
                await self.application.updater.start_polling()
                
                logger.info("Bot is running. Waiting for shutdown event...")
                
                # Keep the bot running until shutdown event is set
                await shutdown_event.wait()
                
                logger.info("Shutdown event received. Stopping bot...")
                break
                
            except (NetworkError, TimedOut) as e:
                logger.error(f"Network error in main bot loop: {str(e)}")
                logger.info("Attempting to restart bot in 30 seconds...")
                
                # Clean up current instance
                try:
                    await self.application.updater.stop()
                    await self.application.stop()
                    await self.application.shutdown()
                except Exception as cleanup_error:
                    logger.error(f"Error during cleanup: {str(cleanup_error)}")
                
                # Wait before restarting, but check for shutdown
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=30)
                    logger.info("Shutdown event received during network error recovery. Exiting...")
                    break
                except asyncio.TimeoutError:
                    # Timeout is expected - continue with restart
                    pass
                
                # Recreate application instance
                self.application = Application.builder().token(self.token).build()
                self._setup_handlers()
                
                logger.info("Restarting bot after network error...")
                continue
                
            except Exception as e:
                logger.error(f"Unexpected error in bot main loop: {str(e)}")
                logger.info("Attempting to restart bot in 60 seconds...")
                
                # Clean up and wait longer for unexpected errors
                try:
                    await self.application.updater.stop()
                    await self.application.stop()
                    await self.application.shutdown()
                except Exception as cleanup_error:
                    logger.error(f"Error during cleanup: {str(cleanup_error)}")
                
                # Wait before restarting, but check for shutdown
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=60)
                    logger.info("Shutdown event received during unexpected error recovery. Exiting...")
                    break
                except asyncio.TimeoutError:
                    # Timeout is expected - continue with restart
                    pass
                
                # Recreate application instance
                self.application = Application.builder().token(self.token).build()
                self._setup_handlers()
                
                logger.info("Restarting bot after unexpected error...")
                continue
        
        # Final cleanup
        try:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
            logger.info("Bot stopped successfully.")
        except Exception as final_cleanup_error:
            logger.error(f"Error during final cleanup: {str(final_cleanup_error)}")

async def main():
    """Main function to run the bot."""
    if not config.TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found in configuration. Please set it in your .env file.")
        return
    
    bot = TelegramBot(config.TELEGRAM_BOT_TOKEN)
    await bot.run()

if __name__ == '__main__':
    asyncio.run(main())