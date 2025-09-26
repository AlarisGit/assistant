"""
Stream Cleanup Manager - Simplified version for Redis stream maintenance
"""

import asyncio
import logging
import time
from typing import List, Optional
import redis.asyncio as redis
import config

logger = logging.getLogger(__name__)


class StreamCleanupManager:
    """Simplified manager for cleaning up expired messages from Redis streams."""
    
    def __init__(self, cleanup_interval: int = 60, message_ttl: int = 60):
        """Initialize the cleanup manager.
        
        Args:
            cleanup_interval: How often to run cleanup (seconds)
            message_ttl: Message time-to-live (seconds)
        """
        self.cleanup_interval = cleanup_interval
        self.message_ttl = message_ttl
        self.redis: Optional[redis.Redis] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
    async def start(self):
        """Start the cleanup manager."""
        if self._cleanup_task is not None:
            return  # Already running
            
        self.redis = redis.Redis.from_url(config.REDIS_URL, decode_responses=False)
        self._stop_event.clear()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Stream cleanup manager started")
        
    async def stop(self):
        """Stop the cleanup manager."""
        if self._cleanup_task is None:
            return
            
        self._stop_event.set()
        try:
            await asyncio.wait_for(self._cleanup_task, timeout=5.0)
        except asyncio.TimeoutError:
            self._cleanup_task.cancel()
            
        if self.redis:
            await self.redis.aclose()
            
        self._cleanup_task = None
        logger.info("Stream cleanup manager stopped")
        
    async def _cleanup_loop(self):
        """Main cleanup loop."""
        while not self._stop_event.is_set():
            try:
                await self._cleanup_expired_messages()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(self.cleanup_interval)
                
    async def _cleanup_expired_messages(self):
        """Clean up expired messages from all streams."""
        try:
            # Get all stream keys
            stream_keys = await self.redis.keys('stream:*')
            if not stream_keys:
                return
                
            current_time_ms = int(time.time() * 1000)
            expiration_time_ms = current_time_ms - (self.message_ttl * 1000)
            
            total_cleaned = 0
            streams_processed = 0
            
            for stream_key in stream_keys:
                try:
                    stream_name = stream_key.decode('utf-8') if isinstance(stream_key, bytes) else stream_key
                    cleaned = await self._cleanup_stream(stream_name, expiration_time_ms)
                    total_cleaned += cleaned
                    if cleaned > 0:
                        streams_processed += 1
                except Exception as e:
                    logger.debug(f"Error cleaning stream {stream_key}: {e}")
                    
            if total_cleaned > 0:
                logger.warning(f"🧹 Cleaned {total_cleaned} expired messages from {streams_processed} streams")
            else:
                logger.debug(f"✅ Stream cleanup: {len(stream_keys)} streams checked, no expired messages")
                
        except Exception as e:
            logger.error(f"Error during stream cleanup: {e}")
            
    async def _cleanup_stream(self, stream_name: str, expiration_time_ms: int) -> int:
        """Clean up expired messages from a specific stream."""
        try:
            # Check if stream exists and has messages
            try:
                stream_info = await self.redis.xinfo_stream(stream_name)
                if stream_info.get('length', 0) == 0:
                    return 0
            except Exception:
                return 0
                
            # Find expired messages
            messages_to_delete = []
            batch_size = 100
            last_id = '0-0'
            
            while True:
                try:
                    messages = await self.redis.xrange(
                        stream_name, 
                        min=last_id, 
                        max='+', 
                        count=batch_size
                    )
                    
                    if not messages:
                        break
                        
                    found_non_expired = False
                    for msg_id, fields in messages:
                        msg_timestamp_ms = int(msg_id.split(b'-')[0] if isinstance(msg_id, bytes) else msg_id.split('-')[0])
                        
                        if msg_timestamp_ms < expiration_time_ms:
                            messages_to_delete.append(msg_id)
                        else:
                            found_non_expired = True
                            break
                            
                    if found_non_expired or len(messages) < batch_size:
                        break
                        
                    if messages:
                        last_msg_id = messages[-1][0]
                        last_id = last_msg_id.decode('utf-8') if isinstance(last_msg_id, bytes) else last_msg_id
                        
                except Exception as e:
                    logger.debug(f"Error reading messages from {stream_name}: {e}")
                    break
                    
            # Delete expired messages
            if messages_to_delete:
                try:
                    deleted = await self.redis.xdel(stream_name, *messages_to_delete)
                    return deleted
                except Exception as e:
                    logger.debug(f"Error deleting expired messages from {stream_name}: {e}")
                    
            return 0
            
        except Exception as e:
            logger.debug(f"Error cleaning stream {stream_name}: {e}")
            return 0


# Global instance
_cleanup_manager: Optional[StreamCleanupManager] = None


async def get_cleanup_manager() -> StreamCleanupManager:
    """Get or create the global cleanup manager."""
    global _cleanup_manager
    if _cleanup_manager is None:
        _cleanup_manager = StreamCleanupManager()
    return _cleanup_manager


async def start_cleanup_manager():
    """Start the global cleanup manager."""
    manager = await get_cleanup_manager()
    await manager.start()


async def stop_cleanup_manager():
    """Stop the global cleanup manager."""
    global _cleanup_manager
    if _cleanup_manager:
        await _cleanup_manager.stop()
        _cleanup_manager = None
