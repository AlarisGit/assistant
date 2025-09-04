import config
import logging
import os
import json
from typing import List, Dict, Any, Optional
import random
import hashlib
import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from llm import get_embedding

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_qdrant_url = config.QDRANT_URL
_qdrant_api_key = config.QDRANT_API_KEY
_qdrant_collection_name = config.QDRANT_COLLECTION_NAME

# Initialize Qdrant client
_qdrant_client = QdrantClient(
    url=_qdrant_url,
    api_key=_qdrant_api_key if _qdrant_api_key else None
)

def store_chunk(chunk: str, doc_id: str, chunk_id: Optional[str]) -> str:
    if chunk is None or len(chunk) == 0:
        raise ValueError("Chunk cannot be None or empty")
    
    embedding = get_embedding(chunk)
    
    payload = {
        'chunk': chunk,
        'doc_id': doc_id,
        'chunk_id': chunk_id
    }
    
    if chunk_id is not None:
        point_id = chunk_id
        full_chunk_id = doc_id + '/' + chunk_id
    else:
        full_chunk_id = doc_id

    h = hashlib.sha256(full_chunk_id.encode()).hexdigest()[:32]
    # Format as UUID: 8-4-4-4-12 pattern
    point_id = f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
    
    
    # Create point structure for Qdrant
    point = PointStruct(
        id=point_id,
        vector=embedding,
        payload=payload
    )
    
    try:
        # Ensure collection exists
        _ensure_collection_exists()
        
        # Store the point in Qdrant
        _qdrant_client.upsert(
            collection_name=_qdrant_collection_name,
            points=[point]
        )
        
        logger.info(f"Stored chunk with point_id: {point_id}")
        return point_id
        
    except Exception as e:
        logger.error(f"Failed to store chunk in Qdrant: {str(e)}")
        raise

def _ensure_collection_exists():
    """Ensure the Qdrant collection exists with proper configuration."""
    try:
        collections = _qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if _qdrant_collection_name not in collection_names:
            logger.info(f"Creating Qdrant collection: {_qdrant_collection_name}")
            _qdrant_client.create_collection(
                collection_name=_qdrant_collection_name,
                vectors_config=VectorParams(
                    size=config.EMB_DIM,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Collection {_qdrant_collection_name} created successfully")
        else:
            logger.debug(f"Collection {_qdrant_collection_name} already exists")
            
    except Exception as e:
        logger.error(f"Failed to ensure collection exists: {str(e)}")
        raise

def search_chunks(query: str, k: int = 20, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Search for similar chunks in Qdrant.
    
    Args:
        query: The search query text
        k: Number of results to return
        threshold: Minimum similarity score threshold
        
    Returns:
        List of matching chunks with their payloads and scores
    """
    try:
        query_embedding = get_embedding(query)
        
        search_result = _qdrant_client.search(
            collection_name=_qdrant_collection_name,
            query_vector=query_embedding,
            limit=k,
            score_threshold=threshold
        )
        
        results = []
        for point in search_result:
            result = {
                'id': point.id,
                'score': point.score,
                'payload': point.payload
            }
            results.append(result)
            
        logger.info(f"Found {len(results)} matching chunks for query")
        return results
        
    except Exception as e:
        logger.error(f"Failed to search chunks in Qdrant: {str(e)}")
        return []

if __name__ == '__main__':
    chunks = [
        "# Alaris SMS HTTP API\n\nRequest format:\n\nGET request to HTTP port: [http://1.1.1.1:8001/api?username=<](http://1.1.1.1:8001/api?username=%3C)username>&password=<password>&ani=<ani>&dnis=<dnis>&message=<message>&command=submit&serviceType=<serviceType>&longMessageMode=<longMessageMode>\n\n   \nGET request to HTTPs port: [https://domainname:8002/api?username=<](https://domainname:8002/api?username=%3C)username>&password=<password>&ani=<ani>&dnis=<dnis>&message=<message>&command=submit&serviceType=<serviceType>&longMessageMode=<longMessageMode>\n\n   \nPOST request (credentials are part of the link) to HTTP port: curl -H 'Content-Type: application/json' -X POST -d '{\"ani\":\"ani\",\"dnis\":\"dnis\",\"message\":\"test\"}' '[http://1.1.1.1:8001/api?command=submit&username=username&password=password'](http://1.1.1.1:8001/api?command=submit&username=username&password=password%27)\n\n   \nPOST request (credentials are a part of the JSON body) to HTTP port: curl -H 'Content-Type: application/json' -X POST -d '{\"username\":\"username\",\"password\":\"password\",\"command\":\"submit\",\"ani\":\"ani\",\"dnis\":\"dnis\",\"message\":\"test\"}' '[http://1.1.1.1:8001/api?'](http://1.1.1.1:8001/api?%27)\n\nNOTE: To send multiple SMS messages, add several comma-separated DNIS to the <dnis> field.\n\n| Parameter | Value | Required |\n| --- | --- | --- |\n| command: | Request type. Possible values: “submit”, \"query\", \"mo\". To send a message over HTTP, specify command=submit | Yes |\n| dnis: | Destination number. Must be sent in international E.164 format (up to 15 digits allowed). If the field length exceeds 2048 symbols or 100 numbers, the incoming message will be rejected with the following code: 400 Bad Request (Destination number is too long).",
        "To send a message over HTTP, specify command=submit | Yes |\n| dnis: | Destination number. Must be sent in international E.164 format (up to 15 digits allowed). If the field length exceeds 2048 symbols or 100 numbers, the incoming message will be rejected with the following code: 400 Bad Request (Destination number is too long). Multiple DNIS's cannot be used for longMessageMode=split\\_or\\_payload     If multiple identical numbers are transferred in the field, the duplicates will not be sent further, even though message IDs will be issued for all successfully routed messages. | Yes |\n| message: | Message text | Yes |\n| password: | Password | Yes |\n| serviceType: | Service type, provided by the System owner for the registered interconnection. Can be blank. The maximum length is 9 bytes. | Yes |\n| username: | Login | Yes |\n| ani: | Caller ID. Technical limitation - alpha-numeric up to 32 symbols. Additional limitations can be caused by destination route peculiarities | No |\n| dataCoding | data coding scheme for sending the message to the vendor. Format: integer. Optional. Allowed values are: 0, 1, 3, 6, 7, 8, 10, 14, where:  0: SMSC Default Alphabet (SMPP 3.4)/MC Specific (SMPP 5.0)  1: IA5 (CCITT T.50)/ASCII (ANSI X3.4)  3: Latin 1 (ISO-8859-1)  6: Cyrillic (ISO-8859-5)  7: Latin/Hebrew (ISO-8859-8)  8: UCS2 (ISO/IEC-10646)  10: ISO-2022-JP (Music Codes)  14: KS C 5601 | No |\n| esmClass | corresponds to the same name parameter in SMPP. Format: integer. Optional. Allowed values are: 0-255 | No |\n| flash | Flag that indicates a flash message. Possible values are: 0 (regular message) and 1 (flash message that is shown on the screen and is not stored in the device memory). Please note that the flag is merely written to the EDR (Technical details field) and does not change the message data coding.",
        "Possible values are: 0 (regular message) and 1 (flash message that is shown on the screen and is not stored in the device memory). Please note that the flag is merely written to the EDR (Technical details field) and does not change the message data coding. The flag value can be transmitted for MT message sending. The flag can also be changed with the help of translation rules. | No |\n| longMessageMode: | Type of long message processing (applicable to incoming MT messages only). The following values allowed:  cut (trim message text to 140 bytes) - shortens the message leaving only first 140 bytes to be sent.  split and split\\_sar - split the message according to the logics described below. The difference between them is in the header to be used, for split it is UDH header, for split\\_sar it is sar accordingly.  single\\_id\\_split - split the message but return the message ID common for all segments  payload - message\\_payload field is used for sending the message text  The splitting (options 2/3) depends on the coding:  \"- dataCoding = 0, 1 or 3: one message can contain up to 160 bytes. If more: segment count = 'message length in symbols / 153 symbols' (or: 'message length in bytes / 153 bytes')  - dataCoding 2, 4 - 7: one message can contain up to 140 bytes, if more – segment count = 'message length in symbols / 134 symbols' (or: 'message length in bytes / 134 bytes')  - dataCoding 8: one message can contain up to 140 bytes, if more – segment count = 'message length in symbols / 67 symbols' (as 1 symbol occupies 2 bytes, that is: 'message length in bytes / 134 bytes')  split\\_or\\_payload: serves for sending long messages received over HTTP to SMPP vendors. The mode is not supported if several numbers have been received in the dnis parameter within a single request from the client. When the Send text in payload option is enabled in the vendor channel, the message will be sent in the payload field as a single submit\\_sm packet. Also, a single delivery report is expected for it, whereas the client will be sent as many reports as the number of received parts.",
        "When the Send text in payload option is enabled in the vendor channel, the message will be sent in the payload field as a single submit\\_sm packet. Also, a single delivery report is expected for it, whereas the client will be sent as many reports as the number of received parts. Besides, a single EDR will be written, therefore, the bill by segments option must be set in the client product for correct billing of the client. The default value is \"cut\". | No |\n| incMsgId (inc\\_msg\\_id) | The client message ID (128 symbols maximum) that can be used for incoming HTTP requests with longMessageMode=cut, longMessageMode=split, longMessageMode=split\\_sar or no longMessageMode (which equals to longMessageMode=cut). When the SMS switch receives this parameter it will use its value as a client ID. This will allow clients to use their ID to request information on the message if, for example, no routes are available. In case of long messages (when longMessageMode=split or longMessageMode=split\\_sar), the incoming parameter inc\\_msg\\_id will be one and the same for the entire message, whereas the SMS switch will split the message into several parts and use the same ID for each part, adding the part number to each of them. For example, if a message contains two parts and has the incoming message ID is inc\\_msg\\_id=1gfc4dd56cbndcj741xs, the SMS switch will process the messages with IDs 1gfc4dd56cbndcj741xs-1 and 1gfc4dd56cbndcj741xs-2. | No |\n| srcTon, srcNpi, dstTon, dstNpi | the respective parameters for Sender ID and Destination number. Format: integer. Optional | No |\n| priorityFlag | corresponds to the same name parameter in SMPP. Format: integer. Optional. Allowed values are: 0 and 1 | No |\n| registeredDelivery | corresponds to the same name parameter in SMPP. Format: integer. Optional. Allowed values are: 0 and 1 | No |\n| replaceIfPresentFlag | corresponds to the same name parameter in SMPP. Format: integer. Optional.",
        "Format: integer. Optional. Allowed values are: 0 and 1 | No |\n| registeredDelivery | corresponds to the same name parameter in SMPP. Format: integer. Optional. Allowed values are: 0 and 1 | No |\n| replaceIfPresentFlag | corresponds to the same name parameter in SMPP. Format: integer. Optional. Allowed values are: 0 and 1 | No |\n| silent | Flag that allows sending silent SMS (message that arrives with no sound and is not displayed on the screen)  Allowed values are: 0 and 1, where 0 means NOT silent. Any value other than 0 which has been set explicitly is treated as true, for example, silent=false is interpreted as silent=1. Whether the silent SMS arrives as a silent one to the end user depends on the vendor and the other carriers that handle it. | No |\n| validityPeriod | validity period in absolute or relative format  Examples: validityPeriod=000002000000000R (relative format - the message is valid for 2 days)  validityPeriod=241222080910000-  (absolute format - the message is valid till Dec 22, 2024, 08:09:10GMT+0) | No |\n\nResponse format\n\nIn case of successful processing, the status in the header of the HTTP response is 200 OK. Response body contains the message\\_id.",
        "Response body contains the message\\_id. Sample of a response in JSON format:\n\n|  |\n| --- |\n| HTTP/1.1 200 OK  Content-Type: application/json     {\"message\\_id\":\"alss-a1b2c3d4-e5f67890\"} |\n\nIn case\n\n1) the request contains more than one DNIS (comma-separated)   \n2) the longMessageMode=split/split\\_sar and the message is longer than 160/70 symbols (GSM/Unicode respectively),\n\nthe response will look as follows:\n\n|  |\n| --- |\n| HTTP/1.1 200 OK  Content-Type: application/json     [{\"dnis\":\"34511121\",\"message\\_id\":\"5b4c46a8-8dc9-44b4-f55f-3bef56819305\",  \"segment\\_num\":\"1\"},{\"dnis\":\"34511121\",  \"message\\_id\":\"5b4c46a8-46bc-7ee6-4a16-7d4e5a0d14af\",\"segment\\_num\":\"2\"}] |\n\nIn case of rejected SMS (for example, no compatible routes found), the HTTP response status is - 400  Bad Request. The response body contains a string describing the reason for rejection, for example NO ROUTES.\n\n|  |\n| --- |\n| HTTP/1.1 400  Bad Request  Content-Type: text/html; charset=UTF-8     NO ROUTES |\n\nIn case an incorrect user name or password is provided, the HTTP status is 401 Unauthorized. The response body contains the string describing the reason for rejection.\n\n|  |\n| --- |\n| HTTP/1.1 401 Unauthorized  Content-Type: text/html; charset=UTF-8     not authorized (check login and password) |\n\nIf the service\\_type field exceeds 6 bytes, the response will look as follows:\n\n|  |\n| --- |\n| HTTP/1.1 400 Bad request\\n  Content-Type: text/html; charset=UTF-8\\n  Service type is invalid\\n |\n\nIf a message is considered a loop in accordance with the Loop Detection functionality, the response will be as follows:\n\n|  |\n| --- |\n| HTTP/1.1 508 Loop Detected  Content-Type: text/html; charset=UTF-8  loop detected |"
    ]

    for i, chunk in enumerate(chunks):
        store_chunk(chunk, doc_id="test", chunk_id=f"chunk_{i}")

    query = "Provide general information on how to send SMS message"
    
    for similar_chunk in search_chunks(query, k=5, threshold=0.0):
        print(similar_chunk)
        
     
        