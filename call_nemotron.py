
import requests
import os
import base64
import sys
from dotenv import load_dotenv
import json
from typing import Any, Dict, List
from tools import grocery_price_lookup

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True
query = "Describe the scene"

# Load .env before accessing environment variables
load_dotenv()
kApiKey = os.getenv("NVIDIA_API_KEY", "$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC")

# ext: {mime, media_type}
kSupportedList = {
    "png": ["image/png", "image_url"],
    "jpg": ["image/jpeg", "image_url"],
    "jpeg": ["image/jpeg", "image_url"],
    "webp": ["image/webp", "image_url"],
    "mp4": ["video/mp4", "video_url"],
    "webm": ["video/webm", "video_url"],
    "mov": ["video/mov", "video_url"]
}

def get_extension(filename):
    _, ext = os.path.splitext(filename)
    ext = ext[1:].lower()
    return ext

def mime_type(ext):
    return kSupportedList[ext][0]

def media_type(ext):
    return kSupportedList[ext][1]

def encode_media_base64(media_file):
    """Encode media file to base64 string"""
    with open(media_file, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _post_completion(infer_url: str, headers: Dict[str, str], payload: Dict[str, Any], stream: bool = False):
    response = requests.post(infer_url, headers=headers, json=payload, stream=stream)
    if stream:
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))
    else:
        # Debug print; keep JSON for caller
        pass
    return response.json()


def _maybe_execute_tools(response_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of tool result messages to append, based on tool_calls in response."""
    tool_messages: List[Dict[str, Any]] = []
    try:
        choices = response_json.get("choices") or []
        if not choices:
            return tool_messages
        message = choices[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        for call in tool_calls:
            # Expected OpenAI-style tool call structure
            tool_type = call.get("type")
            if tool_type != "function":
                continue
            func = call.get("function") or {}
            name = func.get("name")
            arguments_raw = func.get("arguments") or "{}"
            try:
                arguments = json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw
            except Exception:
                arguments = {}

            content = ""
            if name == "grocery_price_lookup":
                query = str(arguments.get("query", ""))
                try:
                    results = grocery_price_lookup(query=query)
                    content = json.dumps({"results": results})
                except Exception as tool_err:
                    content = json.dumps({"error": f"grocery_price_lookup failed: {tool_err}"})
            else:
                content = json.dumps({"error": f"Unknown tool: {name}"})

            tool_message = {
                "role": "tool",
                # Some APIs expect tool_call_id; include when present
                **({"tool_call_id": call.get("id")} if call.get("id") else {}),
                "name": name,
                "content": content,
            }
            tool_messages.append(tool_message)
    except Exception:
        # On parsing issues, return no tool messages
        return []
    return tool_messages


def chat_with_text(infer_url, query: str, stream: bool = False):
    headers = {
        "Authorization": f"Bearer {kApiKey}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if stream:
        headers["Accept"] = "text/event-stream"

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that suggests budget-friendly, nutritious meals and "
                "shopping guidance for one week. You have access to the grocery_price_lookup tool "
                "to search for prices of food items at a local grocery store. DO NOT recommend food "
                "unless you know its price."
            ),
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "grocery_price_lookup",
                "description": "Searches for prices of food items at a local grocery store.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Food name in lowercase without punctuation"},
                        # "count": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    payload = {
        "max_tokens": 20000,
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "messages": messages,
        "stream": stream,
        "model": "nvidia/nemotron-nano-12b-v2-vl",
        "tools": tools,
        # Optionally:
        # "tool_choice": "auto",
    }

    # First completion
    first = _post_completion(infer_url, headers, payload, stream=stream)

    # If model requested tools, execute and send follow-up
    tool_messages = _maybe_execute_tools(first)
    if tool_messages:
        # Extend conversation with assistant's tool_calls message and tool results
        # We need the assistant message that contained the tool_calls
        try:
            assistant_msg = (first.get("choices") or [{}])[0].get("message") or {}
        except Exception:
            assistant_msg = {}
        followup_messages = messages + [assistant_msg] + tool_messages

        followup_payload = {
            "max_tokens": 20000,
            "temperature": 1,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "messages": followup_messages,
            "stream": False,  # follow-up as single reply for simplicity
            "model": "nvidia/nemotron-nano-12b-v2-vl",
            "tools": tools,
            # "tool_choice": "auto",
        }
        second = _post_completion(infer_url, headers, followup_payload, stream=False)
        return second

    return first

def chat_with_media(infer_url, media_files, query: str, stream: bool = False):
    assert isinstance(media_files, list), f"{media_files}"
    
    has_video = False
    
    # Build content based on whether we have media files
    if len(media_files) == 0:
        # Text-only mode
        content = query
    else:
        # Build content array with text and media
        content = [{"type": "text", "text": query}]
        
        for media_file in media_files:
            ext = get_extension(media_file)
            assert ext in kSupportedList, f"{media_file} format is not supported"
            
            media_type_key = media_type(ext)
            if media_type_key == "video_url":
                has_video = True
            
            print(f"Encoding {media_file} as base64...")
            base64_data = encode_media_base64(media_file)
            
            # Add media to content array
            media_obj = {
                "type": media_type_key,
                media_type_key: {
                    "url": f"data:{mime_type(ext)};base64,{base64_data}"
                }
            }
            content.append(media_obj)
        
        if has_video:
            assert len(media_files) == 1, "Only single video supported."
    
    headers = {
        "Authorization": f"Bearer {kApiKey}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if stream:
        headers["Accept"] = "text/event-stream"

    # Add system message with appropriate prompt
    # Videos only support /no_think, images support both
    
    system_prompt = "/think"
    
    
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": content,
        }
    ]
    payload = {
        "max_tokens": 4096,
        "temperature": 1,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "messages": messages,
        "stream": stream,
        "model": "nvidia/nemotron-nano-12b-v2-vl",
    }

    response = requests.post(infer_url, headers=headers, json=payload, stream=stream)
    if stream:
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))
    else:
        print(response.json())

if __name__ == "__main__":
    """ Usage:
        python test.py                                    # Text-only
        python test.py sample.mp4                         # Single video
        python test.py sample1.png sample2.png            # Multiple images
    """

    media_samples = list(sys.argv[1:])
    chat_with_media(invoke_url, media_samples, query, stream)
