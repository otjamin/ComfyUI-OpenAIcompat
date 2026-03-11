import json
import urllib.request
import urllib.error
import base64
from io import BytesIO
from PIL import Image
import torch
import numpy as np

class OpenAICompatChatNode:
    """
    A generic OpenAI API compatible chat node.
    Allows connecting to any LLM that supports the OpenAI chat/completions endpoint format.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_url": ("STRING", {
                    "default": "https://api.openai.com/v1",
                    "tooltip": "Base URL or full endpoint URL (e.g., https://api.openai.com/v1 or https://api.openai.com/v1/chat/completions)."
                }),
                "model": ("STRING", {
                    "default": "gpt-4o",
                    "tooltip": "The model ID to use (e.g., gpt-4o, gemini-1.5-pro)."
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail.",
                    "tooltip": "The user prompt."
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a helpful assistant.",
                    "tooltip": "Optional system instructions."
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "Optional API key for authorization."
                }),
                "image": ("IMAGE", {
                    "tooltip": "Optional input image(s). If a batch is provided, each image is processed individually."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "OpenAIcompat"

    def _tensor_to_base64_jpeg(self, tensor):
        """Converts a ComfyUI image tensor [H, W, C] to a base64 encoded JPEG string."""
        i = 255. * tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def _make_request(self, endpoint_url, headers, payload):
        """Makes the HTTP POST request to the API."""
        req = urllib.request.Request(
            endpoint_url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return f"Error: Unexpected response format: {result}"
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            return f"HTTP Error {e.code}: {e.reason}\nDetails: {error_body}"
        except urllib.error.URLError as e:
            return f"URL Error: {e.reason}"
        except Exception as e:
             return f"Error: {str(e)}"

    def generate(self, api_url, model, prompt, system_prompt="", api_key="", image=None):
        # Format the endpoint URL
        endpoint_url = api_url.strip()
        if endpoint_url.endswith("/"):
            endpoint_url = endpoint_url[:-1]
            
        if not endpoint_url.endswith("/chat/completions"):
            endpoint_url += "/chat/completions"

        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        if api_key and isinstance(api_key, str) and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        responses = []

        if image is not None:
            # image is a tensor of shape [batch_size, height, width, channels]
            # Process each image in the batch individually
            for i in range(image.shape[0]):
                single_image_tensor = image[i]
                base64_image = self._tensor_to_base64_jpeg(single_image_tensor)
                
                messages = []
                if system_prompt and isinstance(system_prompt, str) and system_prompt.strip():
                    messages.append({"role": "system", "content": system_prompt})
                
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        }
                    ]
                })

                payload = {
                    "model": model,
                    "messages": messages
                }

                response_text = self._make_request(endpoint_url, headers, payload)
                responses.append(response_text)
            
            # If it was a batch of 1, return the single string (more compatible with other nodes)
            # If it was > 1, return the list. ComfyUI can handle lists for batched string processing.
            if len(responses) == 1:
                 return (responses[0],)
            return (responses,)
            
        else:
            # No image provided, just text chat
            messages = []
            if system_prompt and isinstance(system_prompt, str) and system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": model,
                "messages": messages
            }

            response_text = self._make_request(endpoint_url, headers, payload)
            return (response_text,)

NODE_CLASS_MAPPINGS = {
    "OpenAICompatChatNode": OpenAICompatChatNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAICompatChatNode": "OpenAI Chat (Compatible)"
}
