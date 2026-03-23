import json
from typing import Any, Dict, List, Optional, Protocol

from openai import OpenAI


class BaseLLMClient(Protocol):
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[List[str]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        ...


class DashScopeQwenClient:
    """
    OpenAI-compatible client for Alibaba DashScope / Bailian.

    Notes:
    - model defaults to qwen3.5-flash
    - images should be public URLs if you want multimodal input
    - set DASHSCOPE_API_KEY in your environment
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3.5-flash",
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def _build_user_content(
        self,
        user_prompt: str,
        images: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        content.append(
            {
                "type": "text",
                "text": user_prompt,
            }
        )

        if images:
            for img in images:
                # Assumes img is a URL. If you store local files, upload or convert first.
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img
                        },
                    }
                )

        return content

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[List[str]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": self._build_user_content(
                    user_prompt=user_prompt,
                    images=images,
                ),
            },
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )

        text = resp.choices[0].message.content
        if text is None:
            raise ValueError("Model returned empty content.")

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model did not return valid JSON. Raw output: {text}") from e