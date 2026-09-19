from __future__ import annotations

from openai import OpenAI

from kaggle_vllm_nebius.config import Settings


class NebiusClient:
    def __init__(self, settings: Settings):
        if not settings.nebius_api_key:
            raise RuntimeError(
                "NEBIUS_API_KEY is not set. Copy .env.example to .env and "
                "add your Nebius Builder Program API key."
            )
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.nebius_base_url,
            api_key=settings.nebius_api_key,
        )

    def smoke(self) -> str:
        expected = "KAGGLE_VLLM_NEBIUS_TOKEN_FACTORY_OK"

        completion = self.client.chat.completions.create(
            model=self.settings.nebius_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Follow the user's output instruction exactly. "
                        "Return no explanation and no additional text."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Reply with exactly: {expected}",
                },
            ],
            temperature=1.0,
            top_p=0.95,
            max_tokens=64,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        )

        content = (completion.choices[0].message.content or "").strip()

        if content != expected:
            raise RuntimeError(
                "Token Factory responded, but the smoke-test sentinel "
                f"did not match. Received: {content!r}"
            )

        return content

    def chat(self, *, messages: list[dict], tools: list[dict] | None = None):
        kwargs = {
            "model": self.settings.nebius_model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs)
