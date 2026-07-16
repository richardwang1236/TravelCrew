"""LLM client abstraction layer for the travel planning agent.

This module provides a base class and a concrete DeepSeek implementation
for interacting with OpenAI-compatible chat-completion endpoints.

Key features:
    - **Custom retry logic** with exponential backoff (2^n seconds) that
      logs every failed attempt, replacing the OpenAI SDK's silent retries.
    - **Optional JSON-mode** via ``response_format`` for structured output.
    - **Optional thinking/reasoning** pass-through via ``extra_body`` for
      models that support chain-of-thought prompting.
    - Configurable timeout and max-token limits per request.
"""

import time
import json
import logging

from openai import OpenAI, OpenAIError


# Configure root logger so that retry warnings and errors are visible
# in the console alongside other agent-node log output.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class BaseLLMClient:
    """Abstract base class for LLM chat-completion clients.

    Defines the common constructor signature and the ``chat`` interface
    that every concrete provider must implement.

    Attributes:
        model_name (str): Identifier of the model to call (e.g. ``deepseek-v4-flash``).
        base_url (str): Root URL of the OpenAI-compatible API server.
        api_key (str): Bearer token used for authentication.
        timeout (float): Per-request timeout in seconds.
        max_retries (int): How many times to retry on transient failures.
    """

    def __init__(self, model_name: str, base_url: str, api_key: str, timeout: float = 300.0, max_retries: int = 3) -> None:
        """Initialize base LLM configuration.

        Args:
            model_name: Model identifier string sent in the API request body.
            base_url: Base URL of the OpenAI-compatible inference server.
            api_key: API key for authentication.
            timeout: Maximum seconds to wait for a single API response.
                Defaults to 300.0 (5 minutes) to accommodate reasoning models.
            max_retries: Number of retry attempts on transient errors.
                Defaults to 3.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600, streaming=False, json_format=False, thinking=False) -> str:
        """Send a chat-completion request and return the model's text response.

        Subclasses **must** override this method.

        Args:
            system_prompt: System-level instruction prepended to the message list.
            user_prompt: The user message / question to send to the model.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
                Defaults to 0.2.
            max_tokens: Upper bound on tokens in the generated response.
                Defaults to 600.
            streaming: If True, return a stream iterator instead of a string.
                Defaults to False.
            json_format: If True, request JSON-object output mode from the model.
                Defaults to False.
            thinking: If True, enable the model's internal reasoning/thinking
                chain (provider-specific).  Defaults to False.

        Returns:
            str: The model's text (or JSON) response.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError

    def chat_stream(self, system_prompt: str, user_prompt: str, temperature: float = None, max_tokens: int = None, thinking=False):
        """Stream a chat-completion response, yielding text chunks as they arrive.

        Subclasses **must** override this method.

        Args:
            system_prompt: System-level instruction prepended to the message list.
            user_prompt: The user message / question to send to the model.
            temperature: Sampling temperature.  ``None`` lets the model use its default.
            max_tokens: Upper bound on tokens in the generated response.
            thinking: If True, enable the model's internal reasoning chain.

        Yields:
            str: Each text delta chunk from the model.

        Raises:
            NotImplementedError: Always, unless overridden by a subclass.
        """
        raise NotImplementedError


class DeepSeekChatClient(BaseLLMClient):
    """Concrete LLM client targeting DeepSeek (or any OpenAI-compatible) API.

    Wraps the official ``openai.OpenAI`` SDK with:
        - Library-level retries **disabled** (``max_retries=0``) so that
          our own retry loop — which logs every attempt — has full control.
        - Exponential-backoff retry: sleeps ``2^attempt`` seconds between
          attempts (i.e. 2 s, 4 s, 8 s …) before raising on final failure.
        - Pass-through for DeepSeek-specific features such as the
          ``thinking`` / chain-of-thought toggle via ``extra_body``.

    Example::

        client = DeepSeekChatClient(
            api_key="sk-xxx",
            base_url="https://api.deepseek.com",
            model_name="deepseek-v4-flash",
        )
        answer = client.chat(
            system_prompt="You are a travel expert.",
            user_prompt="Plan a 3-day trip to Tokyo.",
        )
    """

    def __init__(self, model_name: str, base_url: str, api_key: str, timeout: float = 300.0, max_retries: int = 3):
        """Initialize the DeepSeek client and create the underlying OpenAI SDK instance.

        Args:
            model_name: Model identifier (e.g. ``deepseek-v4-flash``).
            base_url: Root URL of the DeepSeek-compatible API.
            api_key: Authentication key for API requests.
            timeout: Per-request timeout in seconds.  Defaults to 300.0.
            max_retries: Number of custom retry attempts.  Defaults to 3.
                Note: the OpenAI SDK's built-in retries are set to 0;
                all retry logic is handled in :meth:`chat`.
        """
        super().__init__(model_name, base_url, api_key, timeout, max_retries)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout,  # Set request timeout
            max_retries=0  # Disable openai library's default silent retries; handled by our own logic for logging
        )

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = None, max_tokens: int = None, streaming=False, json_format=False, thinking=False) -> str:
        """Send a chat-completion request with automatic retry and exponential backoff.

        Builds the request payload, optionally enables JSON-mode and/or
        chain-of-thought reasoning, then attempts the call up to
        ``self.max_retries`` times.  On each transient ``OpenAIError`` the
        method sleeps ``2^attempt`` seconds before retrying.

        Args:
            system_prompt: System-level instruction for the model.
            user_prompt: The user's message / question.
            temperature: Sampling temperature.  ``None`` lets the model use
                its default.
            max_tokens: Maximum response length in tokens.  ``None`` lets
                the model use its default.
            streaming: If True, request a streaming response (not yet
                consumed by callers).  Defaults to False.
            json_format: If True, set ``response_format`` to
                ``{"type": "json_object"}`` so the model returns valid JSON.
                Defaults to False.
            thinking: If True, enable DeepSeek's internal reasoning mode
                via ``extra_body.thinking.type = "enabled"``.
                Defaults to False.

        Returns:
            str: The text content of the first choice in the completion
            response.

        Raises:
            OpenAIError: If all retry attempts are exhausted without a
                successful response.
        """
        thinking_value = "enabled" if thinking else "disabled"

        # Prepare common parameters shared across every request variant.
        # - reasoning_effort="high" tells the model to use its most thorough
        #   internal reasoning pass (DeepSeek-specific parameter).
        # - extra_body carries provider-specific fields not in the OpenAI spec,
        #   such as the thinking/chain-of-thought toggle.
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": streaming,
            "reasoning_effort": "high",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": thinking_value}}
        }

        # When JSON output is requested, force the model to emit a valid
        # JSON object (the caller is responsible for parsing).
        # Note: some models may still return markdown-fenced JSON; callers
        # should handle both raw and fenced responses.
        if json_format:
            kwargs["response_format"] = {'type': 'json_object'}

        # Retry loop with exponential backoff.
        # Sleep durations: 2^1=2s, 2^2=4s, 2^3=8s, ...
        # Each failed attempt is logged at WARNING level so that transient
        # errors (rate limits, network blips) are visible in the console.
        # Only the final failure is logged at ERROR level and re-raised.
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

            except OpenAIError as e:
                if attempt == self.max_retries:
                    # Final attempt exhausted — log and re-raise so the
                    # caller can decide how to handle the failure.
                    logging.error(
                        f"LLM call failed after max retries ({self.max_retries}). Error: {str(e)}")
                    raise e

                # Exponential backoff: 2, 4, 8 seconds...
                sleep_time = 2 ** attempt
                logging.warning(
                    f"LLM call encountered an error (attempt {attempt}/{self.max_retries}): {str(e)}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)

    def chat_stream(self, system_prompt: str, user_prompt: str, temperature: float = None, max_tokens: int = None, thinking=False):
        """Stream a chat-completion response, yielding text chunks as they arrive.

        Uses the OpenAI SDK's streaming API to get a real-time iterator over
        the model's output. Each yielded chunk is a text delta string.

        Args:
            system_prompt: System-level instruction for the model.
            user_prompt: The user's message / question.
            temperature: Sampling temperature.  ``None`` lets the model use its default.
            max_tokens: Maximum response length in tokens.
            thinking: If True, enable DeepSeek's internal reasoning mode.

        Yields:
            str: Each text delta chunk from the model.

        Raises:
            OpenAIError: If the API call fails (no retry for streaming to avoid
                partial output issues).
        """
        thinking_value = "enabled" if thinking else "disabled"

        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "reasoning_effort": "high",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"thinking": {"type": thinking_value}}
        }

        logging.info(f"Starting streaming LLM call (model={self.model_name})...")
        try:
            response = self.client.chat.completions.create(**kwargs)
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
        except OpenAIError as e:
            logging.error(f"Streaming LLM call failed: {str(e)}")
            raise e
        logging.info("Streaming LLM call completed.")

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> dict:
        """Send a chat-completion request with tool/function calling support.

        Used by the ReplanAgent for autonomous, iterative plan modification.
        Returns the full message object from the API response, which may
        contain ``tool_calls`` that the caller must execute and feed back.

        Args:
            messages: List of message dicts with ``role`` and ``content``,
                following the OpenAI chat format (system/user/assistant/tool).
            tools: List of tool definitions in OpenAI function-calling format.
            tool_choice: ``"auto"`` to let the model decide, ``"required"`` to
                force a tool call, or ``"none"`` to suppress tools.
            temperature: Sampling temperature.  Defaults to 0.3.
            max_tokens: Maximum response tokens.  Defaults to 2000.

        Returns:
            dict: The ``message`` object from ``response.choices[0]``,
            containing ``role``, ``content`` (may be null), and optionally
            ``tool_calls``.

        Raises:
            OpenAIError: If all retry attempts are exhausted.
        """
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                # Convert to dict for easier manipulation in agent loop
                result = {"role": msg.role, "content": msg.content}
                if msg.tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                return result
            except OpenAIError as e:
                if attempt == self.max_retries:
                    logging.error(
                        f"LLM tool-call failed after max retries ({self.max_retries}). Error: {str(e)}")
                    raise e
                sleep_time = 2 ** attempt
                logging.warning(
                    f"LLM tool-call error (attempt {attempt}/{self.max_retries}): {str(e)}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
