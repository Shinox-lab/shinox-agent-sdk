"""A2A Bridge — drop-in replacement for a LangGraph brain that proxies to a remote A2A server.

Allows external agents (built with any framework) to join the Shinox mesh
by exposing an A2A protocol server. The bridge implements the same
ainvoke(state, config) interface expected by ShinoxWorkerAgent.

Requires: pip install a2a-sdk
"""

from __future__ import annotations

import uuid


class A2ABridge:
    """Drop-in replacement for a LangGraph brain that proxies to a remote A2A server.

    Implements the ainvoke(state, config) interface expected by ShinoxWorkerAgent.
    """

    def __init__(self, agent_url: str):
        self.agent_url = agent_url
        self._client = None
        self._agent_card = None

    async def _ensure_client(self):
        """Lazy-initialize the A2A client by connecting to the remote agent."""
        if self._client is not None:
            return
        from a2a.client.client import ClientConfig
        from a2a.client.client_factory import ClientFactory

        self._client = await ClientFactory.connect(
            self.agent_url,
            client_config=ClientConfig(streaming=False),
        )
        self._agent_card = await self._client.get_card()

    @property
    def agent_card(self):
        """Return the cached AgentCard, or None if not yet fetched.

        Call fetch_agent_card() to ensure the card is loaded.
        """
        return self._agent_card

    async def fetch_agent_card(self):
        """Fetch the remote agent's AgentCard (for registration/discovery)."""
        await self._ensure_client()
        return self._agent_card

    async def ainvoke(self, state: dict, config: dict | None = None) -> dict:
        """A2A-backed implementation of the LangGraph ainvoke interface.

        Extracts the query from state["messages"][-1].content,
        sends it via A2A send_message, collects the response,
        and returns it in LangGraph state format.
        """
        await self._ensure_client()

        query = state["messages"][-1].content
        context_id = (config or {}).get("configurable", {}).get(
            "thread_id", "default"
        )

        from a2a.types import Message, Part, Role, TextPart

        a2a_message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=query))],
            message_id=str(uuid.uuid4()),
            context_id=context_id,
        )

        response_text = ""
        async for event in self._client.send_message(a2a_message):
            if isinstance(event, Message):
                for part in event.parts:
                    if hasattr(part.root, "text"):
                        response_text += part.root.text
            elif isinstance(event, tuple):
                task, _update = event
                if task.artifacts:
                    for artifact in task.artifacts:
                        for part in artifact.parts:
                            if hasattr(part.root, "text"):
                                response_text = part.root.text

        if not response_text:
            response_text = "No response from remote agent."

        from langchain_core.messages import AIMessage

        return {"messages": state["messages"] + [AIMessage(content=response_text)]}

    async def close(self):
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.close()
            self._client = None
