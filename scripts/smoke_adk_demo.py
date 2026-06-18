from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tri_model_agent_demo.agent import root_agent


async def main() -> None:
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="tri_model_agent_demo", user_id="user")
    runner = Runner(
        agent=root_agent,
        app_name="tri_model_agent_demo",
        session_service=session_service,
    )
    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Analyze https://example.com")],
    )
    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=message,
    ):
        if event.content and event.content.parts:
            print(event.content.parts[0].text)


if __name__ == "__main__":
    asyncio.run(main())
