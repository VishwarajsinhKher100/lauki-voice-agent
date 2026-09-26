from dotenv import load_dotenv

_ = load_dotenv()

from langchain_core.messages import ToolMessage
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
)
from livekit.plugins import deepgram, silero
from livekit.plugins import langchain as lk_langchain

from .config import settings
from .tools import lauki_agent


def _is_tool_token(item: object) -> bool:
    """Filter out ToolMessages emitted by LangGraph so tool outputs don't hit TTS."""
    return isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], ToolMessage)


class SpeechOnlyGraph:
    """Wraps the graph so tool return values never reach TTS."""

    def __init__(self, graph: object) -> None:
        self._graph = graph

    def astream(self, *args: object, **kwargs: object):
        inner = self._graph.astream(*args, **kwargs)

        async def _filtered():
            async for item in inner:
                if not _is_tool_token(item):
                    yield item

        return _filtered()


class LaukiVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are the Lauki Phones customer support voice assistant.",
            llm=lk_langchain.LLMAdapter(graph=SpeechOnlyGraph(lauki_agent)),
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Initialize the voice session with STT, TTS, and Silero VAD
    session = AgentSession(
        stt=deepgram.STT(model=settings.stt_model, language=settings.stt_language),
        tts=deepgram.TTS(),
        vad=silero.VAD.load(),
    )

    # Start the agent in the room
    await session.start(agent=LaukiVoiceAgent(), room=ctx.room)

    # Send initial greeting
    await session.generate_reply(
        user_input="Greet the user as Lauki Phones customer assistant and ask how you can help."
    )


if __name__ == "__main__":
    cli.run_app(server)