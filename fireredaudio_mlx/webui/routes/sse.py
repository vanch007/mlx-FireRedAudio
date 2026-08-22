"""Server-Sent Events streaming endpoint."""

import asyncio
import json
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from ..sse import broadcaster
from ..manager import model_manager

router = APIRouter(tags=["Events"])


@router.get("/events")
async def subscribe_events(request: Request):
    q = broadcaster.subscribe()

    async def event_generator():
        yield {
            "event": "system_status",
            "data": json.dumps(model_manager.get_status(), ensure_ascii=False),
        }
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield {
                        "event": msg["event"],
                        "data": json.dumps(msg["data"], ensure_ascii=False),
                    }
                except asyncio.TimeoutError:
                    yield {
                        "event": "ping",
                        "data": json.dumps({"time": asyncio.get_event_loop().time()}),
                    }
        finally:
            broadcaster.unsubscribe(q)

    return EventSourceResponse(event_generator())
