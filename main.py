import json
import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения из .env
load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
ASSISTANT2_ID = os.getenv("ASSISTANT2_ID")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    question: str
    thread_id: str = None
    assistant_type: str = None

async def get_new_thread_id():
    thread = await client.beta.threads.create()
    return thread.id

async def process_question(question, thread_id=None, assistant_id=None):
    try:
        logger.info("Processing question with GPT-4")
        if not thread_id:
            thread = await client.beta.threads.create()
            thread_id = thread.id
            logger.info(f"New thread created with ID: {thread_id}")
        else:
            logger.info(f"Using existing thread ID: {thread_id}")

        await client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=question
        )
        run = await client.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=assistant_id
        )

        while run.status in ["queued", "in_progress", "cancelling"]:
            await asyncio.sleep(1)
            run = await client.beta.threads.runs.retrieve(
                thread_id=thread_id, run_id=run.id
            )

        if run.status == "completed":
            messages = await client.beta.threads.messages.list(
                thread_id=thread_id
            )
            assistant_messages = [
                msg.content[0].text.value
                for msg in messages.data
                if msg.role == "assistant"
            ]
            if assistant_messages:
                # Проверка наличия JSON в первом ответе
                first_message = assistant_messages[0]
                logger.info(f"First message from assistant: {first_message}")
                if "```json" in first_message:
                    response_text, json_part = first_message.split("```json", 1)
                    response = {
                        "response": response_text.strip(),
                        "thread_id": thread_id,
                        "json": json.loads(json_part.split('```')[0].strip())
                    }
                else:
                    response = {
                        "response": first_message.strip(),
                        "thread_id": thread_id
                    }
                return response
            else:
                logger.warning("No assistant messages found.")
                return {
                    "response": "Не удалось получить ответ от ассистента.",
                    "thread_id": thread_id
                }
        else:
            logger.warning(f"Run not completed. Status: {run.status}")
            return {
                "response": "Не удалось получить ответ от ассистента.",
                "thread_id": thread_id
            }
    except Exception as e:
        logger.error(f"Error in process_question: {e}")
        return {
            "response": "Произошла ошибка при обработке вопроса.",
            "thread_id": thread_id
        }

@app.post("/query")
async def process_query(request: QueryRequest):
    try:
        assistant_id = ASSISTANT2_ID if request.assistant_type == "registration" else ASSISTANT_ID

        response = await process_question(
            request.question, request.thread_id, assistant_id
        )

        return response
    except Exception as e:
        logger.error(f"Error in process_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
