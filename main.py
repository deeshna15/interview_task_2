import os
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

import database as db_lib
import google.generativeai as genai
import openai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Universal Memory Chat API", version="1.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class ThreadCreate(BaseModel):
    title: str
    is_summary_thread: bool = False

class ThreadResponse(BaseModel):
    id: int
    title: str
    summary: Optional[str] = None
    is_summary_thread: bool
    created_at: str

    class Config:
        from_attributes = True

class MessageCreate(BaseModel):
    content: str
    api_key: Optional[str] = None
    provider: Optional[str] = "gemini"  # "gemini" or "openai"

class MessageResponse(BaseModel):
    id: int
    thread_id: int
    sender: str
    content: str
    created_at: str

    class Config:
        from_attributes = True

# Helper: Call LLM
def call_llm(prompt: str, system_instruction: str = None, api_key: str = None, provider: str = "gemini") -> str:
    if provider == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise HTTPException(
                status_code=400,
                detail="Gemini API Key is missing. Please set GEMINI_API_KEY in your env or provide it in the UI."
            )
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Gemini API Error: {str(e)}"
            )

    elif provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API Key is missing. Please set OPENAI_API_KEY in your env or provide it in the UI."
            )
        try:
            client = openai.OpenAI(api_key=key)
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API Error: {str(e)}"
            )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {provider}")


# Helper: Background task to update thread summary
def update_thread_summary_bg(thread_id: int, api_key: Optional[str], provider: str):
    db = next(db_lib.get_db())
    try:
        messages = db_lib.get_messages(db, thread_id)
        if not messages:
            return
        
        # Compile messages for summarization
        chat_transcript = ""
        for msg in messages:
            chat_transcript += f"{msg.sender.capitalize()}: {msg.content}\n"
        
        prompt = (
            f"Please summarize the following chat conversation in 1 or 2 concise sentences. "
            f"Focus on the main topics and decisions. This summary will be used as background memory for future chats.\n\n"
            f"Conversation:\n{chat_transcript}"
        )
        
        summary = call_llm(
            prompt=prompt,
            system_instruction="You are a summarization assistant. Be extremely concise.",
            api_key=api_key,
            provider=provider
        )
        db_lib.update_thread_summary(db, thread_id, summary.strip())
    except Exception as e:
        print(f"Error in background summarization for thread {thread_id}: {e}")
    finally:
        db.close()


# API Endpoints
@app.get("/threads", response_model=List[ThreadResponse])
def read_threads(db: Session = Depends(db_lib.get_db)):
    threads = db_lib.get_threads(db)
    return [
        ThreadResponse(
            id=t.id,
            title=t.title,
            summary=t.summary,
            is_summary_thread=t.is_summary_thread,
            created_at=t.created_at.isoformat()
        ) for t in threads
    ]

@app.post("/threads", response_model=ThreadResponse)
def create_new_thread(thread: ThreadCreate, db: Session = Depends(db_lib.get_db)):
    # If a summary thread is requested, make sure we only have one or just allow creating it
    new_t = db_lib.create_thread(db, title=thread.title, is_summary_thread=thread.is_summary_thread)
    return ThreadResponse(
        id=new_t.id,
        title=new_t.title,
        summary=new_t.summary,
        is_summary_thread=new_t.is_summary_thread,
        created_at=new_t.created_at.isoformat()
    )

@app.delete("/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chat_thread(thread_id: int, db: Session = Depends(db_lib.get_db)):
    success = db_lib.delete_thread(db, thread_id)
    if not success:
        raise HTTPException(status_code=404, detail="Thread not found")
    return

@app.get("/threads/{thread_id}/messages", response_model=List[MessageResponse])
def read_messages(thread_id: int, db: Session = Depends(db_lib.get_db)):
    messages = db_lib.get_messages(db, thread_id)
    return [
        MessageResponse(
            id=m.id,
            thread_id=m.thread_id,
            sender=m.sender,
            content=m.content,
            created_at=m.created_at.isoformat()
        ) for m in messages
    ]

@app.post("/threads/{thread_id}/messages", response_model=MessageResponse)
def post_new_message(
    thread_id: int,
    payload: MessageCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(db_lib.get_db)
):
    thread = db_lib.get_thread(db, thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Save user message
    db_lib.add_message(db, thread_id=thread_id, sender="user", content=payload.content)

    # 1. Handle special summary thread (Thread 3)
    if thread.is_summary_thread:
        # Get histories of all other threads
        all_threads = db_lib.get_threads(db)
        context_parts = []
        for t in all_threads:
            if t.is_summary_thread or t.id == thread_id:
                continue
            t_msgs = db_lib.get_messages(db, t.id)
            if not t_msgs:
                continue
            
            chat_str = f"--- Thread '{t.title}' ---\n"
            for m in t_msgs:
                chat_str += f"{m.sender.upper()}: {m.content}\n"
            context_parts.append(chat_str)
        
        global_context = "\n".join(context_parts) if context_parts else "No other thread messages exist yet."
        
        system_instruction = (
            "You are a Summary Assistant. You have access to the complete conversation history of "
            "all other chat threads. Below is the history of all threads. Answer the user's request "
            "specifically referencing this aggregate history.\n\n"
            f"All Threads History:\n{global_context}"
        )
        
        ai_response = call_llm(
            prompt=payload.content,
            system_instruction=system_instruction,
            api_key=payload.api_key,
            provider=payload.provider
        )
        
        # Save AI message
        ai_msg = db_lib.add_message(db, thread_id=thread_id, sender="ai", content=ai_response)
        return MessageResponse(
            id=ai_msg.id,
            thread_id=ai_msg.thread_id,
            sender=ai_msg.sender,
            content=ai_msg.content,
            created_at=ai_msg.created_at.isoformat()
        )

    # 2. Handle normal chat thread
    else:
        # Get active thread messages to build prompt/history
        current_messages = db_lib.get_messages(db, thread_id)
        
        # Extract summaries of other normal threads for Universal Memory
        all_threads = db_lib.get_threads(db)
        memory_parts = []
        for t in all_threads:
            # Skip current thread and summary threads
            if t.id == thread_id or t.is_summary_thread:
                continue
            if t.summary:
                memory_parts.append(f"- Thread '{t.title}': {t.summary}")
            else:
                # If no summary, compile a brief line or skip
                memory_parts.append(f"- Thread '{t.title}': (No summary yet)")
        
        universal_memory = "\n".join(memory_parts) if memory_parts else "No past conversations."

        system_instruction = (
            "You are a helpful AI Assistant equipped with a 'Universal Memory' across multiple chat threads. "
            "Below are summaries of discussions you had with the user in other threads. Use this memory "
            "to answer contextually if the user refers to past discussions or asks about what they said in other threads.\n\n"
            f"Universal Memory (Other Threads):\n{universal_memory}\n\n"
            "Be conversational, direct, and leverage the memory whenever relevant."
        )

        # Build prompt using current thread history
        prompt_parts = []
        for m in current_messages:
            # The last message is the user's new message (already in current_messages since we saved it)
            # Skip the last one if we want to pass it as the main prompt, or just pass all.
            # Building standard message thread content for generation:
            prompt_parts.append(f"{m.sender.capitalize()}: {m.content}")
        
        prompt = "\n".join(prompt_parts) + "\nAI:"

        ai_response = call_llm(
            prompt=prompt,
            system_instruction=system_instruction,
            api_key=payload.api_key,
            provider=payload.provider
        )

        # Save AI message
        ai_msg = db_lib.add_message(db, thread_id=thread_id, sender="ai", content=ai_response)

        # Queue background task to update this thread's summary so it can be used for memory next time
        background_tasks.add_task(
            update_thread_summary_bg,
            thread_id=thread_id,
            api_key=payload.api_key,
            provider=payload.provider
        )

        return MessageResponse(
            id=ai_msg.id,
            thread_id=ai_msg.thread_id,
            sender=ai_msg.sender,
            content=ai_msg.content,
            created_at=ai_msg.created_at.isoformat()
        )

@app.post("/threads/{thread_id}/regenerate-summary", response_model=MessageResponse)
def regenerate_summary_thread(
    thread_id: int,
    payload: MessageCreate,
    db: Session = Depends(db_lib.get_db)
):
    """Special endpoint to manually trigger a summary synthesis of all other threads."""
    thread = db_lib.get_thread(db, thread_id)
    if not thread or not thread.is_summary_thread:
        raise HTTPException(status_code=400, detail="Not a summary thread")

    # Get histories of all other threads
    all_threads = db_lib.get_threads(db)
    context_parts = []
    for t in all_threads:
        if t.is_summary_thread:
            continue
        t_msgs = db_lib.get_messages(db, t.id)
        if not t_msgs:
            continue
        
        chat_str = f"### Thread: {t.title}\n"
        for m in t_msgs:
            chat_str += f"- **{m.sender.capitalize()}**: {m.content}\n"
        context_parts.append(chat_str)
    
    global_context = "\n\n".join(context_parts) if context_parts else "No conversation history exists in other threads yet."

    prompt = (
        f"Generate a comprehensive, beautifully structured executive summary of all conversations across all threads. "
        f"Organize it logically with sections, bullet points, and key takeaways using Markdown formatting. "
        f"If no conversation history exists, return a welcoming message indicating how to use the app.\n\n"
        f"Here is the history to summarize:\n\n{global_context}"
    )

    summary_content = call_llm(
        prompt=prompt,
        system_instruction="You are a professional executive summaries assistant. Format outputs using markdown with clear headings.",
        api_key=payload.api_key,
        provider=payload.provider
    )

    # Save to the summary thread as a special message
    ai_msg = db_lib.add_message(db, thread_id=thread_id, sender="ai", content=summary_content)
    return MessageResponse(
        id=ai_msg.id,
        thread_id=ai_msg.thread_id,
        sender=ai_msg.sender,
        content=ai_msg.content,
        created_at=ai_msg.created_at.isoformat()
    )
