import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from dotenv import load_dotenv

load_dotenv()

# Use SQLite in /tmp folder on Vercel to bypass read-only filesystem restrictions
if os.getenv("VERCEL"):
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:////tmp/chat_app.db")
else:
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_app.db")

engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ThreadModel(Base):
    __tablename__ = "threads"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=True)  # Stores summary for universal memory
    is_summary_thread = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    messages = relationship("MessageModel", back_populates="thread", cascade="all, delete-orphan")

class MessageModel(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(Integer, ForeignKey("threads.id"), nullable=False)
    sender = Column(String, nullable=False)  # "user" or "ai"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    thread = relationship("ThreadModel", back_populates="messages")

# Create tables
Base.metadata.create_all(bind=engine)

# DB Helpers
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Operations
def create_thread(db, title: str, is_summary_thread: bool = False):
    db_thread = ThreadModel(title=title, is_summary_thread=is_summary_thread)
    db.add(db_thread)
    db.commit()
    db.refresh(db_thread)
    return db_thread

def get_threads(db):
    return db.query(ThreadModel).order_by(ThreadModel.created_at.asc()).all()

def get_thread(db, thread_id: int):
    return db.query(ThreadModel).filter(ThreadModel.id == thread_id).first()

def update_thread_summary(db, thread_id: int, summary: str):
    db_thread = get_thread(db, thread_id)
    if db_thread:
        db_thread.summary = summary
        db.commit()
        db.refresh(db_thread)
    return db_thread

def delete_thread(db, thread_id: int):
    db_thread = get_thread(db, thread_id)
    if db_thread:
        db.delete(db_thread)
        db.commit()
        return True
    return False

def add_message(db, thread_id: int, sender: str, content: str):
    db_message = MessageModel(thread_id=thread_id, sender=sender, content=content)
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message

def get_messages(db, thread_id: int):
    return db.query(MessageModel).filter(MessageModel.thread_id == thread_id).order_by(MessageModel.created_at.asc()).all()

def get_all_messages_across_threads(db):
    """Retrieve all messages across all non-summary threads for global context/summary."""
    return db.query(MessageModel).join(ThreadModel).filter(ThreadModel.is_summary_thread == False).order_by(MessageModel.created_at.asc()).all()
