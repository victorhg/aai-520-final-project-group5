from ..worker.base_worker import BaseWorker
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# Simple in-memory store for this example.
_memory_store: List[Dict[str, Any]] = []
MEMORY_FILE = "./data/agent_memory.json"

def _load_memory():
    """Loads memories from the JSON file into the in-memory list."""
    global _memory_store
    if not os.path.exists(MEMORY_FILE):
        _memory_store = []
        return
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            _memory_store = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        _memory_store = []

def _save_memory():
    """Saves the in-memory list of memories to the JSON file."""
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(_memory_store, f, indent=2)

# Load memory when module is imported
_load_memory()

class MemoryWorker(BaseWorker):
    """
    A BaseWorker implementation for managing agent memories. It stores short text
    memories with optional tags in a JSON file so the agent can learn across runs.
    """
    def execute(self, *inputs) -> Any:
        """
        Manages agent memories. The first input is the operation ('add', 'search', 'get_recent').

        Usage:
            - execute('add', 'some memory text', ['tag1', 'tag2'])
            - execute('search', 'query text')
            - execute('get_recent', 5)
        """
        if not inputs:
            raise ValueError("MemoryWorker requires at least one input for the operation.")

        operation = inputs[0]

        if operation == 'add':
            if len(inputs) < 2:
                raise ValueError("The 'add' operation requires text for the memory.")
            text = inputs[1]
            tags = inputs[2] if len(inputs) > 2 else []
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "text": text,
                "tags": tags,
            }
            _memory_store.append(entry)
            _save_memory()
            return entry
        
        elif operation == 'search':
            if len(inputs) < 2:
                raise ValueError("The 'search' operation requires a query string.")
            query = inputs[1].lower()
            top_k = inputs[2] if len(inputs) > 2 else 5
            
            matches = [
                m for m in reversed(_memory_store) 
                if query in m['text'].lower() or any(query in t.lower() for t in m.get('tags', []))
            ]
            return matches[:top_k]

        elif operation == 'get_recent':
            n = inputs[1] if len(inputs) > 1 else 5
            return list(reversed(_memory_store))[:n]

        else:
            raise ValueError(f"Unknown operation: {operation}. Available operations: 'add', 'search', 'get_recent'.")
