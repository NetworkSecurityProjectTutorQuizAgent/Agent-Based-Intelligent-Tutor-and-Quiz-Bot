"""
Security & Privacy Middleware for RAG System
Implements rate limiting, input sanitization, PII detection, and audit logging
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


# Configuration
RATE_LIMIT_REQUESTS = 50  # Max requests per window
RATE_LIMIT_WINDOW = 60  # Window in seconds
MAX_INPUT_LENGTH = 5000  # Max characters per request
AUDIT_LOG_PATH = Path("logs/audit.log")
SECURITY_LOG_PATH = Path("logs/security.log")

# PII Patterns
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
}

# SQL Injection patterns
SQL_INJECTION_PATTERNS = [
    r"(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+",
    r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER)\s+",
    r"--\s*$",
    r"/\*.*\*/",
    r"\bUNION\b.*\bSELECT\b",
    r"'\s*(OR|AND)\s*'",
]

# XSS patterns
XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",
    r"<iframe",
    r"<object",
    r"<embed",
]


class RateLimiter:
    """Rate limiting to prevent abuse."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clients: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limit."""
        now = time.time()
        
        # Remove old requests outside the window
        self.clients[client_id] = [
            req_time for req_time in self.clients[client_id]
            if now - req_time < self.window_seconds
        ]
        
        # Check if under limit
        if len(self.clients[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.clients[client_id].append(now)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        return max(0, self.max_requests - len(self.clients[client_id]))