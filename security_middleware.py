"""
Security & Privacy Middleware for RAG System
Implements rate limiting, input sanitization, PII detection, and audit logging
"""
#This set of imports provides the essential tools needed for logging, text processing, hashing, structured data, 
#and file management inside our backend or trace system.
from __future__ import annotations
#Provides hashing algorithms (MD5, SHA256, etc.)
import hashlib
#Regular expressions for pattern matching
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
#JSON serialization/conversion
#Saving trace logs
import json
#These imports are typically used inside a security / logging middleware, for example:
#Authenticating requests,#Blocking unauthorized access.#Logging request/response info,#Adding headers / enforcing policies
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

#These settings defend your FastAPI RAG system by limiting request volume, restricting input size, 
#and storing audit + security logs for accountability and threat monitoring.

# Configuration
RATE_LIMIT_REQUESTS = 50  # Max requests per window Prevents Dos Attack
RATE_LIMIT_WINDOW = 60  # Window in seconds
MAX_INPUT_LENGTH = 5000  # Max characters per request Input Validation
AUDIT_LOG_PATH = Path("logs/audit.log") # FOR accountability
SECURITY_LOG_PATH = Path("logs/security.log")

#This dictionary defines regular expression patterns used to detect sensitive personal information (PII) 
#in user input before it reaches your system.

# PII Patterns 
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
}
#is a set of regex signatures designed to detect common SQL injection payloads—such as boolean logic exploits, 
#stacked queries, UNION injections,
#and comment-based attacks—before malicious input reaches your database.
# SQL Injection patterns
SQL_INJECTION_PATTERNS = [
    r"(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+",
    r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER)\s+",
    r"--\s*$",
    r"/\*.*\*/",
    r"\bUNION\b.*\bSELECT\b",
    r"'\s*(OR|AND)\s*'",
]

# XSS patterns  such JS injections 
XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",
    r"<iframe",
    r"<object",
    r"<embed",
]

#prevents users (clients) from sending too many requests in a short period, which protects your API from: DDoS attacks
#Bot spamming
#Brute-force attempts
#Overuse of free endpoints

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
    #This method returns how many requests a client can still make before hitting the rate limit.
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        return max(0, self.max_requests - len(self.clients[client_id]))

#This function scans input text and detects any PII (Personally Identifiable Information) such as emails, 
#phone numbers, SSNs, credit cards, IP addresses, etc., based on regex rules
class SecurityValidator:
    """Validates and sanitizes inputs for security threats."""
    
    @staticmethod
    def detect_pii(text: str) -> List[Dict[str, str]]:
        """Detect potential PII in text."""
        detections = []
        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                detections.append({
                    'type': pii_type,
                    'value': match.group(),
                    'position': match.span()
                })
        return detections
#This function removes or masks sensitive personal information (PII) in user input before that text is:
    
    @staticmethod
    def sanitize_pii(text: str) -> str:
        """Remove/mask PII from text."""
        sanitized = text
        for pii_type, pattern in PII_PATTERNS.items():
            if pii_type == 'email':
                sanitized = re.sub(pattern, '[EMAIL_REDACTED]', sanitized, flags=re.IGNORECASE)
            elif pii_type == 'phone':
                sanitized = re.sub(pattern, '[PHONE_REDACTED]', sanitized, flags=re.IGNORECASE)
            elif pii_type == 'ssn':
                sanitized = re.sub(pattern, '[SSN_REDACTED]', sanitized, flags=re.IGNORECASE)
            elif pii_type == 'credit_card':
                sanitized = re.sub(pattern, '[CARD_REDACTED]', sanitized, flags=re.IGNORECASE)
        return sanitized

    #Scans input text for SQL injection patterns using regular expressions.
    
    @staticmethod
    def detect_sql_injection(text: str) -> bool:
        """Detect potential SQL injection attempts."""
        for pattern in SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

#Detects possible Cross-Site Scripting (XSS) attacks in input text.
    
    @staticmethod
    def detect_xss(text: str) -> bool:
        """Detect potential XSS attempts."""
        for pattern in XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

#Ensures user input does not exceed maximum allowed length.
    
    @staticmethod
    def validate_input_length(text: str, max_length: int = MAX_INPUT_LENGTH) -> bool:
        """Validate input length."""
        return len(text) <= max_length

    #mask it and send
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """General input sanitization."""
        # Remove null bytes
        sanitized = text.replace('\x00', '')
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized)
        # Strip leading/trailing whitespace
        sanitized = sanitized.strip()
        return sanitized


class AuditLogger:
    """Audit logging for security and compliance."""
    
    def __init__(self):
        self.audit_log_path = AUDIT_LOG_PATH
        self.security_log_path = SECURITY_LOG_PATH
        # Create log directories
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.security_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_request(self, client_id: str, endpoint: str, method: str, 
                   payload: Optional[Dict] = None, sanitized: bool = False):
        """Log API request for audit trail."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'client_id': client_id,
            'endpoint': endpoint,
            'method': method,
            'sanitized': sanitized,
            'payload_hash': hashlib.sha256(
                json.dumps(payload or {}, sort_keys=True).encode()
            ).hexdigest()[:16]
        }
        self._write_log(self.audit_log_path, log_entry)
    
    def log_security_event(self, event_type: str, client_id: str, 
                          details: Dict, severity: str = 'MEDIUM'):
        """Log security events."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'client_id': client_id,
            'severity': severity,
            'details': details
        }
        self._write_log(self.security_log_path, log_entry)
    
    @staticmethod
    def _write_log(log_path: Path, entry: Dict):
        """Write log entry to file."""
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')


class SecurityMiddleware(BaseHTTPMiddleware):
    """Main security middleware combining all security features."""
    
    def __init__(self, app):
        super().__init__(app)
        self.rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW)
        self.validator = SecurityValidator()
        self.logger = AuditLogger()
    
    async def dispatch(self, request: Request, call_next):
        """Process request through security checks."""
        
        # Skip security for static files and health check
        if request.url.path.startswith('/static') or request.url.path == '/health':
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Rate limiting
        if not self.rate_limiter.is_allowed(client_id):
            print(f"\n⚠️  RATE LIMIT EXCEEDED!")
            print(f"   Client: {client_id}")
            print(f"   Endpoint: {request.url.path}")
            print(f"   Action: Request BLOCKED (429)\n")
            
            self.logger.log_security_event(
                'RATE_LIMIT_EXCEEDED',
                client_id,
                {'endpoint': request.url.path},
                severity='HIGH'
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Process POST requests with body
        if request.method == "POST":
            body = await request.body()
            body_str = body.decode('utf-8')
            
            try:
                payload = json.loads(body_str) if body_str else {}
            except json.JSONDecodeError:
                payload = {}
            
            # Extract text fields for validation
            text_fields = self._extract_text_fields(payload)
            
            # Security validation
            sanitized = False
            for field_name, text in text_fields.items():
                # Length validation
                if not self.validator.validate_input_length(text):
                    print(f"\n⚠️  SECURITY: Input Too Long!")
                    print(f"   Field: {field_name}")
                    print(f"   Length: {len(text)} chars (max: {MAX_INPUT_LENGTH})")
                    print(f"   Client: {client_id}")
                    print(f"   Action: Request BLOCKED\n")
                    
                    self.logger.log_security_event(
                        'INPUT_TOO_LONG',
                        client_id,
                        {'field': field_name, 'length': len(text)},
                        severity='LOW'
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Input too long. Maximum {MAX_INPUT_LENGTH} characters."
                    )
                
                # SQL Injection detection
                if self.validator.detect_sql_injection(text):
                    print(f"\n🚨 SECURITY THREAT BLOCKED: SQL Injection Attempt!")
                    print(f"   Field: {field_name}")
                    print(f"   Client: {client_id}")
                    print(f"   Severity: CRITICAL")
                    print(f"   Action: Request BLOCKED\n")
                    
                    self.logger.log_security_event(
                        'SQL_INJECTION_ATTEMPT',
                        client_id,
                        {'field': field_name, 'endpoint': request.url.path},
                        severity='CRITICAL'
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Potentially malicious input detected."
                    )
                
                # XSS detection
                if self.validator.detect_xss(text):
                    print(f"\n🚨 SECURITY THREAT BLOCKED: XSS Attack Attempt!")
                    print(f"   Field: {field_name}")
                    print(f"   Client: {client_id}")
                    print(f"   Severity: CRITICAL")
                    print(f"   Action: Request BLOCKED\n")
                    
                    self.logger.log_security_event(
                        'XSS_ATTEMPT',
                        client_id,
                        {'field': field_name, 'endpoint': request.url.path},
                        severity='CRITICAL'
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Potentially malicious input detected."
                    )
                
                # PII detection
                pii_detections = self.validator.detect_pii(text)
                if pii_detections:
                    print(f"\n⚠️  SECURITY ALERT: PII Detected!")
                    print(f"   Field: {field_name}")
                    print(f"   Types: {[d['type'] for d in pii_detections]}")
                    print(f"   Count: {len(pii_detections)}")
                    print(f"   Action: Auto-sanitizing before processing")
                    print(f"   Client: {client_id}\n")
                    
                    self.logger.log_security_event(
                        'PII_DETECTED',
                        client_id,
                        {
                            'field': field_name,
                            'pii_types': [d['type'] for d in pii_detections],
                            'count': len(pii_detections)
                        },
                        severity='MEDIUM'
                    )
                    # Sanitize PII
                    sanitized_text = self.validator.sanitize_pii(text)
                    payload[field_name] = sanitized_text
                    sanitized = True
            
            # Log audit trail
            self.logger.log_request(
                client_id,
                request.url.path,
                request.method,
                payload,
                sanitized
            )
            
            # Recreate request with sanitized body if needed
            if sanitized:
                from starlette.datastructures import Headers
                new_body = json.dumps(payload).encode('utf-8')
                
                async def receive():
                    return {'type': 'http.request', 'body': new_body}
                
                request._receive = receive
        
        # Add security headers to response
        #Logs all activity,Adds security headers, Returns updated request to FastAPI
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        response.headers['X-RateLimit-Remaining'] = str(self.rate_limiter.get_remaining(client_id))
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier."""
        # Use IP address as client ID (in production, use more sophisticated method)
        forwarded = request.headers.get('X-Forwarded-For')
        if forwarded:
            return forwarded.split(',')[0].strip()
        return request.client.host if request.client else 'unknown'
    
    def _extract_text_fields(self, payload: Dict) -> Dict[str, str]:
        """Extract text fields from payload for validation."""
        text_fields = {}
        if 'question' in payload:
            text_fields['question'] = str(payload['question'])
        if 'user_answer' in payload:
            text_fields['user_answer'] = str(payload['user_answer'])
        if 'topic' in payload and payload['topic']:
            text_fields['topic'] = str(payload['topic'])
        return text_fields


def get_security_stats() -> Dict:
    """Get security statistics from logs."""
    stats = {
        'total_requests': 0,
        'security_events': defaultdict(int),
        'pii_detections': 0,
        'blocked_requests': 0
    }
    
    try:
        if AUDIT_LOG_PATH.exists():
            with open(AUDIT_LOG_PATH, 'r') as f:
                stats['total_requests'] = sum(1 for _ in f)
        
        if SECURITY_LOG_PATH.exists():
            with open(SECURITY_LOG_PATH, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        event_type = entry.get('event_type', 'unknown')
                        stats['security_events'][event_type] += 1
                        
                        if event_type == 'PII_DETECTED':
                            stats['pii_detections'] += 1
                        elif event_type in ['SQL_INJECTION_ATTEMPT', 'XSS_ATTEMPT', 'RATE_LIMIT_EXCEEDED']:
                            stats['blocked_requests'] += 1
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading security stats: {e}")
    
    return dict(stats)


