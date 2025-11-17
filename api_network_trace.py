"""
Network Trace API - Captures and explains data flow throughout the RAG pipeline
Monitors request/response flow, embedding generation, vector search, and LLM interactions
Now includes HTTP-level packet capture for protocol analysis
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from fastapi import HTTPException
from pydantic import BaseModel

#Captures HTTP packets (method, URL, headers, body preview, status codes) and logs them to disk.

# Configuration
TRACE_LOG_PATH = Path("logs/network_trace.log")
HTTP_PACKET_LOG_PATH = Path("logs/http_packets.log")
MAX_TRACES = 100  # Keep last 100 traces in memory

#Records internal events (e.g., embedding lookup, vector search, LLM call) with timestamps, durations, sizes, and metadata.

class TraceEvent(BaseModel):
    """Individual trace event in the data flow."""
    timestamp: str
    event_type: str
    component: str
    duration_ms: Optional[float] = None
    data_size_bytes: Optional[int] = None
    metadata: Dict = {}


class HTTPPacket(BaseModel):
    """HTTP-level packet capture."""
    timestamp: str
    direction: str  # "request" or "response"
    method: Optional[str] = None
    url: Optional[str] = None
    status_code: Optional[int] = None
    headers: Dict[str, str] = {}
    body_preview: str = ""
    body_size_bytes: int = 0
    protocol: str = "HTTP/1.1"

#Full pipeline traces->network trace log
#http_packets.log → Raw HTTP capture

class NetworkTrace(BaseModel):
    """Complete network trace for a request."""
    trace_id: str
    start_time: str
    end_time: Optional[str] = None
    total_duration_ms: Optional[float] = None
    endpoint: str
    method: str
    status: str
    events: List[TraceEvent] = []
    http_packets: List[HTTPPacket] = []
    summary: Dict = {}


class NetworkTracer:
    """Captures and analyzes network traces for the RAG pipeline."""
    
    def __init__(self):
        self.traces: Dict[str, NetworkTrace] = {}
        self.trace_log_path = TRACE_LOG_PATH
        self.http_packet_log_path = HTTP_PACKET_LOG_PATH
        self.trace_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.http_packet_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def start_trace(self, trace_id: str, endpoint: str, method: str) -> None:
        """Start a new network trace."""
        trace = NetworkTrace(
            trace_id=trace_id,
            start_time=datetime.utcnow().isoformat(),
            endpoint=endpoint,
            method=method,
            status='in_progress'
        )
        self.traces[trace_id] = trace
        
        # Add initial event
        self.add_event(
            trace_id,
            'REQUEST_RECEIVED',
            'API_Gateway',
            metadata={'endpoint': endpoint, 'method': method}
        )
    
    def capture_http_request(self, trace_id: str, method: str, url: str, 
                            headers: Dict[str, str], body: str = "") -> None:
        """Capture HTTP request packet details."""
        if trace_id not in self.traces:
            return
        
        # Truncate body for preview (first 500 chars)
        body_preview = body[:500] if body else ""
        if len(body) > 500:
            body_preview += "... [truncated]"
        
        packet = HTTPPacket(
            timestamp=datetime.utcnow().isoformat(),
            direction="request",
            method=method,
            url=url,
            headers=headers,
            body_preview=body_preview,
            body_size_bytes=len(body.encode('utf-8')) if body else 0,
            protocol="HTTP/1.1"
        )
        
        self.traces[trace_id].http_packets.append(packet)
        
        # Log to file immediately
        self._write_http_packet_log(trace_id, packet)
        
        # Print to console
        print(f"\n📦 HTTP REQUEST CAPTURED")
        print(f"   Trace ID: {trace_id}")
        print(f"   Method: {method}")
        print(f"   URL: {url}")
        print(f"   Headers: {len(headers)} headers")
        print(f"   Body Size: {len(body.encode('utf-8')) if body else 0} bytes")
        print(f"   Content-Type: {headers.get('content-type', 'N/A')}")
        if body_preview:
            print(f"   Body Preview: {body_preview[:100]}...")
        print()
    
    def capture_http_response(self, trace_id: str, status_code: int,
                             headers: Dict[str, str], body: str = "") -> None:
        """Capture HTTP response packet details."""
        if trace_id not in self.traces:
            return
        
        # Truncate body for preview
        body_preview = body[:500] if body else ""
        if len(body) > 500:
            body_preview += "... [truncated]"
        
        packet = HTTPPacket(
            timestamp=datetime.utcnow().isoformat(),
            direction="response",
            status_code=status_code,
            headers=headers,
            body_preview=body_preview,
            body_size_bytes=len(body.encode('utf-8')) if body else 0,
            protocol="HTTP/1.1"
        )
        
        self.traces[trace_id].http_packets.append(packet)
        
        # Log to file immediately
        self._write_http_packet_log(trace_id, packet)
        
        # Print to console
        print(f"\n📦 HTTP RESPONSE CAPTURED")
        print(f"   Trace ID: {trace_id}")
        print(f"   Status Code: {status_code}")
        print(f"   Headers: {len(headers)} headers")
        print(f"   Body Size: {len(body.encode('utf-8')) if body else 0} bytes")
        print(f"   Content-Type: {headers.get('content-type', 'N/A')}")
        print()
    
    def _write_http_packet_log(self, trace_id: str, packet: HTTPPacket) -> None:
        """Write HTTP packet to log file."""
        try:
            log_entry = {
                'trace_id': trace_id,
                'packet': packet.model_dump()
            }
            with open(self.http_packet_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Error writing HTTP packet log: {e}")
    
    def add_event(self, trace_id: str, event_type: str, component: str,
                  duration_ms: Optional[float] = None,
                  data_size_bytes: Optional[int] = None,
                  metadata: Optional[Dict] = None) -> None:
        """Add an event to the trace."""
        if trace_id not in self.traces:
            return
        
        event = TraceEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            component=component,
            duration_ms=duration_ms,
            data_size_bytes=data_size_bytes,
            metadata=metadata or {}
        )
        self.traces[trace_id].events.append(event)
    
    def end_trace(self, trace_id: str, status: str = 'completed') -> None:
        """End a trace and calculate summary."""
        if trace_id not in self.traces:
            return
        
        trace = self.traces[trace_id]
        trace.end_time = datetime.utcnow().isoformat()
        trace.status = status
        
        # Calculate total duration
        start = datetime.fromisoformat(trace.start_time)
        end = datetime.fromisoformat(trace.end_time)
        trace.total_duration_ms = (end - start).total_seconds() * 1000
        
        # Generate summary
        trace.summary = self._generate_summary(trace)
        
        # Write to log
        self._write_trace_log(trace)
        
        # Clean up old traces (keep only last MAX_TRACES)
        if len(self.traces) > MAX_TRACES:
            oldest_traces = sorted(self.traces.keys())[:len(self.traces) - MAX_TRACES]
            for old_trace_id in oldest_traces:
                del self.traces[old_trace_id]
    
    def get_trace(self, trace_id: str) -> Optional[NetworkTrace]:
        """Get a specific trace."""
        return self.traces.get(trace_id)
    
    def get_recent_traces(self, limit: int = 10) -> List[NetworkTrace]:
        """Get recent traces."""
        sorted_traces = sorted(
            self.traces.values(),
            key=lambda t: t.start_time,
            reverse=True
        )
        return sorted_traces[:limit]
    
    def _generate_summary(self, trace: NetworkTrace) -> Dict:
        """Generate summary statistics for a trace."""
        summary = {
            'total_events': len(trace.events),
            'components': defaultdict(int),
            'event_types': defaultdict(int),
            'total_data_transferred_bytes': 0,
            'breakdown_ms': {},
            'pipeline_stages': []
        }
        
        for event in trace.events:
            summary['components'][event.component] += 1
            summary['event_types'][event.event_type] += 1
            
            if event.data_size_bytes:
                summary['total_data_transferred_bytes'] += event.data_size_bytes
            
            if event.duration_ms:
                summary['breakdown_ms'][event.event_type] = event.duration_ms
        
        # Identify pipeline stages
        pipeline_stages = []
        for event in trace.events:
            stage = {
                'stage': event.event_type,
                'component': event.component,
                'timestamp': event.timestamp,
                'duration_ms': event.duration_ms
            }
            pipeline_stages.append(stage)
        
        summary['pipeline_stages'] = pipeline_stages
        summary['components'] = dict(summary['components'])
        summary['event_types'] = dict(summary['event_types'])
        
        return summary
    
    def _write_trace_log(self, trace: NetworkTrace) -> None:
        """Write trace to log file."""
        try:
            with open(self.trace_log_path, 'a', encoding='utf-8') as f:
                f.write(trace.model_dump_json() + '\n')
        except Exception as e:
            print(f"Error writing trace log: {e}")
    
    def get_analytics(self) -> Dict:
        """Get analytics across all traces."""
        if not self.traces:
            return {
                'total_traces': 0,
                'avg_duration_ms': 0,
                'total_data_transferred_mb': 0,
                'endpoint_stats': {},
                'component_stats': {}
            }
        
        total_duration = 0
        total_data = 0
        endpoint_counts = defaultdict(int)
        component_counts = defaultdict(int)
        
        for trace in self.traces.values():
            if trace.total_duration_ms:
                total_duration += trace.total_duration_ms
            endpoint_counts[trace.endpoint] += 1
            
            for event in trace.events:
                component_counts[event.component] += 1
                if event.data_size_bytes:
                    total_data += event.data_size_bytes
        
        return {
            'total_traces': len(self.traces),
            'avg_duration_ms': total_duration / len(self.traces) if self.traces else 0,
            'total_data_transferred_mb': total_data / (1024 * 1024),
            'endpoint_stats': dict(endpoint_counts),
            'component_stats': dict(component_counts)
        }


# Global tracer instance
_tracer: Optional[NetworkTracer] = None


def get_tracer() -> NetworkTracer:
    """Get or create the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = NetworkTracer()
    return _tracer


def explain_trace(trace: NetworkTrace) -> str:
    """Generate human-readable explanation of the trace."""
    if not trace:
        return "No trace data available."
    
    explanation = f"""
=== Network Trace Analysis ===
Trace ID: {trace.trace_id}
Endpoint: {trace.method} {trace.endpoint}
Status: {trace.status}
Total Duration: {trace.total_duration_ms:.2f}ms
Total Events: {len(trace.events)}

=== Data Flow Pipeline ===
"""
    
    for i, event in enumerate(trace.events, 1):
        explanation += f"\n{i}. {event.event_type} ({event.component})"
        if event.duration_ms:
            explanation += f" - {event.duration_ms:.2f}ms"
        if event.data_size_bytes:
            explanation += f" - {event.data_size_bytes} bytes"
        if event.metadata:
            explanation += f"\n   Details: {event.metadata}"
    
    explanation += f"""

=== Summary ===
Total Data Transferred: {trace.summary.get('total_data_transferred_bytes', 0)} bytes
Components Involved: {', '.join(trace.summary.get('components', {}).keys())}

=== Performance Breakdown ===
"""
    
    breakdown = trace.summary.get('breakdown_ms', {})
    for stage, duration in breakdown.items():
        percentage = (duration / trace.total_duration_ms * 100) if trace.total_duration_ms else 0
        explanation += f"- {stage}: {duration:.2f}ms ({percentage:.1f}%)\n"
    
    return explanation


# Request/Response Models for API endpoints
class TraceQueryRequest(BaseModel):
    trace_id: str


class TraceListResponse(BaseModel):
    traces: List[NetworkTrace]
    total: int


class AnalyticsResponse(BaseModel):
    analytics: Dict

