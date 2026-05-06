"""
Google Calendar API Tool — FREE tier.

Books, lists, and cancels appointments by writing real Google Calendar events.
Uses OAuth2 (user consent flow) — credentials are cached after first authorization.

SETUP (one-time):
  1. Go to https://console.cloud.google.com/
  2. Create a project → Enable "Google Calendar API"
  3. APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client IDs
     → Application type: Desktop app → Download JSON
  4. Save the downloaded file as:  backend/data/calendar/credentials.json
  5. Set GOOGLE_CALENDAR_ID in .env (use "primary" for your main calendar)
  6. On first tool call, a browser window opens for one-time Google authorization.
     The token is cached in  backend/data/calendar/token.json  for all future calls.

ENV VARS:
  GOOGLE_CALENDAR_ID   — calendar to write to (default: "primary")
  GOOGLE_TIMEZONE      — timezone for events (default: "America/New_York")
"""
import os
import json
import logging
import asyncio
import pathlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from tools.base import BaseTool

logger = logging.getLogger("medical_bot")

CALENDAR_DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "calendar"
CREDENTIALS_FILE = CALENDAR_DATA_DIR / "credentials.json"
TOKEN_FILE = CALENDAR_DATA_DIR / "token.json"
SCOPES = ["https://www.googleapis.com/auth/calendar"]

GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")
GOOGLE_TIMEZONE = os.getenv("GOOGLE_TIMEZONE", "America/New_York")

_service_cache = None   # Cached Google API service object


def _is_configured() -> bool:
    """Return True only if OAuth credentials file is present."""
    return CREDENTIALS_FILE.exists()


def _get_calendar_service():
    """
    Build and return the Google Calendar service, loading/refreshing credentials.
    Runs the OAuth browser flow on first call if no token.json exists.
    Raises RuntimeError if credentials.json is missing.
    """
    global _service_cache
    if _service_cache is not None:
        return _service_cache

    if not _is_configured():
        raise RuntimeError(
            "Google Calendar not configured. "
            f"Place your OAuth credentials at: {CREDENTIALS_FILE}\n"
            "Setup guide: https://developers.google.com/calendar/api/quickstart/python"
        )

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise RuntimeError(
            "Google API libraries not installed. "
            "Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
        )

    creds = None
    CALENDAR_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("Google Calendar token refreshed")
            except Exception as e:
                logger.warning(f"Token refresh failed: {e} — re-authorizing")
                creds = None

        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0, prompt="consent")
            logger.info("Google Calendar OAuth flow completed")

        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    _service_cache = build("calendar", "v3", credentials=creds)
    return _service_cache


def _parse_datetime(dt_str: str, fallback_days: int = 1) -> datetime:
    """
    Parse a natural-language or ISO date string into a datetime object.
    Falls back to (now + fallback_days) if parsing fails.
    """
    dt_str = dt_str.strip().lower()
    now = datetime.now()

    # Relative keywords
    if dt_str in ("today",):
        return now.replace(hour=9, minute=0, second=0, microsecond=0)
    if dt_str in ("tomorrow",):
        return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)

    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    }
    for day_name, day_num in day_map.items():
        if day_name in dt_str:
            days_ahead = (day_num - now.weekday() + 7) % 7 or 7
            return (now + timedelta(days=days_ahead)).replace(hour=9, minute=0, second=0, microsecond=0)

    # Try standard ISO / common formats
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(dt_str, fmt).replace(
                hour=9, minute=0, second=0, microsecond=0
            )
        except ValueError:
            continue

    return (now + timedelta(days=fallback_days)).replace(hour=9, minute=0, second=0, microsecond=0)


def _format_event_time(dt_iso: str) -> str:
    """Format an ISO datetime string to a human-readable format."""
    try:
        dt = datetime.fromisoformat(dt_iso.replace("Z", "+00:00"))
        return dt.strftime("%A, %B %d at %I:%M %p")
    except Exception:
        return dt_iso


# ── Tool classes ─────────────────────────────────────────────────────────────

class CreateCalendarEventTool(BaseTool):
    name = "create_calendar_event"
    description = (
        "Book a medical appointment by creating a real event in Google Calendar. "
        "Use this when the patient wants to schedule a follow-up visit, check-up, "
        "physical therapy session, medication review, or any clinic appointment. "
        "Requires Google Calendar to be configured."
    )
    parameters = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Title of the appointment. E.g., 'Post-Op Follow-Up — Dr. Johnson'.",
            },
            "date": {
                "type": "string",
                "description": "Date of the appointment. E.g., 'next Monday', 'tomorrow', '2025-03-15'.",
            },
            "time": {
                "type": "string",
                "description": "Appointment time. E.g., '10:00 AM', '2:30 PM'. Default: 9:00 AM.",
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Duration of the appointment in minutes. Default: 30.",
                "default": 30,
            },
            "description": {
                "type": "string",
                "description": "Additional notes for the calendar event (reason for visit, doctor name, location, etc.).",
            },
            "send_reminder": {
                "type": "boolean",
                "description": "Whether to add an email + popup reminder (24h before). Default: true.",
                "default": True,
            },
        },
        "required": ["summary", "date"],
    }

    async def execute(
        self,
        summary: str,
        date: str,
        time: str = "9:00 AM",
        duration_minutes: int = 30,
        description: str = "",
        send_reminder: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:

        if not _is_configured():
            return {
                "success": False,
                "error": "Google Calendar not configured.",
                "setup_instructions": (
                    "To enable Google Calendar:\n"
                    "1. Visit https://console.cloud.google.com/\n"
                    "2. Create a project, enable 'Google Calendar API'\n"
                    "3. Create OAuth 2.0 credentials (Desktop app)\n"
                    f"4. Save credentials.json to: {CREDENTIALS_FILE}\n"
                    "5. Set GOOGLE_CALENDAR_ID=primary in .env\n"
                    "6. Restart the server — the first tool call will open a browser for authorization."
                ),
            }

        try:
            start_dt = _parse_datetime(date)

            # Parse time override
            for fmt in ("%I:%M %p", "%H:%M", "%I %p"):
                try:
                    t = datetime.strptime(time.strip().upper(), fmt.replace("%p", "%p"))
                    start_dt = start_dt.replace(hour=t.hour, minute=t.minute)
                    break
                except ValueError:
                    continue

            end_dt = start_dt + timedelta(minutes=duration_minutes)
            tz = GOOGLE_TIMEZONE

            event_body = {
                "summary": summary,
                "description": description or f"Medical appointment scheduled via Nurse GPT-E.\nDate requested: {date}",
                "start": {"dateTime": start_dt.isoformat(), "timeZone": tz},
                "end": {"dateTime": end_dt.isoformat(), "timeZone": tz},
            }

            if send_reminder:
                event_body["reminders"] = {
                    "useDefault": False,
                    "overrides": [
                        {"method": "email", "minutes": 24 * 60},
                        {"method": "popup", "minutes": 30},
                    ],
                }

            service = await asyncio.to_thread(_get_calendar_service)
            event = await asyncio.to_thread(
                lambda: service.events()
                .insert(calendarId=GOOGLE_CALENDAR_ID, body=event_body)
                .execute()
            )

            return {
                "success": True,
                "event_id": event.get("id"),
                "summary": summary,
                "start": _format_event_time(event["start"]["dateTime"]),
                "end": _format_event_time(event["end"]["dateTime"]),
                "calendar_link": event.get("htmlLink"),
                "message": (
                    f"Appointment '{summary}' has been added to Google Calendar. "
                    f"Scheduled for: {_format_event_time(event['start']['dateTime'])}. "
                    f"View it here: {event.get('htmlLink', 'N/A')}"
                ),
            }

        except RuntimeError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Google Calendar create error: {e}")
            return {"success": False, "error": f"Failed to create calendar event: {e}"}


class ListCalendarEventsTool(BaseTool):
    name = "list_calendar_events"
    description = (
        "List upcoming medical appointments from Google Calendar. "
        "Use this when the patient asks 'what appointments do I have?' or "
        "'when is my next check-up?' or 'show me my upcoming schedule'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "max_results": {
                "type": "integer",
                "description": "Maximum number of upcoming events to return (default 5).",
                "default": 5,
            },
            "days_ahead": {
                "type": "integer",
                "description": "Look ahead this many days (default 30).",
                "default": 30,
            },
        },
        "required": [],
    }

    async def execute(self, max_results: int = 5, days_ahead: int = 30, **kwargs) -> Dict[str, Any]:
        if not _is_configured():
            return {
                "found": False,
                "message": "Google Calendar is not configured. See setup instructions.",
            }

        try:
            now_iso = datetime.utcnow().isoformat() + "Z"
            max_time = (datetime.utcnow() + timedelta(days=days_ahead)).isoformat() + "Z"

            service = await asyncio.to_thread(_get_calendar_service)
            events_result = await asyncio.to_thread(
                lambda: service.events()
                .list(
                    calendarId=GOOGLE_CALENDAR_ID,
                    timeMin=now_iso,
                    timeMax=max_time,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])
            if not events:
                return {
                    "found": False,
                    "message": f"No upcoming appointments found in the next {days_ahead} days.",
                }

            formatted = []
            for ev in events:
                start = ev["start"].get("dateTime", ev["start"].get("date", ""))
                formatted.append({
                    "event_id": ev.get("id"),
                    "summary": ev.get("summary", "(No title)"),
                    "start": _format_event_time(start) if "T" in start else start,
                    "description": (ev.get("description", "")[:100] or ""),
                    "link": ev.get("htmlLink", ""),
                })

            return {
                "found": True,
                "event_count": len(formatted),
                "events": formatted,
                "calendar": GOOGLE_CALENDAR_ID,
            }

        except RuntimeError as e:
            return {"found": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Google Calendar list error: {e}")
            return {"found": False, "error": f"Failed to list events: {e}"}


class DeleteCalendarEventTool(BaseTool):
    name = "delete_calendar_event"
    description = (
        "Cancel or delete an appointment from Google Calendar by its event ID. "
        "Use this when the patient wants to cancel a scheduled appointment. "
        "The event ID is returned when an appointment is created or listed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "event_id": {
                "type": "string",
                "description": "The Google Calendar event ID to delete (returned by create_calendar_event or list_calendar_events).",
            },
        },
        "required": ["event_id"],
    }

    async def execute(self, event_id: str, **kwargs) -> Dict[str, Any]:
        if not _is_configured():
            return {"success": False, "message": "Google Calendar is not configured."}

        try:
            service = await asyncio.to_thread(_get_calendar_service)
            await asyncio.to_thread(
                lambda: service.events()
                .delete(calendarId=GOOGLE_CALENDAR_ID, eventId=event_id)
                .execute()
            )
            return {
                "success": True,
                "event_id": event_id,
                "message": f"Appointment (event ID: {event_id}) has been successfully cancelled from Google Calendar.",
            }

        except RuntimeError as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Google Calendar delete error: {e}")
            return {"success": False, "error": f"Failed to delete event: {e}"}
