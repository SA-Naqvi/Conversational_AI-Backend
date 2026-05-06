"""
OpenFDA API Tool — FREE, no API key required.

Provides real-time drug information from the FDA's public database:
  - Drug labels (warnings, indications, dosage, contraindications)
  - Adverse event reports (real-world side effects)
  - Drug recall information

API docs: https://open.fda.gov/apis/
"""
import logging
import httpx
from typing import Any, Dict, List, Optional

from tools.base import BaseTool

logger = logging.getLogger("medical_bot")

OPENFDA_BASE = "https://api.fda.gov"
REQUEST_TIMEOUT = 10.0


def _clean_fda_text(text: Any) -> str:
    """FDA text fields are sometimes lists of strings; flatten and truncate."""
    if isinstance(text, list):
        combined = " ".join(str(t) for t in text)
    else:
        combined = str(text) if text else ""
    # Truncate very long FDA label text
    return combined[:800].strip() if combined else ""


async def _fda_get(endpoint: str, params: Dict) -> Optional[Dict]:
    """Make a GET request to the OpenFDA API with error handling."""
    url = f"{OPENFDA_BASE}{endpoint}"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(url, params=params)
            if resp.status_code == 404:
                return None  # Not found — not an error
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            logger.warning(f"OpenFDA timeout for {endpoint}")
            return None
        except httpx.ConnectError:
            logger.warning("OpenFDA unreachable — no internet?")
            return None
        except Exception as e:
            logger.error(f"OpenFDA error: {e}")
            return None


class DrugLabelTool(BaseTool):
    name = "search_drug_label"
    description = (
        "Look up official FDA drug label information for a medication, including "
        "its indications (what it treats), dosage instructions, warnings, "
        "contraindications, and precautions. "
        "Use this when a patient asks about a medication's official FDA information, "
        "warnings, or how to take it correctly."
    )
    parameters = {
        "type": "object",
        "properties": {
            "drug_name": {
                "type": "string",
                "description": "Generic or brand name of the drug to look up. E.g., 'ibuprofen', 'oxycodone', 'amoxicillin', 'Tylenol'.",
            },
            "sections": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional: specific sections to retrieve. "
                    "Options: 'warnings', 'dosage_and_administration', 'indications_and_usage', "
                    "'contraindications', 'adverse_reactions', 'drug_interactions', 'description'. "
                    "Default: all available sections."
                ),
            },
        },
        "required": ["drug_name"],
    }

    async def execute(
        self,
        drug_name: str,
        sections: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Try generic name search first, then brand name
        search_queries = [
            f'openfda.generic_name:"{drug_name}"',
            f'openfda.brand_name:"{drug_name}"',
            f'openfda.substance_name:"{drug_name}"',
        ]

        data = None
        for query in search_queries:
            data = await _fda_get("/drug/label.json", {"search": query, "limit": 1})
            if data and data.get("results"):
                break

        if not data or not data.get("results"):
            return {
                "found": False,
                "drug_name": drug_name,
                "message": (
                    f"No FDA label found for '{drug_name}'. "
                    "Please verify the drug name or consult a pharmacist."
                ),
            }

        label = data["results"][0]
        openfda = label.get("openfda", {})

        # Sections to return
        wanted = set(sections) if sections else {
            "warnings",
            "dosage_and_administration",
            "indications_and_usage",
            "contraindications",
            "adverse_reactions",
            "drug_interactions",
            "warnings_and_cautions",
            "boxed_warning",
        }

        result: Dict[str, Any] = {
            "found": True,
            "drug_name": drug_name,
            "generic_names": openfda.get("generic_name", []),
            "brand_names": openfda.get("brand_name", []),
            "manufacturer": openfda.get("manufacturer_name", ["Unknown"])[0]
                if openfda.get("manufacturer_name") else "Unknown",
            "source": "FDA Drug Label Database (openFDA)",
        }

        # Populate requested sections
        for section in wanted:
            raw = label.get(section)
            if raw:
                result[section] = _clean_fda_text(raw)

        # Boxed warning is high-priority — always include if present
        if label.get("boxed_warning") and "boxed_warning" not in result:
            result["boxed_warning"] = _clean_fda_text(label["boxed_warning"])

        result["disclaimer"] = (
            "This information is sourced from the official FDA drug label. "
            "Follow your prescriber's specific instructions, which may differ."
        )
        return result


class DrugAdverseEventsTool(BaseTool):
    name = "search_drug_adverse_events"
    description = (
        "Search the FDA Adverse Event Reporting System (FAERS) for real-world "
        "reported side effects and adverse reactions for a medication. "
        "Use this when a patient asks 'what side effects does X cause?' or "
        "'can this medication cause dizziness/nausea/etc.?'"
    )
    parameters = {
        "type": "object",
        "properties": {
            "drug_name": {
                "type": "string",
                "description": "Generic or brand name of the drug. E.g., 'aspirin', 'metformin', 'lisinopril'.",
            },
            "top_n": {
                "type": "integer",
                "description": "Number of top adverse reactions to return (default 10, max 20).",
                "default": 10,
            },
        },
        "required": ["drug_name"],
    }

    async def execute(
        self,
        drug_name: str,
        top_n: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        top_n = min(max(top_n, 1), 20)

        # Count most common adverse reactions reported for this drug
        data = await _fda_get(
            "/drug/event.json",
            {
                "search": f'patient.drug.openfda.generic_name:"{drug_name}"',
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": top_n,
            },
        )

        if not data or not data.get("results"):
            # Try brand name
            data = await _fda_get(
                "/drug/event.json",
                {
                    "search": f'patient.drug.openfda.brand_name:"{drug_name}"',
                    "count": "patient.reaction.reactionmeddrapt.exact",
                    "limit": top_n,
                },
            )

        if not data or not data.get("results"):
            return {
                "found": False,
                "drug_name": drug_name,
                "message": (
                    f"No adverse event data found for '{drug_name}' in the FDA database. "
                    "This could mean the drug name is spelled differently in the FDA system, "
                    "or no reports have been filed. Consult the drug's package insert for side effects."
                ),
            }

        reactions = [
            {"reaction": item["term"].title(), "reports": item["count"]}
            for item in data["results"]
        ]

        total_reports = sum(r["reports"] for r in reactions)

        return {
            "found": True,
            "drug_name": drug_name,
            "top_adverse_reactions": reactions,
            "total_reports_sampled": total_reports,
            "source": "FDA Adverse Event Reporting System (FAERS) via openFDA",
            "note": (
                "These counts reflect FDA adverse event reports. "
                "Frequency does not establish causation — many factors affect reporting rates."
            ),
            "disclaimer": (
                "Adverse event reports are submitted voluntarily and may not represent "
                "the true incidence of side effects. Consult your healthcare provider."
            ),
        }


class DrugRecallTool(BaseTool):
    name = "check_drug_recall"
    description = (
        "Check if a drug has any recent FDA recalls or enforcement actions. "
        "Use this when a patient asks if their medication has been recalled "
        "or if there are safety concerns about a specific medication."
    )
    parameters = {
        "type": "object",
        "properties": {
            "drug_name": {
                "type": "string",
                "description": "Name of the drug to check for recalls.",
            },
        },
        "required": ["drug_name"],
    }

    async def execute(self, drug_name: str, **kwargs) -> Dict[str, Any]:
        data = await _fda_get(
            "/drug/enforcement.json",
            {
                "search": f'product_description:"{drug_name}"',
                "limit": 3,
            },
        )

        if not data or not data.get("results"):
            return {
                "found": False,
                "drug_name": drug_name,
                "message": (
                    f"No recent FDA enforcement actions or recalls found for '{drug_name}'. "
                    "For the most current recall information, visit: https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts"
                ),
            }

        recalls = []
        for r in data["results"]:
            recalls.append({
                "recall_number": r.get("recall_number", "N/A"),
                "product": r.get("product_description", "")[:200],
                "reason": r.get("reason_for_recall", "")[:200],
                "classification": r.get("classification", ""),
                "status": r.get("status", ""),
                "date": r.get("recall_initiation_date", ""),
            })

        return {
            "found": True,
            "drug_name": drug_name,
            "recall_count": len(recalls),
            "recalls": recalls,
            "source": "FDA Enforcement Reports via openFDA",
            "fda_link": "https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts",
        }
