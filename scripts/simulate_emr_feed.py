"""
G_One_Sync AI — EMR Data Feed Simulator
=========================================
Clinical simulator for testing production EMR interfaces.
Emits HL7v2 MLLP streams and FHIR R4 payloads to test end-to-end
ingestion pipelines with realistic physiological trajectories.

Usage:
    # Send 5 hours of deteriorating patient vitals via HL7 MLLP
    python scripts/simulate_emr_feed.py --protocol hl7 --mode deteriorating --count 5

    # Send stable patient vitals via FHIR Webhook
    python scripts/simulate_emr_feed.py --protocol fhir --mode stable --count 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402
from loguru import logger  # noqa: E402

from src.ingestion.emr.hl7_parser import MLLP_END_BYTES, MLLP_START_BYTE, HL7Parser  # noqa: E402

# ── Physiological trajectory generator ───────────────────────────────


def _noise(scale: float) -> float:
    """Generate Gaussian noise."""
    return random.gauss(0, scale)


def generate_vitals_trajectory(
    hour: int,
    mode: str = "stable",
    base_hr: float = 75.0,
    base_spo2: float = 98.0,
    base_rr: float = 16.0,
    base_sbp: float = 120.0,
    base_dbp: float = 80.0,
    base_temp: float = 37.0,
) -> dict[str, float]:
    """Generate realistic vital signs for a given hour."""
    if mode == "deteriorating":
        # Simulating sepsis / septic shock deterioration over time
        progress = min(1.0, hour / 12.0)
        hr = base_hr + progress * 55.0 + _noise(2.0)          # 75 -> 130 BPM
        spo2 = base_spo2 - progress * 12.0 + _noise(1.0)      # 98 -> 86%
        rr = base_rr + progress * 18.0 + _noise(1.5)          # 16 -> 34 breaths/min
        sbp = base_sbp - progress * 40.0 + _noise(3.0)        # 120 -> 80 mmHg (hypotension)
        dbp = base_dbp - progress * 25.0 + _noise(2.0)        # 80 -> 55 mmHg
        temp = base_temp + progress * 2.2 + _noise(0.2)       # 37 -> 39.2 °C (fever)
    else:
        # Stable ICU patient with physiological variation
        hr = base_hr + _noise(3.0)
        spo2 = min(100.0, base_spo2 + _noise(0.8))
        rr = base_rr + _noise(1.0)
        sbp = base_sbp + _noise(4.0)
        dbp = base_dbp + _noise(3.0)
        temp = base_temp + _noise(0.15)

    return {
        "heart_rate": round(max(30.0, min(220.0, hr)), 1),
        "respiratory_rate": round(max(8.0, min(50.0, rr)), 1),
        "spo2_pct": round(max(70.0, min(100.0, spo2)), 1),
        "temperature_c": round(max(34.0, min(42.0, temp)), 2),
        "systolic_bp": round(max(60.0, min(220.0, sbp)), 1),
        "diastolic_bp": round(max(35.0, min(140.0, dbp)), 1),
    }


def generate_labs_trajectory(hour: int, mode: str = "stable") -> dict[str, float]:
    """Generate realistic laboratory test results."""
    if mode == "deteriorating":
        progress = min(1.0, hour / 12.0)
        wbc = 9.0 + progress * 14.0 + _noise(0.5)      # Leukocytosis (9 -> 23)
        lactate = 1.2 + progress * 4.5 + _noise(0.2)   # Lactic acidosis (1.2 -> 5.7)
        creatinine = 0.9 + progress * 1.8 + _noise(0.1)# Acute kidney injury (0.9 -> 2.7)
        crp = 10.0 + progress * 120.0 + _noise(5.0)   # Systemic inflammation
        hgb = 13.5 - progress * 2.5 + _noise(0.3)
    else:
        wbc = 7.5 + _noise(0.5)
        lactate = 1.1 + _noise(0.1)
        creatinine = 0.9 + _noise(0.05)
        crp = 8.0 + _noise(1.5)
        hgb = 13.5 + _noise(0.2)

    return {
        "wbc_count": round(max(1.0, wbc), 2),
        "lactate": round(max(0.4, lactate), 2),
        "creatinine": round(max(0.2, creatinine), 2),
        "crp_level": round(max(1.0, crp), 2),
        "hemoglobin": round(max(5.0, hgb), 2),
    }


# ── HL7v2 MLLP Sender ────────────────────────────────────────────────


def build_hl7_oru_message(
    patient_id: int,
    vitals: dict[str, float],
    control_id: str,
) -> str:
    """Construct standard HL7v2 ORU^R01 observation message."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    lines = [
        f"MSH|^~\\&|ICU_BEDSIDE|HOSPITAL|||{ts}||ORU^R01|{control_id}|P|2.5",
        f"PID|||{patient_id}^^^HOSP||PATIENT^{patient_id}||19750101|M",
        f"OBX|1|NM|8867-4^Heart Rate^LN||{vitals['heart_rate']}|/min|||||F",
        f"OBX|2|NM|9279-1^Respiratory Rate^LN||{vitals['respiratory_rate']}|/min|||||F",
        f"OBX|3|NM|2708-6^SpO2^LN||{vitals['spo2_pct']}|%|||||F",
        f"OBX|4|NM|8310-5^Temperature^LN||{vitals['temperature_c']}|Cel|||||F",
        f"OBX|5|NM|8480-6^Systolic BP^LN||{vitals['systolic_bp']}|mmHg|||||F",
        f"OBX|6|NM|8462-4^Diastolic BP^LN||{vitals['diastolic_bp']}|mmHg|||||F",
    ]
    return "\r".join(lines) + "\r"


def send_hl7_message(host: str, port: int, message: str) -> str:
    """Send an MLLP-framed HL7 message via TCP socket and return ACK."""
    mllp_data = MLLP_START_BYTE + message.encode("utf-8") + MLLP_END_BYTES

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(5.0)
        sock.connect((host, port))
        sock.sendall(mllp_data)

        # Receive ACK
        response = b""
        while MLLP_END_BYTES not in response:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk

        ack_text = HL7Parser.strip_mllp_framing(response)
        return ack_text


# ── FHIR Webhook Sender ──────────────────────────────────────────────


async def send_fhir_notification(
    api_url: str,
    patient_id: int,
    subscription_id: str = "sub-icu-monitor",
) -> dict:
    """Send a FHIR Subscription notification webhook."""
    payload = {
        "subscriptionId": subscription_id,
        "resourceType": "Observation",
        "resourceId": f"obs-vitals-{patient_id}",
        "patientReference": f"Patient/{patient_id}",
        "eventType": "create",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{api_url}/emr/fhir/webhook", json=payload)
        return {"status_code": resp.status_code, "body": resp.json() if resp.status_code == 200 else resp.text}


# ── CLI Runner ───────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="G_One_Sync AI — EMR Feed Simulator")
    parser.add_argument("--protocol", choices=["hl7", "fhir"], default="hl7", help="Protocol to simulate")
    parser.add_argument("--mode", choices=["stable", "deteriorating"], default="deteriorating", help="Patient trajectory")
    parser.add_argument("--patient-id", type=int, default=1001, help="Synthetic patient ID")
    parser.add_argument("--count", type=int, default=5, help="Number of hourly readings to emit")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between hourly emissions")
    parser.add_argument("--hl7-host", default="127.0.0.1", help="HL7 MLLP server host")
    parser.add_argument("--hl7-port", type=int, default=2575, help="HL7 MLLP server port")
    parser.add_argument("--fhir-api-url", default="http://127.0.0.1:8000", help="FastAPI ingest API URL")

    args = parser.parse_args()

    logger.info(
        "Starting EMR Feed Simulation | protocol={} | mode={} | patient={} | hours={}",
        args.protocol.upper(), args.mode, args.patient_id, args.count,
    )

    for hour in range(1, args.count + 1):
        vitals = generate_vitals_trajectory(hour=hour, mode=args.mode)
        labs = generate_labs_trajectory(hour=hour, mode=args.mode)

        logger.info(
            "Hour {}/{} | HR={:.0f} | SpO2={:.0f}% | RR={:.0f} | BP={:.0f}/{:.0f} | Lactate={:.1f} | WBC={:.1f}",
            hour, args.count,
            vitals["heart_rate"], vitals["spo2_pct"], vitals["respiratory_rate"],
            vitals["systolic_bp"], vitals["diastolic_bp"], labs["lactate"], labs["wbc_count"],
        )

        if args.protocol == "hl7":
            ctrl_id = f"SIM_{args.patient_id}_{hour:03d}"
            hl7_msg = build_hl7_oru_message(args.patient_id, vitals, ctrl_id)
            try:
                ack = send_hl7_message(args.hl7_host, args.hl7_port, hl7_msg)
                status = "ACCEPTED" if "MSA|AA|" in ack else "REJECTED"
                logger.success("HL7 Sent: {} -> MLLP Response: {}", ctrl_id, status)
            except ConnectionRefusedError:
                logger.error("Could not connect to HL7 MLLP server at {}:{}", args.hl7_host, args.hl7_port)
                logger.info("Tip: Start the HL7 server with: python -m src.ingestion.emr.hl7_listener")
                break
            except Exception as e:
                logger.error("Error sending HL7 message: {}", e)

        elif args.protocol == "fhir":
            try:
                res = asyncio.run(send_fhir_notification(args.fhir_api_url, args.patient_id))
                logger.success("FHIR Webhook response: status={} result={}", res["status_code"], res.get("body"))
            except Exception as e:
                logger.error("Error sending FHIR notification: {}", e)
                logger.info("Tip: Start the Ingestion API with JEEVAN_EMR_ENABLED=true")
                break

        if hour < args.count:
            time.sleep(args.delay)

    logger.info("Simulation completed.")


if __name__ == "__main__":
    main()
