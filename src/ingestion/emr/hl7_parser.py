"""
G_One_Sync AI — HL7v2 Message Parser
========================================
Parses HL7v2 messages (ADT^A01, ORU^R01) into structured Python dicts.
Uses hl7apy for robust HL7v2 parsing with segment-level extraction.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

# ── HL7v2 Constants ──────────────────────────────────────────────────

# MLLP framing bytes
MLLP_START_BYTE = b"\x0b"
MLLP_END_BYTES = b"\x1c\x0d"

# Message types we handle
SUPPORTED_MESSAGE_TYPES = {"ORU^R01", "ADT^A01", "ADT^A03"}


class HL7ParseError(Exception):
    """Raised when HL7v2 message parsing fails."""


class HL7Parser:
    """
    Parse HL7v2 messages into structured data.

    Handles:
        - ORU^R01: Observation results (vitals, labs)
        - ADT^A01: Patient admission
        - ADT^A03: Patient discharge
    """

    @staticmethod
    def strip_mllp_framing(raw: bytes) -> str:
        """
        Remove MLLP framing bytes from raw message.

        MLLP format: <SB>message<EB><CR>
        where SB = 0x0B, EB = 0x1C, CR = 0x0D

        Args:
            raw: Raw bytes from TCP socket

        Returns:
            Decoded HL7v2 message string
        """
        message = raw
        if message.startswith(MLLP_START_BYTE):
            message = message[1:]
        if message.endswith(MLLP_END_BYTES):
            message = message[:-2]
        elif message.endswith(b"\x1c"):
            message = message[:-1]
        return message.decode("utf-8", errors="replace").strip()

    @staticmethod
    def parse_message(raw_message: str) -> dict[str, Any]:
        """
        Parse an HL7v2 message into a structured dict.

        Args:
            raw_message: HL7v2 message string (pipe-delimited)

        Returns:
            Dict with 'message_type', 'segments', and parsed content

        Raises:
            HL7ParseError: If message format is invalid
        """
        try:
            from hl7apy.exceptions import HL7apyException
            from hl7apy.parser import parse_message as hl7_parse

            msg = hl7_parse(raw_message, find_groups=False)

            # Extract MSH segment for message type
            msh = msg.segment("MSH")
            message_type = str(msh.msh_9.msh_9_1.value)
            trigger_event = str(msh.msh_9.msh_9_2.value)
            full_type = f"{message_type}^{trigger_event}"

            result: dict[str, Any] = {
                "message_type": full_type,
                "message_control_id": str(msh.msh_10.value) if msh.msh_10 else "",
                "sending_facility": str(msh.msh_4.value) if msh.msh_4 else "",
                "timestamp": str(msh.msh_7.value) if msh.msh_7 else "",
            }

            if full_type == "ORU^R01":
                result["observations"] = HL7Parser.parse_oru_r01(msg)
            elif full_type == "ADT^A01":
                result["patient"] = HL7Parser.parse_adt_a01(msg)
            elif full_type == "ADT^A03":
                result["patient"] = HL7Parser.parse_adt_a01(msg)
                result["event"] = "discharge"
            else:
                logger.warning("Unsupported HL7v2 message type: {}", full_type)

            return result

        except ImportError:
            # Fallback: manual pipe-delimited parsing
            return HL7Parser._parse_manual(raw_message)
        except Exception as e:
            raise HL7ParseError(f"Failed to parse HL7v2 message: {e}") from e

    @staticmethod
    def parse_oru_r01(msg: Any) -> list[dict[str, Any]]:
        """
        Extract OBX segments from an ORU^R01 message.

        Each OBX segment contains a single observation (vital sign or lab result).

        Args:
            msg: Parsed HL7v2 message object

        Returns:
            List of observation dicts with 'code', 'value', 'unit', 'status'
        """
        observations = []

        try:
            # Extract patient ID from PID segment
            pid = msg.segment("PID")
            patient_id = str(pid.pid_3.value) if pid.pid_3 else ""
        except Exception:
            patient_id = ""

        try:
            obx_segments = msg.segments("OBX")
        except Exception:
            obx_segments = []

        for obx in obx_segments:
            try:
                obs: dict[str, Any] = {
                    "patient_id": patient_id,
                    "set_id": str(obx.obx_1.value) if obx.obx_1 else "",
                    "value_type": str(obx.obx_2.value) if obx.obx_2 else "",
                    "observation_identifier": "",
                    "observation_code": "",
                    "value": None,
                    "unit": "",
                    "status": str(obx.obx_11.value) if obx.obx_11 else "F",
                }

                # OBX-3: Observation Identifier (format: code^display^system)
                if obx.obx_3:
                    identifier = str(obx.obx_3.value)
                    parts = identifier.split("^")
                    obs["observation_code"] = parts[0] if parts else ""
                    obs["observation_identifier"] = parts[1] if len(parts) > 1 else parts[0]

                # OBX-5: Observation Value
                if obx.obx_5:
                    raw_value = str(obx.obx_5.value)
                    try:
                        obs["value"] = float(raw_value)
                    except ValueError:
                        obs["value"] = raw_value

                # OBX-6: Units
                if obx.obx_6:
                    obs["unit"] = str(obx.obx_6.value).split("^")[0]

                observations.append(obs)

            except Exception as e:
                logger.warning("Failed to parse OBX segment: {}", e)

        logger.debug("Parsed {} OBX segments from ORU^R01", len(observations))
        return observations

    @staticmethod
    def parse_adt_a01(msg: Any) -> dict[str, Any]:
        """
        Extract patient information from an ADT^A01 message.

        Reads PID (Patient Identification) and PV1 (Patient Visit) segments.

        Args:
            msg: Parsed HL7v2 message object

        Returns:
            Dict with patient demographics
        """
        patient: dict[str, Any] = {
            "patient_id": "",
            "gender": "",
            "birth_date": "",
            "admission_type": "ED",
        }

        try:
            pid = msg.segment("PID")

            # PID-3: Patient Identifier List
            if pid.pid_3:
                patient["patient_id"] = str(pid.pid_3.value).split("^")[0]

            # PID-7: Date of Birth (YYYYMMDD)
            if pid.pid_7:
                dob = str(pid.pid_7.value)
                if len(dob) >= 8:
                    patient["birth_date"] = f"{dob[:4]}-{dob[4:6]}-{dob[6:8]}"

            # PID-8: Sex
            if pid.pid_8:
                sex_val = str(pid.pid_8.value).upper()
                patient["gender"] = "M" if sex_val == "M" else "F"

        except Exception as e:
            logger.warning("Failed to parse PID segment: {}", e)

        try:
            pv1 = msg.segment("PV1")

            # PV1-4: Admission Type
            if pv1.pv1_4:
                adm_type = str(pv1.pv1_4.value).upper()
                if adm_type in ("E", "EMERGENCY"):
                    patient["admission_type"] = "ED"
                elif adm_type in ("EL", "ELECTIVE"):
                    patient["admission_type"] = "Elective"
                elif adm_type in ("T", "TRANSFER"):
                    patient["admission_type"] = "Transfer"

        except Exception as e:
            logger.debug("PV1 segment not found or parse error: {}", e)

        return patient

    @staticmethod
    def _parse_manual(raw_message: str) -> dict[str, Any]:
        """
        Fallback manual parser when hl7apy is not installed.
        Uses simple pipe/tilde delimited parsing.
        """
        segments = raw_message.strip().split("\r")
        if not segments:
            segments = raw_message.strip().split("\n")

        result: dict[str, Any] = {
            "message_type": "UNKNOWN",
            "segments": [],
            "observations": [],
        }

        for seg_str in segments:
            fields = seg_str.split("|")
            seg_type = fields[0] if fields else ""

            if seg_type == "MSH" and len(fields) >= 10:
                msg_type = fields[8] if len(fields) > 8 else ""
                result["message_type"] = msg_type.replace("^", "^")
                result["message_control_id"] = fields[9] if len(fields) > 9 else ""

            elif seg_type == "OBX" and len(fields) >= 6:
                code_field = fields[3] if len(fields) > 3 else ""
                code_parts = code_field.split("^")
                try:
                    value = float(fields[5]) if len(fields) > 5 else None
                except (ValueError, IndexError):
                    value = fields[5] if len(fields) > 5 else None

                result["observations"].append({
                    "observation_code": code_parts[0] if code_parts else "",
                    "observation_identifier": code_parts[1] if len(code_parts) > 1 else "",
                    "value": value,
                    "unit": fields[6].split("^")[0] if len(fields) > 6 else "",
                    "status": fields[11] if len(fields) > 11 else "F",
                })

            elif seg_type == "PID" and len(fields) >= 9:
                result["patient"] = {
                    "patient_id": fields[3].split("^")[0] if len(fields) > 3 else "",
                    "birth_date": fields[7] if len(fields) > 7 else "",
                    "gender": fields[8] if len(fields) > 8 else "",
                }

        return result

    @staticmethod
    def build_ack(
        message_control_id: str,
        ack_code: str = "AA",
        text_message: str = "Message accepted",
    ) -> str:
        """
        Build an HL7v2 ACK message.

        Args:
            message_control_id: Original message control ID (MSH-10)
            ack_code: AA (accepted), AE (error), AR (rejected)
            text_message: Human-readable status message

        Returns:
            HL7v2 ACK message string
        """
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

        ack = (
            f"MSH|^~\\&|JEEVANSYNC|JEEVANSYNC|||||ACK^A01|{timestamp}|P|2.5\r"
            f"MSA|{ack_code}|{message_control_id}|{text_message}\r"
        )
        return ack
