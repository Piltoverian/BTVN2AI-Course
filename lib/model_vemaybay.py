# -*- coding: utf-8 -*-
"""BTVN#3 · Model giả lập đặt vé máy bay, tương thích LangChain."""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

BIEN_THE_SAN_BAY = ["Vung Tau", "Vũng Tàu", "VTG", "Vung Tau"]


class ModelGiaVeMayBay(BaseChatModel):
    """Model giả lập chỉ thực hiện gọi tool dựa trên observation trong messages.
    Không phân nhánh theo mẫu thiết kế; dùng chung Middleware với model_that().
    """

    kich_ban: str = "chuan"
    _luot: int = 0

    @property
    def _llm_type(self) -> str:
        return "se373-model-vemaybay"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ModelGiaVeMayBay":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._luot += 1
        ai = self._quyet_dinh(messages)
        return ChatResult(generations=[ChatGeneration(message=ai)])

    def _quyet_dinh(self, messages: list[BaseMessage]) -> AIMessage:
        if self.kich_ban == "lap":
            dst = BIEN_THE_SAN_BAY[(self._luot - 1) % len(BIEN_THE_SAN_BAY)]
            return self._goi("search_flights", {"origin": "HAN", "destination": dst, "date": "06/09"})

        quan_sat = [self._doc(m.content) for m in messages if m.type == "tool"]
        ds_chuyen, chi_tiet_ok, don_giu_cho = [], {}, None

        for obs in quan_sat:
            if not isinstance(obs, dict):
                continue
            if "flights" in obs:
                ds_chuyen = obs["flights"]
            elif obs.get("status") == "ok" and "seats_left" in obs:
                chi_tiet_ok[obs["flight_id"]] = obs
            elif obs.get("status") in ("HELD", "ISSUED") and "booking_id" in obs:
                don_giu_cho = obs

        if not ds_chuyen:
            return self._goi("search_flights", {"origin": "HAN", "destination": "DAD", "date": "06/09"})

        if don_giu_cho:
            return AIMessage(content=self._tra_loi(don_giu_cho))

        da_thu_hold = {
            c["args"].get("flight_id")
            for m in messages
            for c in (getattr(m, "tool_calls", None) or [])
            if c["name"] == "hold_booking"
        }

        for fid, f in chi_tiet_ok.items():
            if fid not in da_thu_hold:
                return self._goi("hold_booking", {
                    "flight_id": fid,
                    "passenger_name": "Nguyen Van A",
                    "price": f["price"],
                    "cabin": f["cabin"],
                    "origin": f["origin"],
                    "destination": f["destination"],
                    "dep_time": f["dep_time"],
                })

        da_kiem_tra = {
            c["args"].get("flight_id")
            for m in messages
            for c in (getattr(m, "tool_calls", None) or [])
            if c["name"] in ("get_flight_detail", "hold_booking")
        }

        con_lai = [f for f in ds_chuyen if f["flight_id"] not in da_kiem_tra]
        if con_lai:
            tiep = con_lai[0]
            if tiep["cabin"] != "Economy":
                return self._goi("hold_booking", {
                    "flight_id": tiep["flight_id"],
                    "passenger_name": "Nguyen Van A",
                    "price": tiep["price"],
                    "cabin": tiep["cabin"],
                    "origin": tiep["origin"],
                    "destination": tiep["destination"],
                    "dep_time": tiep["dep_time"],
                })
            return self._goi("get_flight_detail", {"flight_id": tiep["flight_id"]})

        return AIMessage(content="Không còn chuyến bay nào khả dụng trong kế hoạch.")

    @staticmethod
    def _tra_loi(don: dict) -> str:
        return (
            f"Đã giữ chỗ mã {don['booking_id']} trên chuyến {don['flight_id']} "
            f"ngày {don['date']} lúc {don['dep_time']}, giá {don['price_str']}."
        )

    def _goi(self, ten: str, args: dict) -> AIMessage:
        return AIMessage(content="", tool_calls=[self._mot(ten, args, f"call_{self._luot}")])

    @staticmethod
    def _mot(ten: str, args: dict, cid: str) -> dict:
        return {"name": ten, "args": args, "id": cid, "type": "tool_call"}

    @staticmethod
    def _doc(content: Any) -> Any:
        try:
            return json.loads(content)
        except Exception:
            return content


def model_that():
    """Model thật, đọc SE373_MODEL trong .env. Chỉ dùng khi có API key.

    Ví dụ SE373_MODEL: "openai:gpt-4.1-mini" · "anthropic:claude-sonnet-4-5"
    """
    from langchain.chat_models import init_chat_model

    ten = os.environ.get("SE373_MODEL")
    if not ten:
        raise SystemExit(
            "Chưa đặt SE373_MODEL. Copy .env.example thành .env, điền model và khoá API,\n"
            "rồi chạy:  set -a && source .env && set +a"
        )
    return init_chat_model(ten)
