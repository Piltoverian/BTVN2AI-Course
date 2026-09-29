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

try:
    from lib.harness import RANG_BUOC_MAC_DINH, kiem_rang_buoc
except ImportError:
    from harness import RANG_BUOC_MAC_DINH, kiem_rang_buoc

BIEN_THE_SAN_BAY = ["Vung Tau", "Vũng Tàu", "VTG", "Vung Tau"]


class ModelGiaVeMayBay(BaseChatModel):
    """Model giả lập cho Agent đặt vé máy bay.

    Tham số kich_ban:
        "react"         suy luận từng bước theo observation thực tế
        "plan-execute"  lập kế hoạch chọn chuyến đầu tiên, không re-plan khi hết ghế
        "lai"           lập kế hoạch có lọc ràng buộc + tự re-plan sang chuyến dự phòng
        "lap"           tìm sân bay không có trong dữ liệu, lặp lại cách viết ở V4
    """

    kich_ban: str = "react"
    rb: dict = RANG_BUOC_MAC_DINH
    _luot: int = 0

    @property
    def _llm_type(self) -> str:
        return f"se373-model-vemaybay-{self.kich_ban}"

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

        da_goi_tool = [
            (c["name"], c["args"])
            for m in messages
            for c in (getattr(m, "tool_calls", None) or [])
        ]
        da_kiem_tra = {
            args.get("flight_id")
            for ten, args in da_goi_tool
            if ten in ("get_flight_detail", "hold_booking")
        }
        da_thu_xuat_ve = any(ten == "issue_ticket" for ten, _ in da_goi_tool)

        if don_giu_cho:
            if self.kich_ban == "react" and not da_thu_xuat_ve:
                return self._goi("issue_ticket", {"booking_id": don_giu_cho["booking_id"]})
            return AIMessage(content=self._tra_loi(don_giu_cho))

        for fid, f in chi_tiet_ok.items():
            return self._goi("hold_booking", {
                "flight_id": fid,
                "passenger_name": "Nguyen Van A",
                "price": f["price"],
                "cabin": f["cabin"],
                "origin": f["origin"],
                "destination": f["destination"],
                "dep_time": f["dep_time"],
            })

        if self.kich_ban == "plan-execute":
            muc_tieu = ds_chuyen[0]
            if muc_tieu["flight_id"] not in da_kiem_tra:
                return self._goi("get_flight_detail", {"flight_id": muc_tieu["flight_id"]})
            return AIMessage(
                content=(
                    f"Chuyến {muc_tieu['flight_id']} lúc {muc_tieu['dep_time']} "
                    f"giá {muc_tieu['price_str']} đã hết ghế, kế hoạch cố định không thể hoàn tất."
                )
            )

        if self.kich_ban == "lai":
            ung_vien = [f for f in ds_chuyen if not kiem_rang_buoc(f, self.rb)]
            ung_vien.sort(key=lambda x: x["price"])
        else:
            ung_vien = sorted(ds_chuyen, key=lambda x: x["price"], reverse=True)

        con_lai = [f for f in ung_vien if f["flight_id"] not in da_kiem_tra]
        if con_lai:
            tiep = con_lai[0]
            if self.kich_ban == "react" and tiep["price"] > self.rb["ngan_sach_toi_da"]:
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

        return AIMessage(content="Không tìm được chuyến bay nào còn ghế thỏa mãn yêu cầu.")

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
    from langchain.chat_models import init_chat_model

    ten = os.environ.get("SE373_MODEL")
    if not ten:
        raise SystemExit("Chưa đặt biến môi trường SE373_MODEL trong .env")
    return init_chat_model(ten)
