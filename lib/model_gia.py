# -*- coding: utf-8 -*-
"""SE373 · Buổi 03 · Model giả lập, tương thích LangChain.

Đây KHÔNG phải LLM. Nó là một BaseChatModel trả tool_calls theo kịch bản,
để create_agent chạy được trên giảng đường khi không có API key và để mỗi
lần chiếu đều ra đúng một kết quả.

Ba kịch bản, khớp với ba thứ cần chỉ cho sinh viên thấy:

    "chuan"     tìm điểm đến trước, rồi hỏi thời tiết, rồi trả lời
                → vòng lặp agent bình thường, dừng khi model thôi gọi tool

    "bo_tool"   hỏi thẳng thời tiết bằng tên tự nhớ, không gọi search
                → có tool không có nghĩa là agent sẽ dùng tool (S31)

    "lap"       hỏi thời tiết một địa danh không có trong dữ liệu, đổi cách
                viết tên rồi quay lại cách cũ
                → lặp không tiến bộ (S36, S57)

Đổi model thật: xem ham `model_that()` ở cuối file.
"""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# Năm cách viết tên mà agent thử đi thử lại trong kịch bản "lap".
# Phần tử 3 lặp lại phần tử 0: đó chính là vòng V4 trên slide.
BIEN_THE_TEN = ["Vung Tau", "Vũng Tàu", "Vung Tau City", "Vung Tau", "Ba Ria - Vung Tau"]


class ModelGia(BaseChatModel):
    """Model giả lập sinh tool_calls theo kịch bản.

    Tham số:
        kich_ban   "chuan" · "bo_tool" · "lap"
        so_diem    hỏi thời tiết tối đa bao nhiêu điểm đến (kịch bản "chuan")
    """

    kich_ban: str = "chuan"
    so_diem: int = 3
    _luot: int = 0

    @property
    def _llm_type(self) -> str:
        return "se373-model-gia"

    # LangChain gọi hàm này khi agent khai báo tool. Model giả lập không cần
    # schema tool, nên chỉ trả về chính nó.
    def bind_tools(self, tools: Any, **kwargs: Any) -> "ModelGia":
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

    # ------------------------------------------------------------ kịch bản
    def _quyet_dinh(self, messages: list[BaseMessage]) -> AIMessage:
        if self.kich_ban == "lap":
            ten = BIEN_THE_TEN[(self._luot - 1) % len(BIEN_THE_TEN)]
            return self._goi("weather_forecast", {"town": ten})

        quan_sat = [self._doc(m.content) for m in messages if m.type == "tool"]
        diem_den, thoi_tiet = [], []
        for obs in quan_sat:
            if not isinstance(obs, dict):
                continue
            if "results" in obs:
                diem_den += [r["town"] for r in obs["results"]]
            elif obs.get("status") == "ok" and "weather" in obs:
                thoi_tiet.append(obs)

        if self.kich_ban == "bo_tool" and not thoi_tiet:
            # Tên điểm đến lấy từ trí nhớ của model, không từ tool.
            return AIMessage(
                content="",
                tool_calls=[
                    self._mot("weather_forecast", {"town": t}, f"call_w{i}")
                    for i, t in enumerate(["Đà Nẵng", "Nha Trang"], 1)
                ],
            )

        if not diem_den and not thoi_tiet:
            return self._goi("search_travel_info", {"query": "bãi biển đẹp Nam Trung Bộ"})

        da_hoi = {
            c["args"].get("town")
            for m in messages
            for c in (getattr(m, "tool_calls", None) or [])
            if c["name"] == "weather_forecast"
        }
        con_lai = [t for t in diem_den[: self.so_diem] if t not in da_hoi]
        if con_lai:
            return AIMessage(
                content="",
                tool_calls=[
                    self._mot("weather_forecast", {"town": t}, f"call_w{i}")
                    for i, t in enumerate(con_lai, 1)
                ],
            )

        return AIMessage(content=self._tra_loi(thoi_tiet))

    @staticmethod
    def _tra_loi(thoi_tiet: list[dict]) -> str:
        if not thoi_tiet:
            return "Chưa lấy được dữ liệu thời tiết nào."
        nang = [w for w in thoi_tiet if w["weather"].startswith("nắng")] or thoi_tiet
        dong = " · ".join(f"{w['town']} {w['weather']} {w['temperature']}°C" for w in nang)
        return f"Nên đi: {dong}."

    # ------------------------------------------------------------ tiện ích
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
