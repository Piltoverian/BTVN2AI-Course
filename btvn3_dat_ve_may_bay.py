# -*- coding: utf-8 -*-
"""BTVN#3 · Agent đặt vé máy bay với 4 lớp Harness và 3 mẫu thiết kế.

Chạy thử:
    python3 demo2-sv/btvn3_dat_ve_may_bay.py --mau react
    python3 demo2-sv/btvn3_dat_ve_may_bay.py --mau plan-execute
    python3 demo2-sv/btvn3_dat_ve_may_bay.py --mau lai
    python3 demo2-sv/btvn3_dat_ve_may_bay.py --mau tat-ca
"""
import argparse
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, ToolMessage

from lib.harness import (
    LoopDetector,
    QUYEN_MAC_DINH,
    QuyenHan,
    RANG_BUOC_MAC_DINH,
    RangBuocDatVe,
    ban_giao,
    kiem_quyen_tool,
    kiem_tieu_chi_hoan_thanh,
    prompt_tu_rang_buoc,
)
from lib.model_vemaybay import ModelGiaVeMayBay
from lib.tools_vemaybay import DON_DAT_VE, tools_vemaybay_langchain

CAU_HOI = (
    "Đặt giúp tôi 1 vé máy bay từ Hà Nội (HAN) đi Đà Nẵng (DAD) ngày 06/09 "
    "cho hành khách Nguyen Van A, ưu tiên chuyến giá tốt và đúng quy định."
)


class HarnessDatVeMiddleware(AgentMiddleware):
    def __init__(self, rb: RangBuocDatVe, quyen: QuyenHan):
        super().__init__()
        self.rb = rb
        self.quyen = quyen
        self.det = LoopDetector(window=6, repeat_k=2)
        self.da_thu: list[str] = []
        self.so_lan_chan_quyen = 0
        self.ket_qua_kiem_tra: dict | None = None

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        loi_quyen = kiem_quyen_tool(call["name"], call["args"], self.rb, self.quyen)
        if loi_quyen:
            self.so_lan_chan_quyen += 1
            return ToolMessage(
                tool_call_id=call["id"],
                content=json.dumps({"status": "blocked", "error": loi_quyen}, ensure_ascii=False),
            )
        return handler(request)

    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime):
        cuoi = state["messages"][-1]
        tool_calls = getattr(cuoi, "tool_calls", None)

        if tool_calls:
            for c in tool_calls:
                canh_bao = self.det.check(c["name"], c["args"], progress=0)
                self.da_thu.append(f"{c['name']}({c['args']})")
                if canh_bao:
                    bao_cao = ban_giao(
                        ly_do=canh_bao,
                        da_thu=self.da_thu,
                        trang_thai={"so_lan_goi_tool": len(self.da_thu), "chan_quyen": self.so_lan_chan_quyen},
                        cau_hoi="Agent bị lặp hành động. Chuyển nhân viên hỗ trợ xử lý tiếp?",
                    )
                    self.ket_qua_kiem_tra = {"hoan_thanh": False, "ban_giao": bao_cao}
                    return {"jump_to": "end", "messages": [AIMessage(content=_in_ban_giao(bao_cao))]}
            return None

        ket_qua_tool = [m.content for m in state["messages"] if m.type == "tool"]
        kq = kiem_tieu_chi_hoan_thanh(str(cuoi.content), ket_qua_tool, self.rb)
        self.ket_qua_kiem_tra = kq
        if not kq["hoan_thanh"]:
            bao_cao = ban_giao(
                ly_do="CHƯA ĐẠT TIÊU CHÍ HOÀN THÀNH · " + "; ".join(kq["loi"]),
                da_thu=self.da_thu,
                trang_thai={"so_lan_goi_tool": len(self.da_thu), "chan_quyen": self.so_lan_chan_quyen},
                cau_hoi="Kế hoạch đặt vé gặp sự cố (hết ghế/vượt ràng buộc). Bạn muốn chọn chuyến thay thế hay đổi ngày?",
            )
            self.ket_qua_kiem_tra["ban_giao"] = bao_cao
            return {"jump_to": "end", "messages": [AIMessage(content=_in_ban_giao(bao_cao))]}
        return None


def _in_ban_giao(b: dict) -> str:
    return (
        "DỪNG BẤT THƯỜNG · " + b["stop_reason"] + "\n"
        "  Đã thử          : " + " → ".join(b["da_thu"]) + "\n"
        "  Trạng thái      : " + str(b["trang_thai"]) + "\n"
        "  Hỏi người dùng  : " + b["cau_hoi_cho_nguoi"]
    )


def chay(mau: str) -> dict:
    DON_DAT_VE.clear()
    rb = RANG_BUOC_MAC_DINH
    quyen = QUYEN_MAC_DINH
    model = ModelGiaVeMayBay(kich_ban=mau, rb=rb)
    tools = tools_vemaybay_langchain()
    mw = HarnessDatVeMiddleware(rb=rb, quyen=quyen)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt_tu_rang_buoc(rb),
        middleware=[mw],
    )

    print("=" * 78)
    print(f"BTVN#3 · mẫu thiết kế: {mau}")
    print("=" * 78)

    kq = agent.invoke({"messages": [{"role": "user", "content": CAU_HOI}]})
    _in_phien(kq["messages"])
    print("-" * 78)

    hoan_thanh = bool(mw.ket_qua_kiem_tra and mw.ket_qua_kiem_tra.get("hoan_thanh"))
    booking_id = (
        mw.ket_qua_kiem_tra["booking"]["booking_id"]
        if hoan_thanh and mw.ket_qua_kiem_tra and mw.ket_qua_kiem_tra.get("booking")
        else "BÀN GIAO"
    )
    return {
        "mau": mau,
        "so_luot_model": model._luot,
        "so_luot_tool": len(mw.da_thu),
        "chan_quyen": mw.so_lan_chan_quyen,
        "hoan_thanh": hoan_thanh,
        "ket_qua": booking_id,
    }


def _in_phien(messages) -> None:
    vong = 0
    for m in messages:
        if m.type == "human":
            print(f'[V1] Human  "{m.content}"')
        elif m.type == "ai" and getattr(m, "tool_calls", None):
            vong += 1
            for c in m.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
                print(f"[V{vong}] AI     {c['name']}({args})")
        elif m.type == "tool":
            print(f"[V{vong}] Tool   {m.content}")
        elif m.type == "ai":
            for dong in str(m.content).splitlines():
                print(f"[V{vong}] AI     {dong}")


def in_bang_danh_gia(thong_ke: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("BẢNG ĐÁNH GIÁ HIỆU QUẢ 3 MẪU THIẾT KẾ AGENT (YÊU CẦU 3)")
    print("=" * 78)
    print(f"{'Mẫu thiết kế':<16} | {'Lượt Model':^10} | {'Lượt Tool':^10} | {'Chặn Quyền/RB':^13} | {'Hoàn thành':^10} | {'Kết quả':<10}")
    print("-" * 78)
    for t in thong_ke:
        tt = "ĐẠT" if t["hoan_thanh"] else "KHÔNG"
        print(
            f"{t['mau']:<16} | {t['so_luot_model']:^10} | {t['so_luot_tool']:^10} | "
            f"{t['chan_quyen']:^13} | {tt:^10} | {t['ket_qua']:<10}"
        )
    print("=" * 78)


def main() -> None:
    p = argparse.ArgumentParser(description="BTVN#3 · Agent đặt vé máy bay")
    p.add_argument(
        "--mau",
        default="tat-ca",
        choices=["react", "plan-execute", "lai", "lap", "tat-ca"],
    )
    chon = p.parse_args().mau
    if chon == "tat-ca":
        tk = [chay(m) for m in ("react", "plan-execute", "lai")]
        in_bang_danh_gia(tk)
    else:
        tk = [chay(chon)]
        in_bang_danh_gia(tk)


if __name__ == "__main__":
    sys.exit(main())
