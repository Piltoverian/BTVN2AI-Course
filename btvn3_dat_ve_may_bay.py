# -*- coding: utf-8 -*-
r"""BTVN#3 · Agent đặt vé máy bay với 4 lớp Harness và 3 mẫu thiết kế.

Model giả (ModelGiaVeMayBay) và Model thật (model_that) dùng chung một
lớp HarnessDatVeMiddleware điều phối cả 4 lớp Harness lẫn 3 mẫu thiết kế.

Chạy thử:
    .\demo2-sv\.venv\Scripts\python.exe demo2-sv/btvn3_dat_ve_may_bay.py --mau tat-ca
    .\demo2-sv\.venv\Scripts\python.exe demo2-sv/btvn3_dat_ve_may_bay.py --mau tat-ca --that
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
    kiem_rang_buoc,
    kiem_tieu_chi_hoan_thanh,
    prompt_tu_rang_buoc,
)
from lib.model_vemaybay import ModelGiaVeMayBay, model_that
from lib.tools_vemaybay import DON_DAT_VE, tools_vemaybay_langchain

CAU_HOI = (
    "Đặt giúp tôi 1 vé máy bay từ Hà Nội (HAN) đi Đà Nẵng (DAD) ngày 06/09 "
    "cho hành khách Nguyen Van A, ưu tiên chuyến giá tốt và đúng quy định."
)


class HarnessDatVeMiddleware(AgentMiddleware):
    """Middleware dùng chung cho cả ModelGiaVeMayBay và model_that().

    Kết hợp:
      1. Điều phối 3 mẫu thiết kế (react, plan-execute, lai)
      2. Kiểm quyền & Ràng buộc dữ liệu trước khi chạy tool (wrap_tool_call)
      3. Phát hiện lặp, Kiểm tra tiêu chí hoàn thành bằng code & Bàn giao (after_model)
    """

    def __init__(self, mau: str, rb: RangBuocDatVe, quyen: QuyenHan):
        super().__init__()
        self.mau = mau
        self.rb = rb
        self.quyen = quyen
        self.det = LoopDetector(window=6, repeat_k=2)
        self.da_thu: list[str] = []
        self.so_luot_model = 0
        self.so_lan_chan_quyen = 0
        self.chuyen_ke_hoach: str | None = None
        self.ket_qua_kiem_tra: dict | None = None

    def wrap_tool_call(self, request, handler):
        call = request.tool_call
        ten = call["name"]
        args = call["args"]

        if self.mau == "plan-execute" and ten in ("get_flight_detail", "hold_booking"):
            fid = args.get("flight_id")
            if self.chuyen_ke_hoach and fid != self.chuyen_ke_hoach:
                return ToolMessage(
                    tool_call_id=call["id"],
                    content=json.dumps(
                        {
                            "status": "blocked",
                            "error": f"PLAN_LOCKED · Mẫu plan-execute chỉ thực thi theo kế hoạch cố định ({self.chuyen_ke_hoach}), không tự đổi sang {fid}",
                        },
                        ensure_ascii=False,
                    ),
                )

        loi_quyen = kiem_quyen_tool(ten, args, self.rb, self.quyen)
        if loi_quyen:
            self.so_lan_chan_quyen += 1
            return ToolMessage(
                tool_call_id=call["id"],
                content=json.dumps({"status": "blocked", "error": loi_quyen}, ensure_ascii=False),
            )

        kq_tool = handler(request)
        if ten == "search_flights" and isinstance(kq_tool, ToolMessage):
            return self._dieu_phoi_ke_hoach(kq_tool)
        return kq_tool

    def _dieu_phoi_ke_hoach(self, msg: ToolMessage) -> ToolMessage:
        try:
            data = json.loads(str(msg.content))
        except Exception:
            return msg
        if "flights" not in data:
            return msg

        ds = data["flights"]
        if self.mau == "plan-execute":
            # Plan-then-Execute: Chốt cứng kế hoạch chỉ gồm chuyến đầu tiên, không Re-plan
            chon = ds[:1]
            if chon:
                self.chuyen_ke_hoach = chon[0]["flight_id"]
            data["flights"] = chon
            data["plan_mode"] = f"PLAN_FIXED: chỉ thực thi trên {self.chuyen_ke_hoach}"
        elif self.mau == "lai":
            # Lai (Hybrid): Lọc trước bằng Ràng buộc dữ liệu + xếp thứ tự ưu tiên để Re-plan khi hết ghế
            hop_le = [f for f in ds if not kiem_rang_buoc(f, self.rb)]
            hop_le.sort(key=lambda x: x["price"])
            data["flights"] = hop_le
            data["plan_mode"] = "HYBRID_PLAN: đã lọc theo ràng buộc dữ liệu và sắp xếp ứng viên dự phòng"
        else:
            # ReAct: Không lập kế hoạch lọc ràng buộc trước, thử tuần tự cả chuyến vi phạm lẫn chuyến hết ghế
            data["flights"] = sorted(ds, key=lambda x: (x["cabin"] == "Economy", x["price"]))
            data["plan_mode"] = "REACT_RAW: chưa lọc ràng buộc trước, tự thử-sai từng bước"

        return ToolMessage(
            tool_call_id=msg.tool_call_id,
            content=json.dumps(data, ensure_ascii=False),
        )

    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime):
        self.so_luot_model += 1
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


def prompt_theo_mau(mau: str, rb: RangBuocDatVe) -> str:
    goc = prompt_tu_rang_buoc(rb)
    if mau == "plan-execute":
        return goc + " Chế độ Plan-then-Execute: chỉ thực thi trên chuyến bay đầu tiên trong kế hoạch, không tự đổi kế hoạch."
    if mau == "lai":
        return goc + " Chế độ Lai (Hybrid): kiểm tra lần lượt danh sách ứng viên đã lọc theo ràng buộc, nếu hết ghế thì chuyển sang ứng viên dự phòng tiếp theo."
    return goc + " Chế độ ReAct: suy luận và gọi tool từng bước dựa trên observation."


def chay(mau: str, dung_model_that: bool = False) -> dict:
    DON_DAT_VE.clear()
    rb = RANG_BUOC_MAC_DINH
    quyen = QUYEN_MAC_DINH

    if dung_model_that:
        model = model_that()
    else:
        kich_ban = "lap" if mau == "lap" else "chuan"
        model = ModelGiaVeMayBay(kich_ban=kich_ban)

    tools = tools_vemaybay_langchain()
    mw = HarnessDatVeMiddleware(mau=mau, rb=rb, quyen=quyen)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt_theo_mau(mau, rb),
        middleware=[mw],
    )

    print("=" * 78)
    print(f"BTVN#3 · mẫu thiết kế: {mau} · model: {'THẬT' if dung_model_that else 'GIẢ LẬP'}")
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
        "so_luot_model": mw.so_luot_model,
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
    p.add_argument("--that", action="store_true", help="Chạy với model thật từ biến SE373_MODEL")
    args = p.parse_args()
    if args.mau == "tat-ca":
        tk = [chay(m, dung_model_that=args.that) for m in ("react", "plan-execute", "lai")]
        in_bang_danh_gia(tk)
    else:
        tk = [chay(args.mau, dung_model_that=args.that)]
        in_bang_danh_gia(tk)


if __name__ == "__main__":
    sys.exit(main())
