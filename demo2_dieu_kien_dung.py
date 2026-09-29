# -*- coding: utf-8 -*-
"""DEMO 2 · Điều kiện dừng  (slide 34–41)

Cùng một agent, cùng một câu hỏi, cùng một bộ tool. Chỉ đổi phần điều kiện
dừng, và kết quả đi từ "cháy tiền" sang "dừng đúng chỗ, báo đúng người".

Câu hỏi hỏi thời tiết Vũng Tàu — một địa danh KHÔNG có trong dữ liệu — và
tool đang ở chế độ observation tệ: chỉ trả '{"error": "not found"}', không
nói được phải làm gì khác.

Chạy ba chế độ theo đúng thứ tự này trên lớp:

    python3 demo2_dieu_kien_dung.py --che-do khong-gioi-han
        không có trần nào → chạy tới khi LangGraph ném GraphRecursionError

    python3 demo2_dieu_kien_dung.py --che-do ngan-sach
        ModelCallLimitMiddleware → dừng xác định, nhưng chỉ biết "hết lượt"

    python3 demo2_dieu_kien_dung.py --che-do phat-hien-lap
        LoopDetector tự viết → dừng ngay V4, nói được vì sao, bàn giao cho người
"""
import argparse
import os
import sys

# Bật observation tệ TRƯỚC khi import tool: đây là nguyên nhân gốc của vòng lặp.
os.environ.setdefault("SE373_LOI_TE", "1")

from langchain.agents import create_agent                                  # noqa: E402
from langchain.agents.middleware import (                                  # noqa: E402
    AgentMiddleware,
    ModelCallLimitMiddleware,
    hook_config,
)
from langchain_core.messages import AIMessage                              # noqa: E402

from lib.harness import LoopDetector, ban_giao                             # noqa: E402
from lib.model_gia import ModelGia                                         # noqa: E402
from lib.tools_dulich import tools_langchain                               # noqa: E402

CAU_HOI = "Ngày mai giao hàng ở Vũng Tàu có mưa không? Nếu mưa thì dời sang ngày kia."


# ============================================== lớp harness bạn phải tự viết
class PhatHienLapMiddleware(AgentMiddleware):
    """Chạy sau mỗi lần model đề xuất tool. Framework không có sẵn lớp này.

    So (tool, args) chứ không so observation: gọi lại get_booking để chờ
    confirmed là polling hợp lệ, không phải lặp (S41).
    """

    def __init__(self):
        super().__init__()
        self.det = LoopDetector(window=6, repeat_k=2)
        self.da_thu = []

    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime):
        cuoi = state["messages"][-1]
        if not getattr(cuoi, "tool_calls", None):
            return None

        for c in cuoi.tool_calls:
            canh_bao = self.det.check(c["name"], c["args"], progress=0)
            self.da_thu.append(f"{c['name']}({c['args']})")
            if canh_bao:
                bao_cao = ban_giao(
                    ly_do=canh_bao,
                    da_thu=self.da_thu,
                    trang_thai={"so_lan_goi_tool": len(self.da_thu), "ket_qua_dung": 0},
                    cau_hoi="Vũng Tàu không có trong dữ liệu thời tiết. "
                            "Dùng nguồn nào thay thế, hay trả lời là chưa hỗ trợ?",
                )
                return {
                    "jump_to": "end",
                    "messages": [AIMessage(content=_in_ban_giao(bao_cao))],
                }
        return None


def _in_ban_giao(b: dict) -> str:
    return (
        "DỪNG BẤT THƯỜNG · " + b["stop_reason"] + "\n"
        "  Đã thử          : " + " → ".join(b["da_thu"]) + "\n"
        "  Trạng thái      : " + str(b["trang_thai"]) + "\n"
        "  Hỏi người dùng  : " + b["cau_hoi_cho_nguoi"]
    )


# ================================================================ ba chế độ
def chay(che_do: str) -> None:
    model = ModelGia(kich_ban="lap")
    tools = tools_langchain()

    middleware, config = [], {}
    if che_do == "ngan-sach":
        middleware = [ModelCallLimitMiddleware(run_limit=8, exit_behavior="end")]
    elif che_do == "phat-hien-lap":
        middleware = [PhatHienLapMiddleware()]
    else:
        # Không có trần nào của riêng agent. Chỉ còn recursion_limit của LangGraph,
        # đặt cao để thấy rõ agent chạy bao nhiêu vòng trước khi bị chặn cứng.
        config = {"recursion_limit": 36}

    agent = create_agent(model=model, tools=tools, middleware=middleware)

    print("=" * 78)
    print(f"DEMO 2 · chế độ: {che_do}")
    print("=" * 78)

    try:
        kq = agent.invoke({"messages": [{"role": "user", "content": CAU_HOI}]}, config)
    except Exception as e:                      # GraphRecursionError
        print(f"[V1] Human  \"{CAU_HOI}\"")
        print("[V…] AI     weather_forecast(town=…) lặp năm cách viết tên, kết quả y hệt")
        print("-" * 78)
        print(f"Dừng vì  : {type(e).__name__}")
        print(f"           {str(e).splitlines()[0]}")
        print("Nhận xét : trần cứng của LangGraph cứu được hoá đơn, không cứu được")
        print("           chẩn đoán. Không ai biết agent hỏng từ vòng nào.")
        return

    _in_phien(kq["messages"])
    print("-" * 78)
    if che_do == "ngan-sach":
        print("Dừng vì  : ModelCallLimitMiddleware(run_limit=8) · dừng xác định, không")
        print("           phụ thuộc model. Nhưng log chỉ nói 'hết lượt', không nói vì sao.")
    else:
        print("Dừng vì  : LoopDetector báo động ngay khi (tool, args) trùng lần thứ hai.")
        print("           Đây là code bạn tự viết, và là bản vá duy nhất nói được")
        print("           chuyện gì đã xảy ra cho người nhận.")


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


def main() -> None:
    p = argparse.ArgumentParser(description="Demo 2 · điều kiện dừng")
    p.add_argument("--che-do", default="khong-gioi-han",
                   choices=["khong-gioi-han", "ngan-sach", "phat-hien-lap"])
    chay(p.parse_args().che_do)


if __name__ == "__main__":
    sys.exit(main())
