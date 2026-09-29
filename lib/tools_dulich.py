# -*- coding: utf-8 -*-
"""SE373 · Buổi 03 · Bộ tool du lịch dùng chung cho ba demo.

Dữ liệu tĩnh, không gọi mạng, để demo trên giảng đường luôn chạy được và
mỗi lần chiếu đều ra cùng một kết quả.

    search_travel_info(query) -> danh sách điểm đến khớp truy vấn
    weather_forecast(town)    -> thời tiết của một điểm đến

Biến môi trường SE373_LOI_TE=1 bật chế độ "observation tệ": tool trả
'{"error": "not found"}' thay vì lỗi có cấu trúc kèm hướng đi khác.
Đây là nguyên nhân gốc của vòng lặp ở Demo 2 và ở Trace A.
"""
import os

# ---------------------------------------------------------------- dữ liệu
DIEM_DEN = [
    dict(town="Nha Trang", tags=["biển", "nam trung bộ", "khánh hoà", "lặn biển"],
         note="Bãi biển dài trong thành phố, nhiều đảo gần bờ."),
    dict(town="Quy Nhơn", tags=["biển", "nam trung bộ", "bình định", "yên tĩnh"],
         note="Bãi Xép và Kỳ Co, ít đông hơn Nha Trang."),
    dict(town="Tuy Hoà", tags=["biển", "nam trung bộ", "phú yên", "yên tĩnh"],
         note="Bãi Xép, Gành Đá Đĩa, đường ven biển vắng."),
    dict(town="Phan Thiết", tags=["biển", "nam trung bộ", "bình thuận", "đồi cát"],
         note="Mũi Né, đồi cát bay, gió mạnh gần như quanh năm."),
    dict(town="Hội An", tags=["biển", "miền trung", "quảng nam", "phố cổ"],
         note="Phố cổ và bãi An Bàng cách trung tâm 4km."),
    dict(town="Đà Nẵng", tags=["biển", "miền trung", "thành phố"],
         note="Bãi Mỹ Khê ngay trung tâm, sân bay quốc tế."),
]

THOI_TIET = {
    "Nha Trang":  dict(weather="nắng", temperature=31),
    "Quy Nhơn":   dict(weather="nắng", temperature=31),
    "Tuy Hoà":    dict(weather="nắng", temperature=30),
    "Phan Thiết": dict(weather="nhiều mây", temperature=29),
    "Hội An":     dict(weather="mưa", temperature=26),
    "Đà Nẵng":    dict(weather="mưa rào", temperature=26),
}

# Vũng Tàu cố tình KHÔNG có trong dữ liệu, để Demo 2 tái hiện được Trace A.


def _loi_te() -> bool:
    return os.environ.get("SE373_LOI_TE") == "1"


def _chuan(s: str) -> str:
    return " ".join(s.lower().split())


# ---------------------------------------------------------------- tool 1
def search_travel_info(query: str) -> dict:
    """Tìm điểm đến du lịch theo từ khoá. Trả về danh sách tên điểm đến kèm mô tả ngắn.

    Dùng tool này TRƯỚC khi hỏi thời tiết, để lấy tên điểm đến hợp lệ.
    """
    q = _chuan(query)
    hits = [d for d in DIEM_DEN
            if any(tok in q for tok in d["tags"]) or _chuan(d["town"]) in q]
    if not hits:
        hits = [d for d in DIEM_DEN if "biển" in d["tags"]]
    # Quy tắc observation: trả cấu trúc, cắt bớt và nói rõ là đã cắt.
    top = hits[:4]
    return {"status": "ok", "matched": len(hits), "returned": len(top),
            "truncated": len(hits) > len(top),
            "results": [{"town": d["town"], "note": d["note"]} for d in top]}


# ---------------------------------------------------------------- tool 2
def weather_forecast(town: str) -> dict:
    """Trả thời tiết dự báo của một điểm đến. Tên điểm đến phải lấy từ search_travel_info."""
    for ten, w in THOI_TIET.items():
        if _chuan(ten) == _chuan(town):
            return {"status": "ok", "town": ten, **w}

    if _loi_te():
        # Observation tệ: không nói được phải làm gì khác.
        return {"error": "not found"}

    # Observation tốt: lỗi có cấu trúc, có hướng đi khác, có danh sách giá trị hợp lệ.
    return {"status": "error", "error": "town_not_found",
            "queried": town,
            "hint": "Gọi search_travel_info để lấy danh sách điểm đến hợp lệ trước khi thử lại",
            "supported_count": len(THOI_TIET),
            "valid_examples": list(THOI_TIET)[:3]}


# ------------------------------------------------- đóng gói cho LangChain
def tools_langchain():
    """Trả danh sách tool LangChain cho create_agent."""
    from langchain_core.tools import tool
    return [tool(search_travel_info), tool(weather_forecast)]


if __name__ == "__main__":
    import json
    print(json.dumps(search_travel_info("bãi biển đẹp Nam Trung Bộ"), ensure_ascii=False, indent=2))
    print(json.dumps(weather_forecast("Quy Nhơn"), ensure_ascii=False))
    print("observation tốt :", json.dumps(weather_forecast("Vung Tau"), ensure_ascii=False))
    os.environ["SE373_LOI_TE"] = "1"
    print("observation tệ  :", json.dumps(weather_forecast("Vung Tau"), ensure_ascii=False))

