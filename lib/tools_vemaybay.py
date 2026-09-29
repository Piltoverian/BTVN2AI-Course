# -*- coding: utf-8 -*-
"""BTVN#3 · Bộ tool mockup đặt vé máy bay."""
import json
import os

CHUYEN_BAY = [
    dict(flight_id="VN101", origin="HAN", destination="DAD", date="06/09",
         dep_time="08:00", cabin="Economy", price=1_500_000, price_str="1.500.000đ",
         seats_left=0, baggage="23kg ký gửi"),
    dict(flight_id="VJ202", origin="HAN", destination="DAD", date="06/09",
         dep_time="09:30", cabin="Economy", price=1_800_000, price_str="1.800.000đ",
         seats_left=5, baggage="7kg xách tay"),
    dict(flight_id="VN105", origin="HAN", destination="DAD", date="06/09",
         dep_time="14:00", cabin="Business", price=3_200_000, price_str="3.200.000đ",
         seats_left=4, baggage="32kg ký gửi"),
    dict(flight_id="QH309", origin="HAN", destination="DAD", date="06/09",
         dep_time="23:15", cabin="Economy", price=1_400_000, price_str="1.400.000đ",
         seats_left=6, baggage="20kg ký gửi"),
    dict(flight_id="VJ501", origin="HAN", destination="SGN", date="06/09",
         dep_time="10:15", cabin="Economy", price=2_100_000, price_str="2.100.000đ",
         seats_left=3, baggage="7kg xách tay"),
]

DON_DAT_VE: dict[str, dict] = {}


def _loi_te() -> bool:
    return os.environ.get("SE373_LOI_TE") == "1"


def search_flights(origin: str, destination: str, date: str = "06/09") -> dict:
    """Tìm danh sách chuyến bay theo sân bay đi (origin), sân bay đến (destination) và ngày (date)."""
    org, dst = origin.strip().upper(), destination.strip().upper()
    hits = [
        f for f in CHUYEN_BAY
        if f["origin"] == org and f["destination"] == dst and f["date"] == date.strip()
    ]
    if not hits:
        if _loi_te():
            return {"error": "not found"}
        return {
            "status": "error",
            "error": "flight_not_found",
            "queried": {"origin": org, "destination": dst, "date": date},
            "hint": "Kiểm tra lại mã sân bay IATA 3 chữ cái (HAN, DAD, SGN, CXR, PQC)",
            "valid_routes": ["HAN->DAD (06/09)", "HAN->SGN (06/09)"],
        }
    return {
        "status": "ok",
        "matched": len(hits),
        "flights": [
            {
                "flight_id": f["flight_id"],
                "origin": f["origin"],
                "destination": f["destination"],
                "date": f["date"],
                "dep_time": f["dep_time"],
                "cabin": f["cabin"],
                "price": f["price"],
                "price_str": f["price_str"],
            }
            for f in hits
        ],
    }


def get_flight_detail(flight_id: str) -> dict:
    """Kiểm tra số ghế trống thực tế và hành lý của một chuyến bay trước khi giữ chỗ."""
    fid = flight_id.strip().upper()
    for f in CHUYEN_BAY:
        if f["flight_id"] == fid:
            if f["seats_left"] <= 0:
                return {
                    "status": "error",
                    "error": "seat_sold_out",
                    "flight_id": fid,
                    "hint": "Chuyến bay này vừa hết ghế trống, hãy chọn chuyến bay khác từ kết quả search_flights",
                }
            return {"status": "ok", **f}
    return {"status": "error", "error": "invalid_flight_id", "flight_id": fid}


def hold_booking(
    flight_id: str,
    passenger_name: str,
    price: int,
    cabin: str = "Economy",
    origin: str = "HAN",
    destination: str = "DAD",
    dep_time: str = "09:30",
) -> dict:
    """Giữ chỗ tạm thời cho hành khách, trả về mã đặt chỗ PNR ở trạng thái HELD."""
    fid = flight_id.strip().upper()
    chuyen = next((f for f in CHUYEN_BAY if f["flight_id"] == fid), None)
    if not chuyen:
        return {"status": "error", "error": "invalid_flight_id", "flight_id": fid}
    if chuyen["seats_left"] <= 0:
        return {
            "status": "error",
            "error": "seat_sold_out",
            "flight_id": fid,
            "hint": "Chuyến bay vừa hết ghế, không thể giữ chỗ. Hãy chọn chuyến khác.",
        }

    booking_id = f"PNR-{1001 + len(DON_DAT_VE)}"
    don = {
        "status": "HELD",
        "booking_id": booking_id,
        "passenger_name": passenger_name,
        "flight_id": chuyen["flight_id"],
        "origin": chuyen["origin"],
        "destination": chuyen["destination"],
        "date": chuyen["date"],
        "dep_time": chuyen["dep_time"],
        "cabin": chuyen["cabin"],
        "price": chuyen["price"],
        "price_str": chuyen["price_str"],
    }
    DON_DAT_VE[booking_id] = don
    return don


def issue_ticket(booking_id: str) -> dict:
    """Xuất vé chính thức và trừ tiền cho mã đặt chỗ đã giữ (yêu cầu quyền phê duyệt)."""
    bid = booking_id.strip().upper()
    if bid not in DON_DAT_VE:
        return {"status": "error", "error": "booking_not_found", "booking_id": bid}
    DON_DAT_VE[bid]["status"] = "ISSUED"
    return DON_DAT_VE[bid]


def tools_vemaybay_langchain():
    """Trả danh sách tool đặt vé máy bay cho create_agent."""
    from langchain_core.tools import tool
    return [
        tool(search_flights),
        tool(get_flight_detail),
        tool(hold_booking),
        tool(issue_ticket),
    ]


if __name__ == "__main__":
    print("── Tool Đặt vé máy bay ──")
    print("1. search_flights(HAN->DAD):", json.dumps(search_flights("HAN", "DAD", "06/09"), ensure_ascii=False))
    print("2. get_flight_detail(VN101):", json.dumps(get_flight_detail("VN101"), ensure_ascii=False))
    print("3. get_flight_detail(VJ202):", json.dumps(get_flight_detail("VJ202"), ensure_ascii=False))
    print("4. hold_booking(VJ202)     :", json.dumps(hold_booking("VJ202", "Nguyen Van A", 1_800_000), ensure_ascii=False))
