# -*- coding: utf-8 -*-
"""SE373 · Buổi 03 · Hai lớp harness viết tay.

Đây là phần framework KHÔNG làm hộ. Cả hai đều là Python thường, không gọi
model, không phụ thuộc LangChain, nên cắm được vào bất kỳ vòng lặp nào.

    LoopDetector   phát hiện lặp và bế tắc, chạy sau mỗi observation   (S41, S48, S49)
    kiem_can_cu    đối chiếu câu trả lời với kết quả tool đã nhận      (S61, S62)

Chạy thử:  python3 lib/harness.py
"""
from collections import deque
import json
import re
from typing import TypedDict


# ================================================== 0 · RÀNG BUỘC DỮ LIỆU
class RangBuocDatVe(TypedDict):
    ngan_sach_toi_da: int
    hang_ghe_cho_phep: list[str]
    gio_khoi_hanh_tu: str
    gio_khoi_hanh_den: str
    san_bay_hop_le: list[str]
    cho_phep_xuat_ve_tu_dong: bool


RANG_BUOC_MAC_DINH: RangBuocDatVe = {
    "ngan_sach_toi_da": 2_500_000,
    "hang_ghe_cho_phep": ["Economy"],
    "gio_khoi_hanh_tu": "06:00",
    "gio_khoi_hanh_den": "22:00",
    "san_bay_hop_le": ["HAN", "SGN", "DAD", "CXR", "PQC"],
    "cho_phep_xuat_ve_tu_dong": False,
}


def prompt_tu_rang_buoc(rb: RangBuocDatVe) -> str:
    return (
        f"Ràng buộc đặt vé: ngân sách tối đa {rb['ngan_sach_toi_da']:,}đ; "
        f"hạng ghế {rb['hang_ghe_cho_phep']}; "
        f"giờ khởi hành từ {rb['gio_khoi_hanh_tu']} đến {rb['gio_khoi_hanh_den']}; "
        f"sân bay hỗ trợ {rb['san_bay_hop_le']}."
    )


def kiem_rang_buoc(chuyen_bay: dict, rb: RangBuocDatVe) -> list[str]:
    loi = []
    if chuyen_bay.get("origin") and chuyen_bay["origin"] not in rb["san_bay_hop_le"]:
        loi.append(f"Sân bay đi '{chuyen_bay['origin']}' ngoài danh sách hỗ trợ")
    if chuyen_bay.get("destination") and chuyen_bay["destination"] not in rb["san_bay_hop_le"]:
        loi.append(f"Sân bay đến '{chuyen_bay['destination']}' ngoài danh sách hỗ trợ")
    if chuyen_bay.get("price", 0) > rb["ngan_sach_toi_da"]:
        loi.append(f"Giá {chuyen_bay['price']} vượt ngân sách {rb['ngan_sach_toi_da']}")
    if chuyen_bay.get("cabin") and chuyen_bay["cabin"] not in rb["hang_ghe_cho_phep"]:
        loi.append(f"Hạng ghế '{chuyen_bay['cabin']}' không thuộc {rb['hang_ghe_cho_phep']}")
    gio = chuyen_bay.get("dep_time", "")
    if gio and not (rb["gio_khoi_hanh_tu"] <= gio <= rb["gio_khoi_hanh_den"]):
        loi.append(f"Giờ bay {gio} nằm ngoài khung {rb['gio_khoi_hanh_tu']}–{rb['gio_khoi_hanh_den']}")
    return loi


# ===================================================== 1 · PHÁT HIỆN LẶP
class LoopDetector:
    """Ba tín hiệu, theo đúng slide S48.

    1. trùng action      cùng (tool, args) lặp lại trong cửa sổ vài vòng
    2. trùng observation tham số khác nhau nhưng kết quả trả về giống hệt
    3. không tiến triển  một đại lượng của bài toán đứng yên qua N vòng

    Tín hiệu 3 bắt được kiểu bế tắc mà hai tín hiệu đầu bỏ sót: agent đổi
    tool mỗi vòng nhưng vẫn đứng yên.
    """

    def __init__(self, window=6, repeat_k=2, same_obs_k=4, stall_n=5):
        self.recent = deque(maxlen=window)   # dấu vân tay (tool, args)
        self.obs = deque(maxlen=window)      # dấu vân tay observation
        self.k, self.k_obs, self.n = repeat_k, same_obs_k, stall_n
        self.last_progress, self.stall = None, 0

    def check(self, tool: str, args: dict, observation=None, progress=None):
        """Trả chuỗi mô tả nếu phát hiện lặp hoặc bế tắc, None nếu bình thường."""
        fp = (tool, repr(sorted(args.items())))
        if self.recent.count(fp) + 1 >= self.k:
            return (f"LOOP · '{tool}' gọi {self.recent.count(fp) + 1} lần "
                    f"với cùng tham số trong {self.recent.maxlen} vòng gần nhất")
        self.recent.append(fp)

        if observation is not None:
            ofp = repr(observation)
            if self.obs.count(ofp) + 1 >= self.k_obs:
                return (f"LOOP · {self.obs.count(ofp) + 1} lời gọi khác tham số "
                        f"nhưng trả về cùng một kết quả")
            self.obs.append(ofp)

        if progress is not None:
            self.stall = self.stall + 1 if progress == self.last_progress else 0
            self.last_progress = progress
            if self.stall >= self.n:
                return f"STALL · tiến triển đứng yên ở {progress!r} qua {self.stall} vòng"
        return None


def ban_giao(ly_do: str, da_thu: list, trang_thai: dict, cau_hoi: str) -> dict:
    """Dừng bất thường thì bàn giao đủ ba thứ, không break im lặng (S50)."""
    return {"stop_reason": ly_do,
            "da_thu": da_thu,
            "trang_thai": trang_thai,
            "cau_hoi_cho_nguoi": cau_hoi}


# ================================================== 2 · KIỂM TRA CĂN CỨ
# số tiền, giờ, ngày, mã đơn, bốn số cuối thẻ, nhiệt độ
MAU_DU_KIEN = [
    r"\d{1,3}(?:\.\d{3})+(?:đ|\s?VNĐ|\s?đồng)?",   # 1.250.000đ
    r"\d{1,2}:\d{2}",                              # 18:40
    r"\d{1,2}/\d{1,2}(?:/\d{2,4})?",               # 06/09
    r"[A-Z]{2,5}-\d{3,8}",                         # ORD-88123, PNR-1001
    r"\b(?:VN|VJ|QH)\d{3,4}\b",                    # VN101, VJ202
    r"(?<![-\w])\d{4}\b",                          # 4412
    r"\d{1,2}°C",                                  # 31°C
]


def trich_du_kien(text: str) -> list:
    """Lấy mọi số, ngày, giờ và mã định danh xuất hiện trong câu trả lời."""
    ra = []
    for mau in MAU_DU_KIEN:
        ra += re.findall(mau, text)
    return list(dict.fromkeys(ra))      # giữ thứ tự, bỏ trùng


def kiem_can_cu(cau_tra_loi: str, ket_qua_tool: list) -> dict:
    """ket_qua_tool: danh sách nội dung các ToolMessage đã nhận trong phiên."""
    nguon = " ".join(str(x) for x in ket_qua_tool)
    chi_tiet = [{"du_kien": d, "co_nguon": d in nguon} for d in trich_du_kien(cau_tra_loi)]
    thieu = [c["du_kien"] for c in chi_tiet if not c["co_nguon"]]
    return {"dat": not thieu, "chi_tiet": chi_tiet, "khong_co_nguon": thieu}


def chan_truoc_khi_tra_loi(cau_tra_loi: str, ket_qua_tool: list) -> str:
    """Trả câu an toàn. Không tự sửa số liệu, chỉ chặn và nói rõ phần thiếu nguồn."""
    kq = kiem_can_cu(cau_tra_loi, ket_qua_tool)
    if kq["dat"]:
        return cau_tra_loi
    return ("Chưa trả lời được đầy đủ. Các dữ kiện sau không truy được về kết quả tool nào: "
            + " · ".join(kq["khong_co_nguon"])
            + ". Cần bổ sung tool cung cấp thông tin này, hoặc chuyển cho nhân viên hỗ trợ.")


# ======================================================== 3 · KIỂM QUYỀN
class QuyenHan(TypedDict):
    cho_phep_tool: list[str]
    can_nguoi_duyet: list[str]
    da_duoc_nguoi_duyet: bool


QUYEN_MAC_DINH: QuyenHan = {
    "cho_phep_tool": ["search_flights", "get_flight_detail", "hold_booking"],
    "can_nguoi_duyet": ["issue_ticket"],
    "da_duoc_nguoi_duyet": False,
}


def kiem_quyen_tool(ten_tool: str, args: dict, rb: RangBuocDatVe, quyen: QuyenHan) -> str | None:
    if ten_tool in quyen["can_nguoi_duyet"] and not quyen["da_duoc_nguoi_duyet"] and not rb["cho_phep_xuat_ve_tu_dong"]:
        return f"PERMISSION_DENIED · Tool '{ten_tool}' cần người dùng xác nhận trước khi thực thi"
    if ten_tool not in quyen["cho_phep_tool"] and ten_tool not in quyen["can_nguoi_duyet"]:
        return f"PERMISSION_DENIED · Tool '{ten_tool}' không nằm trong danh sách được cấp quyền"
    if ten_tool == "hold_booking":
        vi_pham = kiem_rang_buoc(args, rb)
        if vi_pham:
            return "POLICY_DENIED · " + "; ".join(vi_pham)
    return None


# ==================================== 4 · TIÊU CHÍ HOÀN THÀNH KIỂM BẰNG CODE
def kiem_tieu_chi_hoan_thanh(cau_tra_loi: str, ket_qua_tool: list, rb: RangBuocDatVe) -> dict:
    can_cu = kiem_can_cu(cau_tra_loi, ket_qua_tool)
    don_hop_le = None
    loi_nghiep_vu = []

    for raw in ket_qua_tool:
        if isinstance(raw, dict):
            obs = raw
        else:
            try:
                obs = json.loads(str(raw))
            except Exception:
                continue
        if isinstance(obs, dict) and obs.get("status") in ("HELD", "ISSUED") and obs.get("booking_id"):
            vi_pham = kiem_rang_buoc(obs, rb)
            if vi_pham:
                loi_nghiep_vu.extend(vi_pham)
            else:
                don_hop_le = obs

    if not don_hop_le:
        loi_nghiep_vu.append("Chưa có mã giữ chỗ (status=HELD/ISSUED) hợp lệ từ tool")
    elif don_hop_le["booking_id"] not in cau_tra_loi:
        loi_nghiep_vu.append(f"Câu trả lời thiếu mã đặt chỗ {don_hop_le['booking_id']}")

    if not can_cu["dat"]:
        loi_nghiep_vu.append("Bịa dữ kiện không có nguồn: " + ", ".join(can_cu["khong_co_nguon"]))

    return {
        "hoan_thanh": (don_hop_le is not None) and can_cu["dat"] and not loi_nghiep_vu,
        "booking": don_hop_le,
        "can_cu": can_cu,
        "loi": loi_nghiep_vu,
    }


if __name__ == "__main__":
    try:
        from lib.tools_vemaybay import search_flights, get_flight_detail, hold_booking
    except ImportError:
        from tools_vemaybay import search_flights, get_flight_detail, hold_booking

    print("── Trace A · LoopDetector với search_flights (sân bay không có) ──")
    det = LoopDetector()
    for vong, dst in enumerate(["Vung Tau", "Vũng Tàu", "VTG", "Vung Tau"], 1):
        obs = search_flights("HAN", dst, "06/09")
        canh_bao = det.check("search_flights", {"origin": "HAN", "destination": dst},
                             observation=obs, progress=0)
        print(f"  V{vong} search_flights(origin='HAN', destination={dst!r}) → {canh_bao or 'bình thường'}")
        if canh_bao:
            break

    print("\n── Trace B · kiem_can_cu với kết quả tool đặt vé máy bay ──")
    obs_tim = json.dumps(search_flights("HAN", "DAD", "06/09"), ensure_ascii=False)
    obs_giu = json.dumps(hold_booking("VJ202", "Nguyen Van A", 1_800_000), ensure_ascii=False)
    ket_qua_tool = [obs_tim, obs_giu]
    cau_bia = (
        "Đã giữ chỗ mã PNR-1001 trên chuyến VJ202 ngày 06/09 lúc 09:30, giá vé 1.800.000đ. "
        "Phí hành lý 350.000đ, thanh toán qua thẻ đuôi 4412."
    )
    for c in kiem_can_cu(cau_bia, ket_qua_tool)["chi_tiet"]:
        print(("   có nguồn " if c["co_nguon"] else "   BỊA     "), c["du_kien"])
    print("\n  Câu được phép trả ra:\n  " + chan_truoc_khi_tra_loi(cau_bia, ket_qua_tool))

    print("\n── Trace C · Kiểm quyền & Tiêu chí hoàn thành với tools_vemaybay ──")
    rb = RANG_BUOC_MAC_DINH
    quyen = QUYEN_MAC_DINH
    print("  1. Gọi issue_ticket chưa duyệt :", kiem_quyen_tool("issue_ticket", {"booking_id": "PNR-1001"}, rb, quyen))
    print("  2. Gọi hold_booking(VN105)     :", kiem_quyen_tool("hold_booking", {"flight_id": "VN105", "price": 3_200_000, "cabin": "Business"}, rb, quyen))
    cau_chuan = "Đã giữ chỗ mã PNR-1001 trên chuyến VJ202 ngày 06/09 lúc 09:30, giá 1.800.000đ."
    kq_ht = kiem_tieu_chi_hoan_thanh(cau_chuan, ket_qua_tool, rb)
    print("  3. Kiểm tiêu chí hoàn thành    :", kq_ht["hoan_thanh"], "· booking_id =", kq_ht["booking"]["booking_id"])

