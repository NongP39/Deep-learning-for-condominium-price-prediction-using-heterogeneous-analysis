import math
from geopy.distance import geodesic

def get_spatial_feature(house_lat, house_lon, facility_lat, facility_lon):
    """
    ฟังก์ชันสำหรับคืนค่า (Distance_Bin, Quadrant)
    """
    
    # 1. คำนวณระยะห่าง (Distance) หน่วยเป็นกิโลเมตร
    # งานวิจัยใช้ระยะ 1 km เป็นขอบเขต [cite: 169]
    house_loc = (house_lat, house_lon)
    fac_loc = (facility_lat, facility_lon)
    distance = geodesic(house_loc, fac_loc).km
    
    if distance > 1.0:
        return None, None # ตัดทิ้งถ้าไกลเกิน 1 กม.
        
    # แบ่งระยะทางเป็น 3 ช่วง (สมมติช่วงตามงานวิจัย เช่น 0-0.4, 0.41-0.6, 0.61-1.0)
    if distance <= 0.4:
        dist_bin = 1 # ใกล้
    elif distance <= 0.6:
        dist_bin = 2 # กลาง
    else:
        dist_bin = 3 # ไกล

    # 2. คำนวณทิศทาง (Quadrant) [cite: 171, 228]
    # ให้บ้านเป็นจุดศูนย์กลาง (0,0)
    delta_lat = facility_lat - house_lat
    delta_lon = facility_lon - house_lon
    
    quadrant = 0
    if delta_lat >= 0 and delta_lon >= 0:
        quadrant = 1 # Q1 (บน-ขวา)
    elif delta_lat >= 0 and delta_lon < 0:
        quadrant = 2 # Q2 (บน-ซ้าย)
    elif delta_lat < 0 and delta_lon < 0:
        quadrant = 3 # Q3 (ล่าง-ซ้าย)
    else: # delta_lat < 0 and delta_lon >= 0
        quadrant = 4 # Q4 (ล่าง-ขวา)
        
    return dist_bin, quadrant

# --- ตัวอย่างการใช้งาน ---
house = (13.7563, 100.5018) # พิกัดสมมติ (กรุงเทพ)
seven_eleven = (13.7580, 100.5050) # พิกัดสมมติ (อยู่ทางขวาและสูงกว่า)

d_bin, quad = get_spatial_feature(house[0], house[1], seven_eleven[0], seven_eleven[1])

print(f"ระยะทางช่วงที่: {d_bin}")
print(f"อยู่ใน Quadrant ที่: {quad}") 
# ผลลัพธ์ควรจะได้ Quadrant 1 เพราะพิกัดร้านมากกว่าบ้านทั้งคู่