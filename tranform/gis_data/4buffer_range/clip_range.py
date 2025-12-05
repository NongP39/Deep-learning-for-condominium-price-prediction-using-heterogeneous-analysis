import geopandas as gpd
import os
import glob

# ================== ตั้งค่า Path ตรงนี้ ==================
# 1. ไฟล์ตัวตั้ง (วงใหญ่ 500m)
FOLDER_LARGE = r"C:\Users\Asus\Desktop\ingest_data\gis_data\buffer_range\split_1000m" 

# 2. ไฟล์ตัวลบ (วงเล็ก 200m)
FOLDER_SMALL = r"C:\Users\Asus\Desktop\ingest_data\gis_data\buffer_range\split_500m" 

# 3. โฟลเดอร์สำหรับเก็บผลลัพธ์ (วงแหวน 200-500m)
OUTPUT_FOLDER = r"C:\Users\Asus\Desktop\ingest_data\gis_data\buffer_range\clip_1000m"
# ======================================================

# สร้างโฟลเดอร์ปลายทาง
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def batch_difference():
    # ลิสต์ไฟล์จากโฟลเดอร์วงใหญ่
    files_large = glob.glob(os.path.join(FOLDER_LARGE, "*.shp"))
    
    total_files = len(files_large)
    print(f"พบไฟล์ต้นทาง {total_files} ไฟล์... เริ่มสร้างวงแหวน (Donut)")

    for index, path_large in enumerate(files_large):
        try:
            filename = os.path.basename(path_large)
            path_small = os.path.join(FOLDER_SMALL, filename)
            
            # เช็คว่ามีไฟล์คู่กันไหม
            if os.path.exists(path_small):
                
                # อ่านไฟล์
                gdf_large = gpd.read_file(path_large) # 500m
                gdf_small = gpd.read_file(path_small) # 200m
                
                # เช็ค CRS
                if gdf_large.crs != gdf_small.crs:
                    gdf_small = gdf_small.to_crs(gdf_large.crs)
                
                # *** ทำ Difference (เอาใหญ่ตั้ง - เล็กลบ) ***
                # how='difference' จะตัดส่วนที่ซ้อนทับออกไป
                ring_result = gpd.overlay(gdf_large, gdf_small, how='difference')
                
                # บันทึกไฟล์
                if not ring_result.empty:
                    output_path = os.path.join(OUTPUT_FOLDER, filename)
                    ring_result.to_file(output_path)
                    print(f"[{index+1}/{total_files}] สำเร็จ: {filename}")
                else:
                    print(f"[{index+1}/{total_files}] ว่างเปล่า (อาจจะไม่ซ้อนทับกัน): {filename}")
                    
            else:
                print(f"[{index+1}/{total_files}] ข้าม: ไม่พบไฟล์วงเล็กคู่กัน ({filename})")

        except Exception as e:
            print(f"[{index+1}/{total_files}] Error ไฟล์ {filename}: {e}")

    print("\n--- เสร็จสิ้นกระบวนการ ---")

if __name__ == "__main__":
    batch_difference()