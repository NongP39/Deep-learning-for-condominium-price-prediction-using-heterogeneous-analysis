import geopandas as gpd
import pandas as pd
import os
import glob

# ================== ตั้งค่า Path ตรงนี้ ==================
# โฟลเดอร์ที่เก็บไฟล์ย่อยๆ ที่ต้องการรวม
INPUT_FOLDER = r"C:\Users\Asus\Desktop\ingest_data\gis_data\4buffer_range\clip_1000m"

# ชื่อไฟล์ผลลัพธ์ที่ต้องการ (รวมเสร็จแล้วให้ไปอยู่ที่ไหน)
OUTPUT_FILE = r"C:\Users\Asus\Desktop\ingest_data\gis_data\5marge_range\merge_1000m"
# ======================================================

def merge_shapefiles():
    # 1. ค้นหาไฟล์ .shp ทั้งหมดในโฟลเดอร์
    shp_files = glob.glob(os.path.join(INPUT_FOLDER, "*.shp"))
    
    if not shp_files:
        print("ไม่พบไฟล์ .shp ในโฟลเดอร์ที่ระบุ")
        return

    print(f"พบไฟล์ทั้งหมด {len(shp_files)} ไฟล์ กำลังเริ่มอ่านข้อมูล...")

    # 2. ลิสต์สำหรับเก็บข้อมูลชั่วคราว
    gdf_list = []
    
    for index, file_path in enumerate(shp_files):
        try:
            # อ่านไฟล์
            gdf = gpd.read_file(file_path)
            
            # (Optional) เพิ่มคอลัมน์ชื่อไฟล์ต้นทาง เผื่ออยากรู้ว่าข้อมูลนี้มาจากไฟล์ไหน
            gdf['source_file'] = os.path.basename(file_path)
            
            gdf_list.append(gdf)
            
            # แสดงความคืบหน้าทุกๆ 100 ไฟล์
            if (index + 1) % 100 == 0:
                print(f"อ่านไปแล้ว {index + 1} ไฟล์...")
                
        except Exception as e:
            print(f"อ่านไฟล์ {os.path.basename(file_path)} ไม่ได้: {e}")

    if not gdf_list:
        print("ไม่มีข้อมูลที่สามารถรวมได้")
        return

    print("กำลังรวมไฟล์ (Merging)... ขั้นตอนนี้อาจใช้เวลาสักพักถ้าไฟล์เยอะ")

    # 3. รวม DataFrame ทั้งหมดเข้าด้วยกัน
    # ignore_index=True เพื่อรันเลข index ใหม่ ไม่ให้ซ้ำกัน
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))

    # 4. ตั้งค่า CRS (ระบบพิกัด) ให้เหมือนกับไฟล์แรกที่อ่านมา (เพื่อความชัวร์)
    # ถ้าไฟล์ย่อยมี CRS ต่างกัน อาจจะต้องมีการ convert ก่อนในลูปด้านบน
    if gdf_list:
         merged_gdf.set_crs(gdf_list[0].crs, allow_override=True, inplace=True)

    # 5. บันทึกเป็นไฟล์ใหม่
    # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    merged_gdf.to_file(OUTPUT_FILE, encoding='utf-8')
    print(f"เสร็จสิ้น! บันทึกไฟล์รวมไว้ที่: {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_shapefiles()