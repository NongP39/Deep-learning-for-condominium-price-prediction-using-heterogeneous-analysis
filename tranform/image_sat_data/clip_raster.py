import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask

# =================ตั้งค่าไฟล์ตรงนี้=================
INPUT_RASTER = r"C:\Users\Asus\Desktop\ingest_data\image_sat_data\T47PPR_20251201T034131_TCI_10m.jp2"  # ไฟล์ภาพดาวเทียมตัวแม่
INPUT_SHP = r"C:\Users\Asus\Desktop\ingest_data\gis_data\condo_buffer_shp\buffer_condo.shp"           # ไฟล์ Shapefile ที่มี 1517 polygons
OUTPUT_DIR = r"C:\Users\Asus\Desktop\ingest_data\image_sat_data\clip_raster_split"                        # โฟลเดอร์ที่จะเก็บผลลัพธ์
ID_COLUMN = "id"                                   # ชื่อคอลัมน์ใน shp ที่จะเอามาตั้งชื่อไฟล์ (เช่น id, plot_no)
# ===============================================

# สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def clip_raster_by_shp():
    # 1. อ่าน Shapefile
    gdf = gpd.read_file(INPUT_SHP)
    
    # 2. เปิดไฟล์ภาพ
    with rasterio.open(INPUT_RASTER) as src:
        print(f"กำลังประมวลผล... ภาพหลักระบบพิกัด: {src.crs}")
        
        # ***สำคัญมาก***: เช็คว่า CRS ตรงกันไหม ถ้าไม่ตรงต้องแปลง Shapefile ให้ตรงกับภาพ
        if gdf.crs != src.crs:
            print(f"ระบบพิกัดไม่ตรงกัน! กำลังแปลง Shapefile จาก {gdf.crs} เป็น {src.crs}")
            gdf = gdf.to_crs(src.crs)

        # 3. วนลูปตัดภาพทีละ Polygon
        count = 0
        total = len(gdf)
        
        for index, row in gdf.iterrows():
            try:
                # ดึง Geometry (รูปทรง) ของ Polygon นั้นๆ
                geometry = [row['geometry']]
                
                # ทำการตัดภาพ (Crop)
                # crop=True หมายถึงให้ตัดขอบภาพให้เหลือแค่สี่เหลี่ยมที่คลุม Polygon นั้นพอดี
                out_image, out_transform = mask(src, geometry, crop=True)
                
                # อัปเดต Metadata (ข้อมูลพิกัดและขนาดภาพใหม่)
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                
                # ตั้งชื่อไฟล์ตาม ID
                file_name = f"{row[ID_COLUMN]}.tif"
                out_path = os.path.join(OUTPUT_DIR, file_name)
                
                # บันทึกไฟล์
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                
                count += 1
                if count % 100 == 0:
                    print(f"ตัดเสร็จแล้ว {count}/{total} ภาพ...")
                    
            except Exception as e:
                print(f"Error ที่ ID {row[ID_COLUMN]}: {e}")

    print("เสร็จสิ้นกระบวนการ!")

if __name__ == "__main__":
    clip_raster_by_shp()