import os
import cv2
import joblib
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
import geopandas as gpd
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from shapely.geometry import Point, mapping
from rasterio.mask import mask
from pyproj import Transformer

# ==========================================
# ⚙️ 1. SYSTEM SETUP
# ==========================================

# ปิด Warning ของ TensorFlow ที่ไม่จำเป็น
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ตั้งค่า Logging
logging.basicConfig(
    filename='system_valuation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# --- Dynamic Path Configuration (ย้ายเครื่องได้ไม่พัง) ---
# หา Path ปัจจุบันของไฟล์นี้
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# หรือถ้าต้องการ Fix Path เดิมให้แก้บรรทัดนี้: 
# BASE_DIR = r'C:\Users\Asus\Desktop\project\final\Flask'

MODEL_PATH = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'data')
TIF_PATH = os.path.join(DATA_PATH, 'satellite', 'bangkok_full.tif')
SHP_ROOT_DIR = os.path.join(DATA_PATH, 'shp')
BKK_BOUNDARY_PATH = os.path.join(SHP_ROOT_DIR, 'bangkok.shp')

# --- Global Variables (โหลดครั้งเดียวใช้ยาวๆ) ---
FINAL_MODEL = None
IMG_EXTRACTOR = None
PCA_PROCESSOR = None
MODEL_COLUMNS = None
COORD_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)

# ตัวแปรเก็บข้อมูล GIS ใน RAM (Performance Optimization)
BKK_GDF = None
POI_GDFS = {}

POI_CAT = [
    'clinic', 'convenience_store', 'school', 'university', 'gov_service', 
    'Hospital', 'mall', 'night_club', 'public_park', 'restaurant', 
    'supermarket', 'Airport', 'E_railway', 'railway'
]

# --- Business Logic ---
MIN_PRICE_SQM = 15000
MAX_PRICE_SQM = 800000

# ==========================================
# 📥 2. INITIALIZATION (LOAD ONCE)
# ==========================================

def load_resources():
    """โหลดโมเดลและข้อมูล GIS ทั้งหมดก่อนเริ่ม Server"""
    global FINAL_MODEL, IMG_EXTRACTOR, PCA_PROCESSOR, MODEL_COLUMNS, BKK_GDF, POI_GDFS
    
    print("\n" + "="*50)
    print("🚀 STARTING SYSTEM INITIALIZATION")
    print("="*50)

    # --- Step 1: Load ML Models ---
    try:
        print("⏳ [1/5] Loading Final Prediction Model...")
        FINAL_MODEL = joblib.load(os.path.join(MODEL_PATH, 'final_condo_price_model.pkl'))
        
        print("⏳ [2/5] Loading Image Extractor (This may take 30s)...")
        IMG_EXTRACTOR = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'image_feature_extractor.h5'))
        
        print("⏳ [3/5] Loading PCA Processor...")
        PCA_PROCESSOR = joblib.load(os.path.join(MODEL_PATH, 'pca_processor.pkl'))
        
        print("⏳ [4/5] Loading Model Columns...")
        MODEL_COLUMNS = joblib.load(os.path.join(MODEL_PATH, 'model_columns.pkl'))
        
    except Exception as e:
        logging.critical(f"Model Loading Failed: {e}")
        print(f"❌ Error Loading Models: {e}")
        raise e

    # --- Step 2: Load GIS Data (Optimization) ---
    print("⏳ [5/5] Loading GIS Data into RAM...")
    
    # 2.1 Load Bangkok Boundary
    try:
        if os.path.exists(BKK_BOUNDARY_PATH):
            for enc in ['cp874', 'tis-620', 'utf-8']:
                try:
                    temp_gdf = gpd.read_file(BKK_BOUNDARY_PATH, encoding=enc)
                    if temp_gdf.crs is None or temp_gdf.crs != "EPSG:4326":
                        temp_gdf = temp_gdf.to_crs("EPSG:4326")
                    BKK_GDF = temp_gdf
                    break
                except: continue
        else:
            print("⚠️ Warning: Bangkok boundary file not found.")
    except Exception as e:
        print(f"⚠️ Error loading BKK boundary: {e}")

    # 2.2 Load All POI Shapefiles
    for cat in POI_CAT:
        path = os.path.join(SHP_ROOT_DIR, f"{cat}.shp")
        if os.path.exists(path):
            for enc in ['cp874', 'tis-620', 'utf-8']:
                try:
                    temp_gdf = gpd.read_file(path, encoding=enc)
                    # แปลงเป็น UTM (EPSG:32647) รอไว้เลยเพื่อความเร็วในการคำนวณระยะทาง
                    if temp_gdf.crs is None or temp_gdf.crs != "EPSG:32647":
                        temp_gdf = temp_gdf.to_crs("EPSG:32647")
                    POI_GDFS[cat] = temp_gdf
                    # print(f"   - Loaded {cat}")
                    break
                except: continue
    
    print(f"✅ Loaded {len(POI_GDFS)} POI categories.")
    print("\n✅ SYSTEM READY! Waiting for requests...")
    print("="*50 + "\n")

# เรียกฟังก์ชันโหลดทันทีที่รันไฟล์
load_resources()

# ==========================================
# 🗺️ 3. CORE PROCESSING FUNCTIONS
# ==========================================

def is_within_bkk(lat, lon):
    """ตรวจสอบขอบเขตโดยใช้ข้อมูลที่โหลดไว้ใน RAM"""
    try:
        if BKK_GDF is None: return True # ถ้าโหลดไฟล์ไม่ได้ ให้ปล่อยผ่านไปก่อน
        p = Point(float(lon), float(lat))
        return BKK_GDF.geometry.contains(p).any()
    except Exception as e:
        logging.error(f"Geofence Error: {e}")
        return True

def get_poi_counts_fast(lat, lon):
    """นับ POI แบบเร็ว (ใช้ข้อมูลใน RAM)"""
    try:
        utm_lon, utm_lat = COORD_TRANSFORMER.transform(lon, lat)
        pt = Point(utm_lon, utm_lat)
        poi_results = {}

        for cat in POI_CAT:
            c200, c500, c1000 = 0, 0, 0
            6
            if cat in POI_GDFS:
                gdf = POI_GDFS[cat]
                c200 = int(gdf.geometry.within(pt.buffer(200)).sum())
                c500 = int(gdf.geometry.within(pt.buffer(500)).sum()) - c200
                c1000 = int(gdf.geometry.within(pt.buffer(1000)).sum()) - (c200 + c500)
            
            poi_results[f"{cat}_200m"] = c200
            poi_results[f"{cat}_500m"] = c500
            poi_results[f"{cat}_1000m"] = c1000
        
        return poi_results
    except Exception as e:
        logging.error(f"POI Error: {e}")
        return {f"{cat}_{d}": 0 for cat in POI_CAT for d in ['200m', '500m', '1000m']}

def process_satellite_features(lat, lon):
    """สกัดฟีเจอร์ภาพดาวเทียม พร้อม Fallback"""
    col_names = [f'smart_img_{i}' for i in range(50)]
    try:
        utm_lon, utm_lat = COORD_TRANSFORMER.transform(lon, lat)
        
        if not os.path.exists(TIF_PATH):
            raise FileNotFoundError("TIF file missing")

        with rasterio.open(TIF_PATH) as src:
            if not (src.bounds.left < utm_lon < src.bounds.right and src.bounds.bottom < utm_lat < src.bounds.top):
                logging.warning("Out of satellite bounds")
                raise ValueError("Out of bounds")
            
            out_image, _ = mask(src, [mapping(Point(utm_lon, utm_lat).buffer(1000))], crop=True)
            img = np.moveaxis(out_image, 0, -1)
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32') / 255.0 
            
            raw_feats = IMG_EXTRACTOR.predict(np.expand_dims(img, axis=0), verbose=0)
            if len(raw_feats.shape) > 2:
                raw_feats = raw_feats.reshape(raw_feats.shape[0], -1)
                
            pca_feats = PCA_PROCESSOR.transform(raw_feats)
            return pd.DataFrame(pca_feats, columns=col_names), False

    except Exception as e:
        logging.warning(f"Satellite Fallback used: {e}")
        zero_feats = np.zeros((1, 50))
        return pd.DataFrame(zero_feats, columns=col_names), True

def apply_guardrails(price, area):
    """ตรวจสอบความสมเหตุสมผลของราคา"""
    warnings = []
    final_price = float(price)
    
    if area <= 0:
        warnings.append("ไม่ระบุพื้นที่ใช้สอย (Area) ระบบไม่สามารถตรวจสอบมาตรฐานราคาต่อ ตร.ม. ได้")
        return final_price, warnings

    price_per_sqm = final_price / area
    
    if price_per_sqm < MIN_PRICE_SQM:
        final_price = MIN_PRICE_SQM * area
        warnings.append("ราคาประเมินต่ำกว่าเกณฑ์ตลาด (ปรับเป็นราคาขั้นต่ำ)")
    elif price_per_sqm > MAX_PRICE_SQM:
        final_price = MAX_PRICE_SQM * area
        warnings.append("ราคาประเมินสูงเกินเกณฑ์ตลาด (ปรับเป็นราคาเพดาน)")
        
    return final_price, warnings

# ==========================================
# 🚀 4. FLASK ENDPOINTS
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        lat = float(data.get('lat', 0))
        lon = float(data.get('lon', 0))
        area = float(data.get('area_sqm', 0))

        # 1. Geofence Check
        if not is_within_bkk(lat, lon):
            return jsonify({"error": "ตำแหน่งอยู่นอกเขตกรุงเทพฯ หรือพื้นที่ให้บริการ"}), 400

        # 2. Satellite Image Processing
        img_df, is_fallback = process_satellite_features(lat, lon)
        
        # 3. POI Processing (Fast Version)
        poi_data = get_poi_counts_fast(lat, lon)

        # 4. Data Preparation
        combined_dict = {**data, **poi_data}
        df_tabular = pd.DataFrame([combined_dict])
        df_final = pd.concat([df_tabular, img_df], axis=1)
        
        # Reindex & Clean
        df_model = df_final.reindex(columns=MODEL_COLUMNS)
        df_model['district'] = df_model['district'].astype(str)
        for col in df_model.columns:
            if col != 'district':
                df_model[col] = pd.to_numeric(df_model[col], errors='coerce').fillna(0)

        # 5. Prediction
        pred_log = FINAL_MODEL.predict(df_model)
        raw_price = float(np.expm1(pred_log)[0])

        # 6. Guardrails
        final_price, warning_msgs = apply_guardrails(raw_price, area)

        # Response
        response = {
            "status": "success",
            "predicted_price": f"{final_price:,.2f}",
            "poi_summary": {k: int(v) for k, v in poi_data.items() if '_500m' in k},
            "meta": {
                "fallback_mode": is_fallback,
                "warnings": warning_msgs
            }
        }
        
        logging.info(f"Success: {lat},{lon} -> {final_price}")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Predict Error: {e}")
        return jsonify({"error": "เกิดข้อผิดพลาดภายในระบบ"}), 500

@app.route('/result')
def result():
    return render_template('result.html', 
                         price=request.args.get('price', '0'), 
                         warnings=request.args.get('warnings', ''))

# ==========================================
# 🏁 5. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)