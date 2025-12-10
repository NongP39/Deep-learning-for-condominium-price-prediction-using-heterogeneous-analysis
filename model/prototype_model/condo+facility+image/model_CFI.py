import pandas as pd
import numpy as np
import os
import cv2
import time
import warnings
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input
from sklearn.decomposition import PCA

# ปิด Warning
warnings.filterwarnings('ignore')

# --- 1. Import Tabular Models ---
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb

# Tools
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================================================================
# PART 1: Image Feature Extraction (ดึงข้อมูลจากภาพดาวเทียม)
# ==============================================================================
print("🚀 PART 1: กำลังดึงข้อมูลจากภาพถ่ายดาวเทียม (Image Feature Extraction)...")

# ตั้งค่า
IMAGE_DIR = '/workspaces/Deep-learning-for-condominium-price-prediction-using-heterogeneous-analysis/input_model/image_sat_data/clip_raster_split'  # โฟลเดอร์ที่เก็บไฟล์ภาพ .tif
CSV_PATH = '/workspaces/Deep-learning-for-condominium-price-prediction-using-heterogeneous-analysis/input_model/tabular_data/final_input_condo_poi_data.csv'

# อ่านไฟล์หลัก
df = pd.read_csv(CSV_PATH)

# 1.1 สร้างโมเดลสำหรับอ่านภาพ (EfficientNetB0)
def build_feature_extractor():
    input_shape = (224, 224, 3)
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # แปลงภาพเป็น Vector (1280 features)
    model = Model(inputs=base_model.input, outputs=x)
    return model

feature_extractor = build_feature_extractor()
print("   - โหลดโมเดล EfficientNetB0 เรียบร้อย")

# 1.2 วนลูปอ่านภาพและดึง Feature
image_features = []
valid_indices = []

print(f"   - กำลังประมวลผลภาพ {len(df)} ภาพ...")

# ดึง id มาใช้ (สมมติว่า id ในตารางตรงกับชื่อไฟล์ภาพ เช่น 1.tif)
ids = df['id'].values

for idx, img_id in enumerate(ids):
    img_path = os.path.join(IMAGE_DIR, f"{img_id}.tif")
    
    # อ่านภาพ
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0  # Normalize
            img = np.expand_dims(img, axis=0) # เพิ่ม batch dim
            
            # ให้โมเดลทำนาย Feature ออกมา
            feat = feature_extractor.predict(img, verbose=0)
            image_features.append(feat[0])
            valid_indices.append(idx)
        else:
            # กรณีไฟล์เสีย ใส่ 0 แทน
            image_features.append(np.zeros(1280))
            valid_indices.append(idx)
    else:
        # กรณีไม่มีภาพ ใส่ 0 แทน
        image_features.append(np.zeros(1280))
        valid_indices.append(idx)
    
    if (idx + 1) % 100 == 0:
        print(f"     > Processed {idx + 1}/{len(df)}")

# 1.3 ลดขนาดข้อมูลภาพ (Dimensionality Reduction)
# ข้อมูลจากภาพมี 1280 ตัวแปร ถ้าเอาไปรวมเลยจะทำให้ตารางใหญ่เกินไปและเทรนช้า
# เราจะใช้ PCA ย่อให้เหลือ 50 ตัวแปรที่สำคัญที่สุด (Visual Components)
print("   - กำลังลดขนาดข้อมูลภาพ (PCA) เพื่อให้เหมาะสมกับการเทรน...")
X_img_raw = np.array(image_features)
pca = PCA(n_components=50, random_state=42)
X_img_pca = pca.fit_transform(X_img_raw)

# สร้าง DataFrame ของข้อมูลภาพ
img_col_names = [f'img_feat_{i}' for i in range(X_img_pca.shape[1])]
df_img = pd.DataFrame(X_img_pca, columns=img_col_names)

# รวมร่างกับตารางเดิม (Reset index เพื่อความชัวร์)
df = df.reset_index(drop=True)
df_final = pd.concat([df, df_img], axis=1)

print(f"✅ รวมข้อมูลสำเร็จ! ขนาดตารางใหม่: {df_final.shape}")
print(f"   (เพิ่มตัวแปรจากภาพดาวเทียมเข้ามา {len(img_col_names)} คอลัมน์)")


# ==============================================================================
# PART 2: Tabular Training (Code GridSearch ของคุณ)
# ==============================================================================
print("\n🚀 PART 2: เริ่มกระบวนการเทรนโมเดล (Super Fine-Tuning)...")

# --- เตรียมข้อมูล ---
# เลือก Features และ Target
drop_cols = ['sale_price', 'name', 'id', 'asking_price', 'asking_price_change_quater', 
             'asking_price_change_year', 'rental_price_change_year', 'gross_rental_yield']
existing_drop = [c for c in drop_cols if c in df_final.columns]

X = df_final.drop(columns=existing_drop)
y = np.log1p(df_final['sale_price']) # Log Transform

# แบ่ง Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessor
categorical_cols = ['district']
# ตอนนี้ numerical_cols จะรวม img_feat_0 ถึง img_feat_49 เข้าไปด้วยอัตโนมัติ
numerical_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# --- ตั้งค่า Grid Search (Super Fine-Tuning) ---
model_params = {
    # --- กลุ่ม Linear ---
    'Ridge': {
        'model': Ridge(),
        'params': {
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    },
    'ElasticNet': {
        'model': ElasticNet(),
        'params': {
            'regressor__alpha': [0.01, 0.1, 1.0, 10.0],
            'regressor__l1_ratio': [0.2, 0.4, 0.6, 0.8]
        }
    },

    # --- กลุ่ม Distance ---
    'KNN': {
        'model': KNeighborsRegressor(),
        'params': {
            'regressor__n_neighbors': [3, 5, 10, 20, 30],
            'regressor__weights': ['uniform', 'distance'],
            'regressor__p': [1, 2] # 1=Manhattan, 2=Euclidean
        }
    },

    # --- กลุ่ม Tree Ensemble ---
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'regressor__n_estimators': [100, 200, 300, 500],
            'regressor__max_depth': [10, 20, 30, None],
            'regressor__min_samples_leaf': [1, 2, 4, 8],
            'regressor__max_features': ['sqrt', 'log2', 0.5]
        }
    },
    'ExtraTrees': {
        'model': ExtraTreesRegressor(random_state=42),
        'params': {
            'regressor__n_estimators': [100, 200, 300, 500],
            'regressor__max_depth': [10, 20, 30, None],
            'regressor__min_samples_leaf': [1, 2, 4, 8]
        }
    },

    # --- กลุ่ม Boosting (ตัวเก็งชนะเลิศ) ---
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'regressor__n_estimators': [100, 200, 300, 500],
            'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'regressor__max_depth': [3, 4, 5, 6]
        }
    },
    'XGBoost': {
        'model': xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
        'params': {
            'regressor__n_estimators': [500, 1000, 2000, 3000],
            'regressor__learning_rate': [0.005, 0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5, 7, 9],
            'regressor__subsample': [0.6, 0.7, 0.8, 0.9]
        }
    },
    'LightGBM': {
        'model': lgb.LGBMRegressor(random_state=42, verbose=-1),
        'params': {
            'regressor__n_estimators': [500, 1000, 2000, 3000],
            'regressor__learning_rate': [0.005, 0.01, 0.05, 0.1],
            'regressor__num_leaves': [31, 50, 100, 200], # ยิ่งเยอะยิ่งฉลาด
            'regressor__feature_fraction': [0.6, 0.7, 0.8, 0.9]
        }
    }
}

# --- เริ่มรัน (The Ultimate Battle) ---
results = []
print("="*70)

for model_name, mp in model_params.items():
    start_time = time.time()
    
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', mp['model'])])
    
    # Grid Search (cv=3)
    grid = GridSearchCV(clf, mp['params'], cv=3, scoring='r2', n_jobs=-1)
    
    print(f"Training {model_name}...")
    try:
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        y_pred_log = best_model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)
        
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        duration = time.time() - start_time
        
        results.append({
            'Model': model_name,
            'Best R2': r2,
            'MAE (Baht)': mae,
            'MAPE (%)': mape,
            'Best Params': grid.best_params_,
            'Training Time (s)': duration
        })
        print(f"  --> Done! R2: {r2:.4f} | MAE: {mae:,.0f} | MAPE: {mape:.2f}% | Used: {duration:.1f}s")
        
    except Exception as e:
        print(f"  --> Error with {model_name}: {e}")

# --- สรุปผล ---
results_df = pd.DataFrame(results).sort_values(by='MAE (Baht)', ascending=True)

print("\n" + "="*70)
print("🏆 Leaderboard (Tabular + Satellite Images)")
print("="*70)
print(results_df[['Model', 'MAE (Baht)', 'MAPE (%)', 'Best R2', 'Training Time (s)']].to_string(index=False))

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x='MAE (Baht)', y='Model', data=results_df, palette='viridis')
plt.title('Model Comparison (with Image Features): Mean Absolute Error')
plt.xlabel('MAE (Baht)')
plt.show()

if not results_df.empty:
    winner = results_df.iloc[0]
    print(f"\n🥇 ผู้ชนะเลิศคือ: {winner['Model']}")
    print("สุดยอดพารามิเตอร์ (Best Params):")
    print(winner['Best Params'])