import joblib

# ระบุ Path ไปยังไฟล์ model_columns.pkl ของคุณ
path = r'C:\Users\Asus\Desktop\project\final\Flask\models\model_columns.pkl'

try:
    cols = joblib.load(path)
    print(f"✅ จำนวนคอลัมน์ทั้งหมด: {len(cols)} คอลัมน์")
    print("-" * 30)
    for i, col in enumerate(cols, 1):
        print(f"{i}. {col}")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาด: {e}")