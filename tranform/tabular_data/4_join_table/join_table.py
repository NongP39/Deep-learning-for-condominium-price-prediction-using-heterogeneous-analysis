import pandas as pd

# 1. อ่านไฟล์ทั้งสอง
df_poi = pd.read_csv(r'C:\Users\Asus\Desktop\ingest_data\tranfrom_data\join_table\poi_summary_condo_by_id.csv')
df_condo = pd.read_csv(r'C:\Users\Asus\Desktop\ingest_data\tranfrom_data\join_table\final_condo_list.csv',encoding='cp874')

# 2. ทำการ Join ข้อมูล (Left Join เพื่อยึดข้อมูลคอนโดเป็นหลัก)
# ใช้ suffixes เพื่อระบุคอลัมน์ที่ชื่อซ้ำกัน
merged_df = pd.merge(df_condo, df_poi, left_on='name', right_on='Name', how='inner', suffixes=('', '_dup'))

# 3. ลบคอลัมน์ที่ชื่อซ้ำ (ลงท้ายด้วย _dup)
cols_to_drop = [col for col in merged_df.columns if col.endswith('_dup')]
merged_df.drop(columns=cols_to_drop, inplace=True)

# ลบคอลัมน์ Name (จากไฟล์ POI) ที่ซ้ำกับ name (จากไฟล์ condo)
if 'Name' in merged_df.columns:
    merged_df.drop(columns=['Name'], inplace=True)

# ลบคอลัมน์ Unnamed ที่อาจติดมา
unnamed_cols = [col for col in merged_df.columns if 'Unnamed' in col]
merged_df.drop(columns=unnamed_cols, inplace=True)

# 4. บันทึกผลลัพธ์
merged_df.to_csv(r'C:\Users\Asus\Desktop\ingest_data\tranfrom_data\join_table\merged_condo_poi_data.csv', index=False)