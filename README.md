# 🏢 Deep Learning for Condominium Price Prediction using Heterogeneous Analysis

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-green)](https://lightgbm.readthedocs.io/)
[![Framework](https://img.shields.io/badge/Web-Flask-lightgrey)](https://flask.palletsprojects.com/)

ระบบประเมินราคาคอนโดมิเนียมอัจฉริยะที่ใช้การวิเคราะห์ข้อมูลหลายรูปแบบ (**Multimodal Heterogeneous Analysis**) โดยผสานข้อมูลคุณลักษณะทางกายภาพ ข้อมูลเชิงพื้นที่ (POI) และฟีเจอร์ที่สกัดจากภาพถ่ายดาวเทียม เพื่อเพิ่มความแม่นยำในการทำนายราคาอสังหาริมทรัพย์ในเขตกรุงเทพมหานคร

---

## 💻 System Requirements

เพื่อให้ระบบสามารถประมวลผลโมเดล Deep Learning และจัดการข้อมูลเชิงพื้นที่ได้อย่างมีประสิทธิภาพ ควรมีคุณสมบัติดังนี้:

### **Hardware Requirements**
* **CPU:** Intel Core i5 / AMD Ryzen 5 ขึ้นไป (แนะนำ 6 Cores+)
* **GPU:** NVIDIA GPU ที่รองรับ CUDA (VRAM 4GB+) สำหรับขั้นตอน Fine-tuning CNN
* **RAM:** ขั้นต่ำ 16GB (แนะนำ 32GB สำหรับการโหลดภาพดาวเทียมจำนวนมาก)
* **Storage:** พื้นที่ว่างอย่างน้อย 10GB สำหรับ Dataset และไฟล์โมเดล

### **Software Requirements**
* **OS:** Windows 10/11 หรือ Ubuntu 20.04+
* **Python:** Version 3.9 หรือ 3.10
* **CUDA & cuDNN:** เวอร์ชันที่สอดคล้องกับ TensorFlow

---

## 🚀 Project Overview
โครงการนี้มุ่งเน้นการสร้าง **Urban Feature Extractor** เพื่อดึงคุณลักษณะเด่นของเมืองจากภาพถ่ายดาวเทียมมาเป็นตัวแปรในการทำนายราคา:

1. **CNN Fine-Tuning**: นำ EfficientNetB0 มาฝึกฝนร่วมกับข้อมูล POI Density 14 ประเภท เพื่อสร้างโมเดลสกัดฟีเจอร์เชิงเมือง
2. **Feature Composition**: รวมเวกเตอร์จากภาพ (ผ่าน PCA บีบอัดเหลือ 50 มิติ) เข้ากับข้อมูล Tabular และ Spatial
3. **Price Prediction**: ใช้ LightGBM Regressor ประมวลผลเวกเตอร์รวม 115+ มิติ เพื่อทำนายราคาขาย

---

## 🏗️ Architecture Design



### **Data Processing Pipeline**
* **Visual Path**: `Satellite Images` ➔ `EfficientNetB0` ➔ `PCA` ➔ `50 Features`
* **Spatial Path**: `Shapefiles` ➔ `Buffer Analysis (1km)` ➔ `POI Density` ➔ `42 Features`
* **Tabular Path**: `Condo Specs` ➔ `Standard Scaling / One-Hot Encoding` ➔ `23 Features`

---

### Urban Feature Extractor ยังติดปัญหาไม่สามารถสกัด Feature ที่สำคัญได้ ต้องมีการนำข้อมูลประเภทอื่นๆ มาช่วยในการทำ Fine-Tuning เพิ่มเติมเพื่อเพิ่มประสิทธิภาพการทำงานของตัว Extractor ให้ดียิ่งขั้น