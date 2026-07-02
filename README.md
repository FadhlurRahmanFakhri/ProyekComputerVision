# Deteksi Penggunaan Masker Wajah Menggunakan Metode Haar Cascade dan Convolutional Neural Network (CNN)

Proyek ini adalah sistem deteksi penggunaan masker wajah secara *real-time* yang menggabungkan metode Computer Vision tradisional dan Deep Learning. Proyek ini dikembangkan sebagai tugas kuliah Computer Vision (Semester 8).

---

## 📌 Deskripsi Proyek

Sistem ini dirancang untuk mengidentifikasi apakah seseorang memakai masker wajah secara benar atau tidak. Alur kerja sistem dibagi menjadi dua tahap utama:
1. **Deteksi Wajah (Face Detection)**: Menggunakan metode **Haar Cascade Classifier** (pada tahap prapemrosesan/Python) dan **BlazeFace** (pada aplikasi Web *real-time*).
2. **Klasifikasi Citra (Image Classification)**: Menggunakan model **Convolutional Neural Network (CNN)** yang dilatih menggunakan TensorFlow/Keras untuk menentukan status wajah (menggunakan masker atau tidak).

---

## 🛠️ Arsitektur & Teknologi

### **1. Sisi Model & Preprocessing (Python)**
* **OpenCV**: Digunakan untuk membaca citra, konversi ruang warna (BGR ke Gray/RGB), dan prapemrosesan data.
* **Haar Cascade (`haarcascade_frontalface_default.xml`)**: Digunakan untuk mendeteksi koordinat wajah pada dataset gambar utuh sebelum dipotong (*cropped*).
* **TensorFlow & Keras**: Digunakan untuk membangun, melatih, dan mengevaluasi arsitektur model CNN.
* **Jupyter Notebook (`ProyekCV_FadhlurRahmanFakhri (1).ipynb`)**: Dokumentasi alur eksperimen, pengolahan dataset, hingga training model.

### **2. Sisi Web Aplikasi (Client-Side Deployment)**
* **HTML5 & Vanilla CSS (Tailwind CSS)**: Antarmuka pengguna (UI) modern dengan tema *dark mode*.
* **TensorFlow.js**: Menjalankan model CNN langsung di peramban web (*browser*) secara lokal tanpa memerlukan server backend.
* **BlazeFace Model**: Digunakan untuk mendeteksi wajah secara *real-time* lewat webcam dengan performa tinggi di sisi klien.

---

## 📁 Struktur Repositori

```text
├── web/
│   ├── model/                  # Model CNN hasil konversi ke format TF.js (model.json & binary shard)
│   ├── app.js                  # Logika aplikasi web, webcam feed, preproses tensor, & inferensi
│   └── index.html              # Antarmuka web utama
├── convert_model.py            # Script Python untuk mengonversi file .h5 ke format TensorFlow.js
├── face_mask_cnn.h5            # Model CNN terlatih dalam format Keras/H5
├── ProyekCV_FadhlurRahmanFakhri (1).ipynb   # Jupyter Notebook untuk training model
└── README.md                   # Dokumentasi proyek ini
```

---

## 🚀 Cara Menjalankan Aplikasi Web

Karena aplikasi menggunakan **TensorFlow.js** untuk memuat model lokal, browser memerlukan server lokal (HTTP Server) karena alasan keamanan (*CORS Policy*).

### **Langkah 1: Jalankan Local Server**
Anda dapat menggunakan ekstensi VS Code **Live Server** atau menggunakan perintah Python berikut di terminal proyek:
```bash
# Python 3
python -m http.server 8000
```

### **Langkah 2: Buka di Browser**
Akses tautan berikut di peramban web Anda:
```text
http://localhost:8000/web/
```

---

## 📊 Hasil Pelatihan Model
Model CNN dilatih menggunakan dataset gambar wajah bermasker dan tidak bermasker yang dipotong (*cropped*) menggunakan Haar Cascade untuk memastikan model fokus pada fitur wajah, menghasilkan akurasi klasifikasi yang optimal.