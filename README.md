# ***Integrasi Analisis STFT dan Ekstraksi Fitur untuk Klasifikasi Kondisi Gearbox Menggunakan Algoritma Machine Learning***
---
# Domain Proyek
---

![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/0befa8b0-0505-4143-a4db-5cd1ab64622d)

Gambar 1. Gear

Industri manufaktur dan otomotif sering bergantung pada peralatan kompleks seperti gearbox untuk menjalankan operasi harian mereka. Kerusakan pada gearbox dapat menyebabkan downtime yang mahal dan mengganggu produktivitas. Oleh karena itu, deteksi dini dan pengawasan kondisi gearbox sangat penting untuk mencegah kerusakan yang lebih serius dan mengurangi biaya perbaikan.

Dalam usaha untuk mengatasi tantangan ini, proyek ini bertujuan untuk mengembangkan solusi yang menggabungkan analisis pemrosesan sinyal Short-Time Fourier Transform (STFT) dengan kemampuan klasifikasi model machine learning. Pendekatan ini akan memungkinkan identifikasi kerusakan atau kondisi buruk pada gearbox secara akurat dan efisien.


# Project Understanding
---
***Project Benefits***

- Deteksi Dini Kerusakan: Dengan sistem ini, kerusakan pada gearbox dapat dideteksi lebih awal sehingga tindakan pencegahan atau perbaikan dapat diambil sebelum kerusakan lebih parah terjadi.
- Mengurangi Downtime: Pencegahan kerusakan yang tepat waktu akan mengurangi waktu henti produksi akibat perbaikan yang tidak terduga.
- Optimalisasi Biaya: Dengan mencegah kerusakan yang lebih serius, biaya perbaikan besar dapat dihindari.
- Automatisasi Pengawasan: Sistem ini memungkinkan pengawasan kondisi gearbox secara otomatis, mengurangi ketergantungan pada pengawas manusia.
- Dengan menggabungkan pemrosesan sinyal STFT dan model machine learning, proyek ini memiliki potensi untuk menciptakan solusi yang andal dan efektif dalam deteksi dini kerusakan gearbox, membantu industri meningkatkan efisiensi operasional dan mengurangi biaya yang tidak perlu.


***Solution Statement***
- Data getaran hasil akuisisi dalam domain waktu dengan frekuensi sampling sebesar 30 Hz akan melalui tahapan feature engineering. Pada tahap ini, pemrosesan sinyal menggunakan Short-Time Fourier Transform (STFT) akan dilakukan untuk mengungkap informasi penting yang terkandung dalam setiap kondisi gearbox dari data getaran yang telah diakuisisi. 
- Hasil pemrosesan sinyal akan dilakukan proses ekstraksi fitur menggunakan beberapa parameter statistik seperti Mean, Standard Deviation, Shape Factor, RMS, Peak to Peak, Kurtosis, Skewness, Impulse Factor, Crest Factor, Variance,  dan Form Factor.
- Data hasil ektrasksi fitur ini yang akan dilakukan proses pelatihan dengan menggunakan beberapa modell *machine learning classifier* berbasis pohon keputusan (Tree Based Algorithm) dan Non Tree Based Algorithm.
- Dari beberapa model yang digunakan akan dilakukan perbandingan performa dari setiap model dengan menggunakan evaluasi metrik seperti ***Accuracy***, ***Precision***, ***Recall***, dan ***F1-Score***.


![machine learning workflow drawio (1) drawio (1)](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/d20538e4-7921-4289-a791-5cd3a7e38da0)

Gambar 2. Project Workflow

# Data Understanding
---
Data yang digunakan dalam projek ini adalah data getaran selama gearbox beroperasi, data getaran tersebut diakuisisi menggunakan SpectraQuest’s Gearbox Fault Diagnostics Simulator. Data getaran diakuisisi menggunakan 4 sensor getaran yang ditempatkan pada empat arah berbeda, dan dengan variasi beban dari 0% - 90%. Terdiri dari dua kondisi yaitu Healthy Gear (Normal) dan Broken Tooth (Rusak).

Berikut ini link dataset yang berektensi csv yang akan digunakan pada proyek ini -> [**Dataset Gearbox**](https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis)

***Exploratory Data Analysis (EDA)***

Jumlah Persebaran Data bisa dikatakan dalam kondisi yang seimbang sehingga dapat menghindari bias prediksi dari model
![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/bf102ad6-4f7a-433c-9ed5-6fba11803bdf)

Gambar 3. Jumlah Persebaran Data

# Feature Engineering
--- 
**Short-Time Fourier Transform (STFT)**

**Healthy Gear (Normal)**

![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/dafdd328-dd9a-44e0-8c0e-3a9b6864642b)

![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/0b7a770c-ae60-4e31-b231-00c92687d74d)

![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/6d5c3471-aa6b-46c2-9b74-4454b1f07608)



**Broken Tooth (Normal)**

![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/7a636b2a-412e-4034-bef1-b857a774f291)


![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/c334639b-1766-4e73-b054-8276ee96ce00)


![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/2ade22b5-dfb7-4457-a33a-e95101477691)

Gambar 4. Transformasi Domain Waktu menjadi Domain Waktu-Frekuensi dengan (STFT)

$$ X(\tau, f) = \int_{-\infty}^{\infty} x(t) w(t - \tau) e^{-j 2 \pi f t} dt $$

Short-Time Fourier Transform (STFT) adalah metode pemrosesan sinyal yang digunakan untuk menganalisis bagaimana spektrum frekuensi dari suatu sinyal berubah seiring waktu. Metode ini berguna untuk mengungkap informasi frekuensi yang berubah-ubah dalam sinyal waktu-domain, seperti dalam kasus data getaran dari gearbox yang ingin dianalisis. STFT mengizinkan identifikasi perubahan frekuensi dalam sinyal sepanjang waktu, yang sangat berguna untuk deteksi kerusakan atau perubahan kondisi dalam peralatan seperti gearbox. Dalam praktiknya, STFT memungkinkan kita untuk mengamati bagaimana distribusi frekuensi dalam sinyal berubah seiring waktu, sehingga kita dapat mengidentifikasi pola yang berkaitan dengan kondisi gearbox yang berbeda.

Dari hasil transformasi diatas dapat dilihat bahwa spektrum STFT pada gear rusak memiliki intensitas yang tinggi dapat dilihat dari gambar spektrum. Terjadi peningkatan atau perubahan yang mencolok pada spektrum frekuensi pada titik-titik tertentu jika dibandingkan dengan spektrum STFT dari gearbox dalam kondisi normal. Peningkatan intensitas atau perubahan dalam spektrum frekuensi memberikan indikasi kuat tentang adanya kondisi tidak normal pada gearbox. Ini membantu dalam deteksi dini kerusakan atau perubahan kondisi sehingga tindakan pencegahan atau perbaikan dapat diambil lebih awal.


**Ekstraksi Fitur Statistik**

Tabel 1. Parameter Statistik

| Parameter Statistik  | Persamaan |
|--------|---------------|
| Mean  | ![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/d5028d55-6cc4-424f-8ac9-677e936e6ef9)            |
| Standard Deviation  | ![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/e07491ad-22a3-4a18-88a1-ae4ef6cef808)             |
| Shape Factor  | ![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/fa4b6bd8-aa12-456a-9f0b-7f02a1d4bbd3)             |
| RMS  | ![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/b951d49a-4e89-4eca-b7bf-c16cf413128c)             |
| Peak to Peak   | ![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/e99be618-3d5b-4481-b84d-1ea61a0ea644)             |
| Kurtosis      | ![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/4d2608eb-8de0-4542-8894-c2163d5fcc3c)             |
| Skewness | ![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/95d4121d-c420-420a-bbb2-7c63552582a4)             |
| Impulse Factor | ![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/bdd56230-9a7d-4b72-b8cb-a1e7ed6b6d9b)            |
| Crest Factor  | ![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/121d88f0-ed75-42b5-8ad1-9e1f32c316ef)             |
| Variance | ![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/c4261b19-8630-4b23-b235-6d727d12cc72)            |
| Form Factor  | ![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/6bcabf2a-94ec-4fa4-a633-1a847fcfba58)            |


# Data Preparation
--- 

- Setelah melakukan tahapan Feature Engineering, data hasilnya akan digunakan dalam proses pelatihan model machine learning. Dataset ini terdiri dari 2009 baris, di mana setiap baris memiliki 44 Kolom Fitur yang merupakan hasil ekstraksi dari data getaran dalam 4 arah akuisisi, dan 1 Kolom Target yang mencerminkan Kondisi Bearing.

- Data Target akan mengalami proses label encoder, sebuah teknik yang mengubah nilai-nilai kategorik menjadi representasi numerik. Hal ini penting karena beberapa algoritma machine learning memerlukan input numerik, dan label encoder digunakan untuk mengubah nilai-nilai kategori menjadi format yang dapat dimengerti oleh model.

Tabel 2. Hasil Label Encoder
  
| Kelas  | Nilai Numerik |
|--------|---------------|
| broken  | 0             |
| healthy  | 1             |


- Inisiasi Data dilakukan sebelum melakukan proses splitting, di mana variabel (X) akan mewakili fitur-fitur data yang terdiri dari 44 kolom, sementara variabel (y) akan mewakili data target (kondisi fault).

- Proses Splitting Data dilakukan untuk membagi dataset menjadi dua bagian, yakni data training (untuk melatih model) dan data testing (untuk menguji performa model). Pembagian ini dilakukan dengan rasio 80% data training dan 20% data testing, dengan penggunaan nilai random_state = 42.

- Setelah pembagian, jumlah data training menjadi 1607, sedangkan data testing berjumlah 402.

- Data akan dibagi menjadi beberapa variabel, yaitu X_train (fitur data training), y_train (target data training), X_test (fitur data testing), dan y_test (target data testing).


# Modelling
--- 

Dalam proyek ini, akan dilibatkan dua jenis algoritma untuk melatih model. Pertama, algoritma berbasis pohon keputusan atau tree-based seperti decision tree, adaboost, gradient boosting, xgboost, catboost, histboost, LGBM, dan random forest. Selain itu, juga akan digunakan algoritma yang tidak berbasis pohon keputusan, seperti SVM, KNN, Naive Bayes, dan logistic regression. Setiap model ini akan melewati fase pelatihan dengan menggunakan perintah .fit pada dataset X_train dan y_train.

Hasil dari pelatihan model akan dievaluasi menggunakan metode Cross-Validation. Setiap model machine learning akan diberi instruksi untuk menjalani pelatihan dengan menggunakan parameter default yang telah ditetapkan.

# Evaluation
--- 
Dalam proyek klasifikasi biner ini akan menggunakan evaluasi metrik seperti ***Accuracy***, ***Precision***, ***Recall***, dan ***F1-Score***.

***Accuracy***

$$Accuracy = {TP + TN\over TP + TN + TN + FP}$$

***Precision***
$$Precision = {TP\over TP + FP}$$ 

***Recall***
$$Recall = {TP\over TP + FN}$$

***F1-Score***
$$Recall = {2×precision×recall\over precision + recall}$$

**Berdasarkan hasil pelatihan didapatkan nilai evaluasi dari model berbasis Tree Based Algorithm :**

Tabel 3. Hasil Evaluasi Model Tree Based Algorithm

| Peringkat | Model                   | Testing Accuracy | Training Accuracy | Precision | Recall    | F1-Score  |
|-----------|-------------------------|------------------|-------------------|-----------|-----------|-----------|
|         1 |           Random Forest |        99.900498 |         99.875583 | 99.907180 | 99.905424 | 99.900443 |
|         2 |                CatBoost |        99.900498 |         99.813278 | 99.954128 | 99.902892 | 99.950197 |
|         3 |      Histogram Boosting |        99.850746 |         99.937695 | 99.860670 | 99.852233 | 99.950197 |
|         4 | Light Gradient Boosting |        99.850622 |         99.689247 | 99.802023 | 99.760684 | 99.701326 |
|         5 |                 XGBoost |        99.800995 |         99.813278 | 99.861727 | 99.852233 | 99.800940 |
|         6 |           Decision Tree |        99.701368 |         99.813278 | 99.616045 | 99.756960 | 99.651678 |
|         7 |                AdaBoost |        99.651617 |         99.813278 | 99.671271 | 99.705147 | 99.701430 |
|         8 |       Gradient Boosting |        99.601866 |         99.813278 | 99.759930 | 99.755380 | 99.751190 |

Berikut ini hasil komparasi dari beberapa model Tree Based Algorithm dengan visualisasi *bar chart* pada setiap metrik evaluasi:

![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/29d3b9be-decf-4809-a058-72599c24a083)

Gambar 5. Bar Chat Hasil Evaluasi Model Tree Based Algorithm

**Berdasarkan hasil pelatihan didapatkan nilai evaluasi dari model berbasis Non Tree Based Algorithm :**

Tabel 4. Hasil Evaluasi Model Non Tree Based Algorithm

| Peringkat | Model               | Testing Accuracy | Training Accuracy | Precision | Recall    | F1-Score  |
|-----------|---------------------|------------------|-------------------|-----------|-----------|-----------|
|         1 | Logistic Regression |        96.763936 |         96.763608 | 97.079641 | 96.965538 | 96.860207 |
|         2 |         Naive Bayes |        95.719284 |         96.203827 | 95.971196 | 95.726608 | 95.711739 |
|         3 |                 SVM |        93.379363 |         92.781873 | 93.612119 | 92.907399 | 93.124392 |
|         4 |                 KNN |        67.994442 |         66.271550 | 68.122009 | 67.162999 | 67.828832 |

Berikut ini hasil komparasi dari beberapa model Non Tree Based Algorithm dengan visualisasi *bar chart* pada setiap metrik evaluasi:

![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/f7b1d827-b7a3-4a5a-94c9-39e0bb572a64)

Gambar 6. Bar Chat Hasil Evaluasi Model Non Tree Based Algorithm

Hasil dari projek menunjukkan bahwa algoritma berbasis pohon keputusan memiliki kinerja yang lebih baik dalam mengklasifikasikan kondisi gearbox dibandingkan dengan algoritma yang tidak berbasis pohon keputusan. Algoritma berbasis pohon keputusan menghasilkan nilai akurasi atau nilai metrik evaluasi yang lebih tinggi, dengan nilai rata-rata sekitar 99%. Ini menunjukkan bahwa model-model seperti decision tree, adaboost, gradient boosting, xgboost, catboost, histboost, LGBM, dan random forest berhasil mengenali dan membedakan kondisi-kondisi berbeda dari gearbox dengan sangat baik.

Ada beberapa alasan mengapa algoritma berbasis pohon keputusan memiliki performa yang baik dalam kasus biner klasifikasi ini, di mana dua kondisi dari gearbox harus diidentifikasi:

- Kemampuan Menangkap Pola Kompleks: Algoritma berbasis pohon keputusan mampu menangkap pola kompleks dalam data. Fitur-fitur hasil ekstraksi dari data getaran, yang mencerminkan berbagai aspek kondisi gearbox, mungkin memiliki pola yang cukup kompleks dan non-linear. Algoritma pohon keputusan dapat memecah masalah ini menjadi keputusan-keputusan lebih sederhana yang membentuk pohon keputusan yang kompleks, memungkinkan pengambilan keputusan yang akurat.

- Overfitting Control: Algoritma pohon keputusan sering kali dilengkapi dengan mekanisme pengendalian overfitting, seperti pruning dalam decision tree atau pengaturan parameter dalam ensemble tree-based algorithms. Hal ini membantu mencegah model beradaptasi terlalu erat dengan data training sehingga dapat menggeneralisasi dengan baik pada data baru, termasuk data testing.

- Kemampuan Mengatasi Ketergantungan dengan Interaksi Fitur: Kondisi gearbox dapat dipengaruhi oleh interaksi kompleks antara berbagai fitur. Algoritma berbasis pohon keputusan cenderung lebih baik dalam menangani ketergantungan semacam ini, karena cabang-cabang yang terbentuk dalam pohon keputusan dapat mewakili interaksi-fitur ini.

- Ensemble Methods: Algoritma ensemble seperti random forest, gradient boosting, dan sejenisnya menggabungkan prediksi dari beberapa pohon keputusan untuk meningkatkan kinerja. Mekanisme ini memberikan lebih banyak variasi dan pembobotan terhadap keputusan, meningkatkan akurasi dan kemampuan generalisasi.

Dengan kombinasi dari keempat faktor di atas, algoritma berbasis pohon keputusan cenderung memberikan hasil yang sangat baik dalam kasus klasifikasi dua kondisi dari gearbox. Hal ini tercermin dalam nilai akurasi atau metrik evaluasi yang tinggi dalam proyek ini.

**Feature Importance**

![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/622178d4-849f-4b97-87e2-ef8749900f34)

Gambar 7. Feature Importance

Dari visualisasi diatas didapatkan fitur-fitur dari data yang sangat berpengaruh atau penting pada proses pelatihan model.

# Conclusion
--- 

Dalam proyek ini, tujuan utama adalah mengembangkan sistem deteksi kerusakan pada gearbox dengan menggabungkan analisis sinyal menggunakan Short-Time Fourier Transform (STFT) dan model machine learning. Hasil dari proyek ini dapat diringkas sebagai berikut:

- Pemrosesan Sinyal STFT: Pemrosesan sinyal STFT digunakan untuk mengonversi sinyal getaran gearbox dari domain waktu menjadi domain frekuensi. Ini memungkinkan deteksi pola frekuensi yang mengindikasikan adanya kerusakan atau kondisi buruk pada gearbox.

- Feature Engineering: Dari hasil pemrosesan sinyal STFT, dilakukan ekstraksi 44 kolom fitur dari 4 arah akuisisi. Ini memberikan representasi fitur-fitur yang menggambarkan aspek getaran gearbox yang bervariasi.

- Pemilihan Algoritma: Dua jenis algoritma digunakan: algoritma berbasis pohon keputusan dan algoritma non-pohon keputusan. Algoritma pohon keputusan, termasuk random forest, gradient boosting, dan lainnya, memiliki kinerja yang lebih baik dalam mengklasifikasikan kondisi gearbox.

- Label Encoder: Data target, yang mencerminkan kondisi bearing, telah diubah menjadi bentuk numerik menggunakan label encoder agar sesuai dengan kebutuhan algoritma machine learning.

- Pelatihan dan Evaluasi: Setiap model machine learning dilatih menggunakan data X_train dan y_train. Hasil pelatihan dievaluasi dengan menggunakan metode Cross-Validation. Algoritma pohon keputusan menunjukkan nilai akurasi rata-rata sekitar 99%, menandakan kemampuannya dalam mengklasifikasikan dua kondisi dari gearbox.

- Kelebihan Algoritma Pohon Keputusan: Algoritma berbasis pohon keputusan memiliki kemampuan untuk menangkap pola kompleks, mengatasi interaksi fitur, dan mengontrol overfitting. Ensemble methods yang digunakan oleh beberapa algoritma pohon keputusan juga meningkatkan kinerja model.

Dalam kesimpulan, proyek ini berhasil mengembangkan solusi yang andal dalam mendeteksi dan mengklasifikasikan kondisi gearbox menggunakan analisis sinyal STFT dan model machine learning. Hasil yang paling menonjol adalah kinerja unggul dari algoritma berbasis pohon keputusan, yang menghasilkan nilai akurasi yang tinggi dalam mengenali kondisi normal dan kerusakan pada gearbox. Dengan menggabungkan pemahaman tentang getaran gearbox dan teknik analisis sinyal dengan kekuatan algoritma machine learning, proyek ini berpotensi memberikan manfaat besar dalam pemeliharaan peralatan industri.

