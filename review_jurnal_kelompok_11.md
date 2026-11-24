**Kelompok 11**  
Yeheskiel Kendek - 23051204388 | Galang Krisna Pramudya - 23051204134

# Review Jurnal

## Identitas Jurnal

| Atribut | Keterangan |
| :--- | :--- |
| **Judul** | Comparative analysis of machine learning models for wind speed forecasting: Support vector machines, fine tree, and linear regression approaches |
| **Penulis** | Yousef Altork |
| **Publikasi** | International Journal of Thermofluids |
| **DOI** | `10.1016/j.ijft.2025.101217` |
| **Link** | [Comparative analysis of machine learning models for wind speed forecasting: Support vector machines, fine tree, and linear regression approaches - ScienceDirect](https://www.sciencedirect.com) |

**Link YouTube:** https://youtu.be/lc31hztGTfw

---

## 1. Data
Paper ini menggunakan data meteorologis yang disediakan oleh *National Wind Technology Center* (NWTC) di Boulder, Colorado, Amerika Serikat. Variabel yang digunakan pada dataset ini meliputi: arah angin (*wind direction*), suhu udara (*air temperature*), kelembaban relatif (*relative humidity*), tekanan udara (*atmospheric pressure*), curah hujan (*precipitation*), dan kecepatan angin (*wind speed*) pada ketinggian 50 meter.

Dataset terdiri atas 60 observasi berupa rata-rata bulanan selama 5 tahun (2019 - 2023), setiap dataset berisi 5 input variabel dan 1 output variabel. Dataset dibagi menjadi 70% untuk *training* dan sisanya dibagi sama rata untuk *validation* dan *testing* model. Sebelum pelatihan, data dilakukan normalisasi ke rentang -1 hingga 1.

## 2. Ringkasan
Paper ini bertujuan membandingkan 3 algoritma *machine learning* dalam memprediksi kecepatan angin berdasarkan sejumlah variabel meteorologi. Topik ini relevan dengan pengembangan energi terbarukan karena kecepatan angin berpengaruh terhadap estimasi potensi energi, perencanaan pembangkit listrik tenaga angin, serta pengelolaan operasionalnya.

Penulis menguji 3 pendekatan, yaitu SVM untuk memodelkan hubungan antara fitur meteorologi dan kecepatan angin, Fine Tree yang menggunakan struktur pohon keputusan, serta Linear Regression yang mengasumsikan hubungan linear antara fitur dan target.

Performa model dinilai menggunakan 4 metrik, yaitu RMSE, MSE, MAE, dan $R^{2}$. Hasil eksperimen menunjukkan bahwa Linear Regression menghasilkan performa terbaik, diikuti oleh SVM dan Fine Tree. Linear Regression memperoleh RMSE sebesar 0,2955 dan $R^{2}$ sebesar 0,97.

Penelitian ini juga melakukan analisis sensitivitas untuk mengetahui fitur meteorologi yang paling berpengaruh. Temperatur menjadi fitur yang paling dominan dalam hasil analisis ketiga model. Berdasarkan hasil tersebut, penulis menyimpulkan bahwa model sederhana seperti Linear Regression dapat memberikan hasil yang baik pada dataset yang digunakan.

## 3. Penelitian Sebelumnya
Paper mengulas pendekatan statistik, *machine learning* tradisional, serta metode *deep learning* untuk peramalan kecepatan angin. Beberapa pendekatan yang disebutkan meliputi ARIMA, Support Vector Regression, ANN, Random Forest, LSTM, dan berbagai model hibrida.

Penelitian terdahulu yang diprioritaskan adalah publikasi 5 tahun terakhir dari paper ini diterbitkan (2020-2025). Secara umum, penelitian terdahulu yang dirangkum memberikan 3 landasan:

| Penelitian | Tahun | Pendekatan yang dibahas |
| :--- | :--- | :--- |
| Faniband dan Shaahid | 2020 | Artificial Neural Networks untuk kecepatan angin *forecasting* |
| Geng et al. | 2020 | PCA dan LSTM untuk prediksi kecepatan angin jangka pendek |
| Saeed et al. | 2020 | Model hibrida *bidirectional* LSTM untuk prediksi interval kecepatan angin |
| Shao et al. | 2021 | LSTM yang dioptimalkan dengan *Firework Algorithm* |
| Qian et al. | 2021 | *Attention-based* LSTM untuk prediksi kecepatan angin jangka pendek |
| Lawal et al. | 2021 | Model hibrida CNN-BLSTM |
| Ding et al. | 2023 | Pendekatan EMD-LSTM untuk prediksi arah angin |

## 4. Masalah
Permasalahan utama yang dibahas adalah bagaimana menentukan model *machine learning* yang memberikan hasil prediksi kecepatan angin terbaik berdasarkan data meteorologi. Penulis mengangkat masalah ini karena kecepatan angin merupakan salah satu faktor penting dalam pemanfaatan energi angin, sedangkan perubahan kondisi meteorologi dapat mempengaruhi karakteristik angin.

Penulis berupaya menjawab permasalahan tersebut melalui 3 perbandingan model dengan menggunakan *input feature* dan data yang sama. Dengan pendekatan ini, performa model dapat dibandingkan menggunakan metrik evaluasi yang terukur.

## 5. Kontribusi
1. Membandingkan performa tiga algoritma, yaitu SVM, Fine Tree, dan Linear Regression, untuk prediksi kecepatan angin.
2. Menggunakan data meteorologi selama lima tahun dari NWTC di Boulder, Colorado.
3. Mengevaluasi performa model menggunakan empat metrik, yaitu RMSE, MSE, MAE, dan $R^{2}$.
4. Menganalisis sensitivitas fitur untuk mengidentifikasi variabel meteorologi yang paling berpengaruh.
5. Memberikan rekomendasi mengenai pemilihan model dengan mempertimbangkan akurasi dan kompleksitasnya.

## 6. Metode

### a. Preprocessing dan pembagian data
Penulis menormalisasikan data ke rentang -1 hingga 1, kemudian membaginya menjadi data pelatihan, validasi, dan pengujian dengan proporsi 70:15:15. Implementasi model dilakukan menggunakan MATLAB Machine Learning Toolbox.

### b. Algoritma yang digunakan
1. **Support Vector Machine**  
   Model ini menggunakan pendekatan *Support Vector Regression* untuk memperkirakan nilai kontinu. Dalam formulasi regresi, SVR berusaha memperoleh prediksi dengan kesalahan yang berada dalam batas toleransi tertentu sekaligus mengendalikan kompleksitas model.
2. **Fine Tree**  
   Fine Tree merupakan model berbasis keputusan yang membagi data berdasarkan kondisi pada fitur tertentu. Model ini relatif mudah diinterpretasikan karena prediksinya dapat dijelaskan melalui serangkaian aturan percabangan. Kelemahannya adalah *tree model* dapat sensitif terhadap perubahan data dan berpotensi mengalami *overfitting*.
3. **Linear Regression**  
   Linear Regression memodelkan hubungan antara fitur masukan dan target melalui kombinasi linear. Secara umum dapat ditulis sebagai:
   $$y = \omega_{0} + \omega_{1}x_{1} + \dots + \omega_{n}x_{n} + \epsilon$$
   Model ini memiliki kelebihan berupa kesederhanaan, efisiensi komputasi, dan interpretasi koefisien yang relatif mudah. Namun, model tidak secara langsung menangkap pola nonlinear tanpa transformasi atau penambahan fitur yang sesuai.

### c. Evaluasi model
Penulis menggunakan empat metrik evaluasi:
1. **RMSE:** mengukur akar rata-rata kuadrat kesalahan dan memberi penalti lebih besar terhadap kesalahan yang besar.
2. **MSE:** mengukur rata-rata kuadrat kesalahan prediksi.
3. **MAE:** mengukur rata-rata nilai absolut kesalahan prediksi.
4. **$R^{2}$:** mengukur kemampuan model menjelaskan variasi target relatif terhadap *baseline* rata-rata.

RMSE, MSE, dan MAE yang lebih kecil umumnya menunjukkan kesalahan prediksi yang lebih rendah, sedangkan nilai $R^{2}$ yang lebih tinggi umumnya menunjukkan kecocokan yang lebih baik.

## 7. Hasil dan Pembahasan
Hasil eksperimen menunjukkan bahwa Linear Regression memperoleh performa terbaik dibandingkan SVM dan Fine Tree pada dataset yang digunakan. Hasil kuantitatif yang dilaporkan adalah sebagai berikut:

| Model | RMSE | MSE | MAE | $R^{2}$ |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 0,29555 | 0,087352 | 0,18061 | 0,97 |
| **SVM** | 0,32275 | 0,104170 | 0,21934 | 0,96 |
| **Fine Tree** | 0,44042 | 0,193970 | 0,34410 | 0,93 |

### a. Analisis Performa Model
* **Linear Regression** memperoleh RMSE, MSE, dan MAE terendah serta nilai $R^{2}$ tertinggi. Hasil tersebut menunjukkan bahwa model ini paling sesuai di antara ketiga model dalam eksperimen penulis. Penulis juga melaporkan bahwa prediksi Linear Regression mengikuti nilai aktual dengan relatif dekat dan memiliki residual yang lebih kecil serta seimbang.
* **SVM** menempati posisi kedua. Model ini menghasilkan kesalahan lebih rendah daripada Fine Tree dan dapat menangkap pola yang lebih kompleks. Namun, paper melaporkan adanya penyebaran kesalahan yang sedikit lebih besar pada kecepatan angin yang tinggi.
* **Fine Tree** memperoleh performa terendah berdasarkan keempat metrik. Model ini dapat mengikuti tren umum, tetapi kesalahannya cenderung meningkat pada nilai kecepatan angin yang tinggi. Hal ini mengindikasikan bahwa konfigurasi Fine Tree yang digunakan belum memberikan hasil sebaik dua model lainnya pada dataset tersebut.

### b. Analisis Sensitivitas Fitur
Penulis melaporkan bahwa temperatur merupakan fitur yang paling berpengaruh pada ketiga model. Tekanan atmosfer dan arah angin juga menunjukkan tingkat pengaruh yang cukup besar pada hasil analisis tertentu, sedangkan kelembaban relatif dan presipitasi memiliki nilai pengaruh yang lebih rendah.

## 8. Kesimpulan
Penulis menyimpulkan bahwa Linear Regression memberikan hasil terbaik untuk prediksi kecepatan angin dalam eksperimen yang dilakukan, diikuti oleh SVM dan Fine Tree. Kesimpulan ini didasarkan pada hasil RMSE, MSE, MAE, dan $R^{2}$. Penulis juga mengidentifikasi temperatur sebagai fitur yang paling berpengaruh dalam analisis sensitivitas.

Paper mengakui bahwa penggunaan rata-rata bulanan memiliki keterbatasan karena tidak menggambarkan variasi kecepatan angin dalam interval yang lebih pendek. Oleh karena itu, penggunaan data harian atau per jam menjadi salah satu arah pengembangan yang disarankan.

## 9. Kelebihan
Beberapa kelebihan paper ini adalah:
1. **Topik relevan.** Prediksi kecepatan angin memiliki manfaat dalam perencanaan dan pengelolaan energi terbarukan.
2. **Menggunakan data nyata.** Data meteorologi dari NWTC memberikan konteks eksperimen yang lebih realistis daripada data simulasi semata.
3. **Perbandingan model mudah dipahami.** Tiga algoritma dengan karakteristik berbeda dibandingkan dalam satu penelitian.
4. **Menggunakan beberapa metrik evaluasi.** Penggunaan RMSE, MSE, MAE, dan $R^{2}$ memberikan perspektif yang lebih lengkap daripada hanya mengandalkan akurasi atau satu ukuran kesalahan.
5. **Menyertakan analisis sensitivitas.** Analisis ini membantu pembaca memahami fitur mana yang paling berpengaruh dalam model yang diuji.
6. **Cocok sebagai referensi pembelajaran.** Struktur eksperimen relatif sederhana sehingga dapat dijadikan titik awal untuk mempelajari regresi dan evaluasi model *machine learning*.

## 10. Kekurangan
Kekurangan paper, baik dari sisi metodologi maupun penyajian, meliputi hal-hal berikut:
1. **Jumlah observasi terbatas**  
   Dataset hanya terdiri atas 60 rata-rata bulanan. Jumlah ini membuat evaluasi dan kesimpulan tentang kemampuan generalisasi model menjadi terbatas.
2. **Resolusi temporal kurang memadai**  
   Data bulanan tidak cukup menggambarkan perubahan kecepatan angin dalam jangka pendek. Keterbatasan ini penting karena *forecasting* dapat digunakan untuk kebutuhan operasional yang membutuhkan prediksi lebih sering.
3. **Detail metodologi belum lengkap**  
   Informasi tentang konfigurasi hiperparameter, prosedur pemilihan model, dan cara pembagian data belum dijelaskan secara cukup terperinci untuk memastikan reproduktibilitas eksperimen.
4. **Perbandingan algoritma terbatas**  
   Penelitian hanya menguji tiga algoritma. Penambahan model pembanding seperti Random Forest, Gradient Boosting, atau model deret waktu dapat memberikan gambaran yang lebih luas mengenai performa berbagai pendekatan.
5. **Interpretasi kontribusi fitur perlu diperjelas**  
   Paper melaporkan hasil sensitivitas fitur, tetapi penjelasan mengenai prosedur perhitungan nilai pengaruh tersebut belum cukup terperinci. Karena itu, interpretasinya perlu dibatasi pada hasil analisis yang dilaporkan.
6. **Istilah *forecasting* belum dijelaskan secara operasional**  
   Paper menggunakan istilah *wind speed forecasting*, tetapi pendekatan yang dijelaskan menggunakan variabel meteorologi untuk memprediksi target kecepatan angin tanpa uraian yang jelas tentang fitur historis atau horizon prediksi. Penjelasan mengenai definisi *forecasting* yang digunakan akan memperkuat konsistensi metodologi.
7. **Kejelasan penulisan metode masih dapat ditingkatkan**  
   Bagian Fine Tree membahas kriteria Gini dan *entropy*, yang umumnya diasosiasikan dengan klasifikasi, sementara target penelitian berupa nilai kontinu. Penulis perlu menjelaskan kriteria pembagian yang benar-benar digunakan untuk tugas regresi. Selain itu, penjelasan preprocessing dan evaluasi dapat dibuat lebih sistematis.

## 11. Future Work
*Future work* yang secara eksplisit disebutkan penulis meliputi:
1. Penggunaan data harian atau per jam.
2. Penambahan variabel iklim atau variabel independen lain.
3. Pengembangan *composite models*.
4. Eksplorasi *ensemble learning*.
5. Penggunaan *Genetic Algorithms*.

## 12. Komentar
Secara keseluruhan, paper ini membahas topik yang relevan dengan penerapan *machine learning* pada energi terbarukan. Kelebihannya adalah menggunakan data meteorologi nyata, membandingkan tiga algoritma, serta mengevaluasi hasil menggunakan empat metrik. Penulis menemukan bahwa Linear Regression memberikan performa terbaik dibandingkan SVM dan Fine Tree pada dataset yang digunakan.

Namun, penelitian memiliki keterbatasan berupa jumlah data yang hanya 60 observasi, penggunaan rata-rata bulanan, serta penjelasan konfigurasi model dan validasi yang belum terperinci. Hasil penelitian juga belum cukup untuk menyimpulkan bahwa Linear Regression selalu menjadi model terbaik untuk prediksi kecepatan angin.

Penelitian selanjutnya dapat menggunakan data per jam, menambahkan fitur historis, dan membandingkan lebih banyak algoritma. Secara umum, paper ini cukup baik sebagai referensi pembelajaran *machine learning*, tetapi masih memerlukan pengujian lebih lanjut agar kesimpulannya dapat digeneralisasikan.