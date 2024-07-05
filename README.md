# Laporan Proyek Machine Learning - Moh Fatchurrohman

## Domain Proyek

Domain proyek ini akan membahas mengenai permasalahan dalam bidang otomotif yang dibuat untuk mengetahui prediksi harga mobil bekas produsen mobil Ford (*Used Car*) berdasarkan data model mobil Ford, tahun registrasi, harga, transmisi, jarak yang ditempuh, jenis bahan bakar, pajak jalan, konsumsi bahan bakar, dan ukuran mesin.

<img src="https://thumb.viva.co.id/media/frontend/thumbs3/2009/12/11/81573_ford_fiesta_saat_dikenalkan_di_thailand_665_374.jpg">

Seiring meningkatnya jumlah produksi mobil baru dan munculnya berbagai model berdampak dengan melimpahnya mobil bekas, yang dapat menciptakan peluang bisnis bagi pembeli dan penjual. Adapun alasan lain mengapa masyarakat maupun pelaku bisnis lebih memilih mobil bekas dikarenakan pajak yang lebih terjangkau dan proses peralihan kepemilikan yang tidak sulit, sehingga menjadikan bisnis ini menjadi lebih menjanjikan. Mobil bekas memiliki harga jual yang menurun secara periodik, oleh karena itu estimasi harga jual mobil bekas sangat diperlukan dalam membantu pengusaha untuk menentukan harga jual. [[1]](https://repository.nusaputra.ac.id/id/eprint/590/1/YusdiansyaHaryadiArsyaf_TI22.pdf 'Mobil Bekas') 

Pasar mobil bekas di Indonesia dari tahun ke tahun terus menunjukkan tren positif (meningkat). Pesatnya pertumbuhan sektor industri otomotif nasional dan melonjaknya angka penjualan mobil baru juga ikut mendorong penjualan mobil bekas di Indonesia. Konsumen dengan dana yang terbatas dinilai lebih tertarik membeli mobil bekas dari pada mobil baru. Apalagi mobil bekas dengan harga yang cukup menggiurkan dengan kualitas bagus. Tren ini bisa mengalihkan selera masyarakat yang akan membeli mobil karena pertimbangan harga yang murah dan terjangkau serta fasilitas dan keunggulan yang ditawarkan oleh mobil bekas tersebut. Munculnya mobil bekas ini akan menjadi tren tersendiri dan membuat segmen baru di masyarakat, khususnya masyarakat kelas menengah baru yang ingin memiliki mobil.[[2]](http://repo.unand.ac.id/907/3/bab%25201.pdf 'Pasar Mobil Bekas')

Ford berdiri sejak 1903 dan menjadi salah satu merek otomotif tertua yang terus eksis hingga kini. Sementara di Indonesia mobil Ford telah hadir di Indonesia sejak 1950an. Nama Ford dengan banyaknya merek tersebut pudar seiring masuknya produsen Jepang pada 1970an. Ford baru kembali hadir pada medio 2000 melalui perusahaan baru Ford Motor Indonesia. Lalu baru pada 2001, perusahaan mulai berbisnis dengan menghadirkan sejumlah model anyar dari Ford di Tanah Air. Sejumlah model yang diniagakan Ford, antaranya Ranger, Everest, Focus, Escape Sporty, Fiesta, hingga EcoSport. Sedangkan data penjualan terakhir perusahaan pada 2015, berhasil melego sebanyak 4.986 unit.[[3]](https://www.gaikindo.or.id/ford-kembali-ke-indonesia-hadirkan-ranger-dan-everest/ 'Mobil Ford')

Dengan berdasarkan data dan latar belakang di atas, maka di dalam proyek ini akan dibuat sebuah model *machine learning* untuk melakukan analisis prediksi terhadap harga mobil bekas Ford. Dengan adanya model *machine learning* yang telah dibangun, diharapkan dapat membantu dalam memperkirakan besarnya harga jual mobil bekas Ford tersebut.

# Business Understanding

## Problem Statements

Berdasarkan latar belakang yang telah dijelaskan di atas, maka diperoleh rumusan masalah yang akan diselesaikan pada proyek ini, yaitu:
1. Bagaimana cara melakukan tahap persiapan data sebelum digunakan untuk membuat model *machine learning*?
2. Bagaimana cara membuat model *machine learning* untuk melakukan prediksi harga jual mobil bekas Ford?

## Goals

Berdasarkan rumusan masalah yang telah dipaparkan di atas, maka didapatkan tujuan dari proyek ini, yaitu:
1. Melakukan tahap persiapan data (*data preparation*) sehingga data dapat digunakan pada model *machine learning* dengan baik.
2. Membuat model *machine learning* untuk melakukan analisis prediksi harga jual mobil bekas Ford dengan tingkat *error* yang cukup rendah.

## Solution Statements

Berdasarkan penjelasan di atas, terdapat beberapa solusi yang dapat dilakukan untuk dapat mencapai tujuan dari proyek ini, yaitu:
1. Tahap persiapan data (*data preparation*) dapat dilakukan dengan beberapa teknik, sebagai berikut:
   - Melakukan pembagian data menjadi 2, yaitu data latih (*training data*) dan data uji (*testing data*) dengan perbandingan rasio sebesar 90 : 10 yang akan digunakan ketika membangun model *machine learning*.
   - Melakukan standarisasi nilai pada data fitur numerik untuk mencegah terjadinya penyimpangan nilai data yang cukup besar.
2. Tahap pembuatan model *machine learning* akan digunakan 3 model dengan algoritma *machine learning* yang berbeda. Algoritma yang akan digunakan adalah K-Nearest Neighbor Algorithm, Random Forest Algorithm, dan Adaptive Boosting Algorithm. Dari ketiga model tersebut akan dilakukan evaluasi performa dan kinerja masing-masing algoritma dan akan dipilih satu algoritma yang memberikan hasil prediksi yang terbaik.
   - **Algoritma K-Nearest Neighbor**  
     Sesuai dengan namanya, yaitu "sejumlah k-tetangga terdekat" adalah algoritma *machine learning* yang tergolong ke dalam *supervised learning* yang bekerja dengan cara mengelompokkan data berdasarkan kemiripan antar data baru dengan sejumlah data (k) yang terdekat. [[4]](https://geospasialis.com/k-nearest-neighbor 'Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning') Cara kerja algoritma K-Nearest Neighbor, sebagai berikut: [[4]](https://geospasialis.com/k-nearest-neighbor 'Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning')
     - Tentukan jumlah tetangga terdekat (`k`) yang akan dipertimbangkan sebagai dasar klasifikasi.
     - Hitung jarak antara data baru terhadap semua titik data dalam *dataset* (tetangga terdekat).
     - Urutkan jarak pada dari kecil ke besar, lalu diambil titik data dengan jarak terkecil dari sejumlah `k` titik.
     - Hitung jumlah titik pada `k` setiap kelas atau kategori.
     - Masukkan data baru ke kelas dengan jumlah `k` terbanyak.
     
     <br>
     <img src="https://user-images.githubusercontent.com/64983961/188507827-0f729ab6-61a5-4dbc-9be2-afa424f6c294.png" alt="Ilustrasi Algoritma K-Nearest Neighbor" title="Ilustrasi Algoritma K-Nearest Neighbor">
     
     Perhitungan jarak ke tetangga terdekat dapat dilakukan dengan menggunakan metrik sebagai berikut:
     - *Euclidean distance*
       $$d(x,y)=\sqrt{\sum_{i=1}^n (x_i-y_i)^2}$$
     - *Manhattan distance*
       $$d(x,y)=\sum_{i=1}^n |x_i-y_i|$$
     - *Hamming distance*
       $$d(x,y)=\frac{1}{n}\sum_{n=1}^{n=n} |x_i-y_i|$$
     - *Minkowski distance*
       $$d(x,y)=\left(\sum_{i=1}^n |x_i-y_i|^p\right)^\frac{1}{p}$$
     
     Adapun kelebihan dari algoritma K-Nearest Neighbor, yaitu: [[4]](https://geospasialis.com/k-nearest-neighbor 'Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning')
     - Sangat sederhana dan mudah untuk dipahami
     - Sangat mudah dalam penerapannya
     - Dapat digunakan dalam kasus klasifikasi maupun regresi
     - Dapat digunakan dalam jumlah kelas yang berbeda-beda
     - Tidak memerlukan proses trainig dan pembangunan model, karena data baru secara langsung akan dikelaskan
     - Mudah jika ingin untuk melakukan penambahan data
     - Parameter yang dibutuhkan hanya sedikit, yaitu jumlah k-tetangga (`n_neighbors`), dan metode perhitungan metrik jaraknya (`metric`) [[5]](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html 'sklearn.neighbors.KNeighborsRegressor')
     - Hasil pemodelan tidak linear, sehingga cocok untuk klasifikasi data yang batasannya tidak linear.
     
     Adapun kelemahan dari algoritma K-Nearest Neighbor, yaitu: [[4]](https://geospasialis.com/k-nearest-neighbor 'Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning')
     - Perlu untuk menentukan nilai `k` yang tepat
     - *Computation cost* yang cukup tinggi
     - Waktu pemrosesan akan berlangsung lama jika *dataset* yang digunakan sangat besar
     - Kurang bagus untuk diterapkan pada *high dimensional data*
     - Sangat sensitif pada data yang memiliki banyak *noise* (*noisy data*), data yang hilang (*missing data*), dan data dengan nilai yang ekstrem serta kemunculannya yang jarang (*outliers*).
     
   - **Algoritma Random Forest**  
     Metode Random Forest merupakan jenis algoritma *supervised learning* dan termasuk ke dalam metode Decision Tree yang menggunakan kombinasi dari masing-masing model tree dan akan digabungkan menjadi sebuah model dalam membuat hasil prediksi akhir. Algoritma Random Forest menggunakan teknik *bagging* (*bootstrap aggregating*), di mana beberapa model akan dilatih dengan cara *random sampling with replacement*. [[6]](https://machinelearning.mipa.ugm.ac.id/2018/07/28/random-forest 'Random Forest')
     
     <img src="https://user-images.githubusercontent.com/64983961/188504775-b7e4aa9b-f1cd-41ef-8a70-a977db8f3d60.png" alt="Ilustrasi Algoritma Random Forest" title="Ilustrasi Algoritma Random Forest">
     
     Setelah dilakukan pelatihan, prediksi untuk sampel yang tidak terlihat ($x'$) dapat dibuat dengan menghitung rata-rata prediksi dari semua pohon setiap individu model pada $x'$. [[7]](https://en.wikipedia.org/wiki/Random_forest#Bagging 'Random Forest - Bagging')
     $$\hat{f}=\frac{1}{B}\sum_{b=1}^{B} f_b(x^{'})$$
     
   - **Algoritma Adaptive Boosting**  
     Algoritma Adaptive Boosting atau biasanya disingkat AdaBoost merupakan algoritma yang melakukan pelatihan model secara berurutan dan dengan proses iteratif atau berulang. Data latih (*training data*) akan mempunyai bobot atau *weight* yang sama, kemudian model akan melakukan pemeriksaan. Bobot yang lebih tinggi akan dimasukkan ke dalam model yang salah, sehingga akan lanjut ke tahap selanjutnya. Proses iteratif tersebut akan terus berlanjut hingga model mencapai tingkat akurasi yang diinginkan.
     
     <img src="https://user-images.githubusercontent.com/64983961/188507801-30224052-cac2-4e99-9c67-2aec18de8e59.png" alt="Ilustrasi Algoritma Adaptive Boosting" title="Ilustrasi Algoritma Adaptive Boosting">
     
     Algoritma AdaBoost mengacu kepada metode tertentu untuk melakukan pelatihan *classifier* yang di-*boosted*. Pengklasifikasian tersebut adalah pengklasifikasian dalam bentuk, [[8]](https://en.wikipedia.org/wiki/AdaBoost#Training 'AdaBoost - Training')
     $$F_T(x)=\sum_{t=q}^{T}f_t(x)$$
     di mana setiap $F_T$ adalah *learner* yang lemah yang mengambil objek $x$ sebagai input dan mengembalikan nilai yang menunjukkan kelas objek. Demikian juga pada pengklasifikasi $T$ merupakan nilai positif jika sampel berada dalam kelas positif, dan negatif jika sebaliknya.

## Data Understanding

<img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/100%2C000%20UK%20Used%20Car%20Data%20set.PNG" alt="100,000 UK Used Car Data set" title="100,000 UK Used Car Data set" width="100%">

Data yang digunakan dalam proyek ini adalah *dataset* yang diambil dari Kaggle Dataset [100,000 UK Used Car Data set](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes '100,000 UK Used Car Data set'). Dalam *dataset* tersebut terdapat sebuah *file* atau berkas dengan nama `ford.csv` yang berekstensi (*file format*) `.csv` atau [comma-separated values](https://en.wikipedia.org/wiki/Comma-separated_values 'Comma-separated values') berukuran 933.29 kB.

Dari *dataset* tersebut, dilakukan proses *Exploratory Data Analysis* (EDA) sebagai investigasi awal untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data dengan menggunakan teknik statistik dan representasi grafis atau visualisasi.

1. **Deskripsi Variabel**  
   Berikut adalah informasi mengenai variabel-variabel yang terdapat pada *dataset* *100,000 UK Used Car Data set* adalah sebagai berikut,
   
   <img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Deskripsi%20Variabel.PNG" alt="Deskripsi Variabel" title="Deskripsi Variabel">
   
   Dari gambar di atas dapat dilihat bahwa terdapat 17.965 baris data dan 9 kolom atribut atau fitur. Di antaranya adalah dua (2) atribut/variabel dengan tipe data `float64 non-null` dan empat (4) atribut/variabel dengan tipe data `int64 non-null`, dan tiga (3) atribut/variabel dengan tipe data `object non-null`. Berikut adalah keterangan untuk masing-masing variabel,
   - `model` : Model mobil Ford
   - `year` : Tahun registrasi
   - `price` : Harga
   - `transmission` : Transmisi
   - `mileage` : Jarak yang tempuh
   - `fuelType` : Jenis bahan bakar
   - `tax` : Pajak jalan
   - `mpg` : Mil per galon
   - `engineSize` : Ukuran mesin (dalam liter)
   
2. **Deskripsi Statistik**  
   
   <img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Deskripsi%20Statistik.PNG" alt="Deskripsi Statistik" title="Deskripsi Statistik">
   
3. **Menangani Missing Value**  
   
   <img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Jumlah%20Nol.PNG" alt="Menangani Missing Value" title="Menangani Missing Value">
   
   Berdasarkan gambar tersebut, terdapat adanya nilai 0 di kolom `tax` ada 2153 dan di kolom `engineSize ` ada 51. Kemudian drop baris dengan nilai `tax` dan `engineSize ` yang terdapat nilai 0.
   
4. **Menangani Outliers**  
   *Outliers* merupakan sampel data yang nilainya berada sangat jauh dari cakupan umum data utama yang dapat merusak hasil analisis data. Berikut adalah visualisasi *boxplot* untuk melakukan pengecekan keberadaan *outliers*.
   
   <img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Menangani%20Outliers%20Sebelum.PNG" alt="Menangani Outliers - Sebelum" title="Menangani Outliers - Sebelum">
     
   Berdasarkan gambar tersebut, terdapat *outliers* pada fitur `year`, `price`, `mileage`, `tax`, `mpg`, dan `engineSize`. Sehingga dilakukan proses pembersihan *outliers* dengan metode IQR (*Inter Quartile Range*).
   
   $$IQR=Q_3-Q_1$$
   
   Kemudian membuat batas bawah dan batas atas untuk mencakup *outliers* dengan menggunakan,
   
   $BatasBawah=Q_1-1.5*IQR$
   
   $BatasAtas=Q_3-1.5*IQR$
   
   
   Setelah dilakukan pembersihan *outliers*, dilakukan kembali visualisasi *outliers* untuk melakukan pengecekan kembali sebagai berikut,
   
   <img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Menangani%20Outliers%20Sesudah.PNG" alt="Menangani Outliers - Sesudah" title="Menangani Outliers - Sesudah">
   
   Dari gambar di atas dapat dilihat bahwa *outliers* telah berkurang. Meskipun *outliers* masih terdapat pada fitur `year`, `mileage`, dan `engineSize`, tetapi masih dalam batas aman.
   
5. **Univariate Analysis**  
   Melakukan proses analisis data *univariate* pada fitur-fitur numerik. Proses analisis ini menggunakan bantuan visualisasi histogram untuk masing-masing fitur numerik.
   
   <img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Univariate%20Analysis.png" alt="Univariate Analysis" title="Univariate Analysis">
   
   Dari data histogram di atas diperoleh informasi, yaitu:
   - Penjualan mobil Ford paling banyak pada tahun 2018.
   - Harga penjualan mobil Ford sebagian besar berada pada rentang 9500 sampai 18000, dan paling tinggi di harga sekitar 11000.
   - Jarak tempuh pada mobil Ford yang terjual sebagian besar berada pada rentang 0 mile hingga 18000 mile, dan jarak tempuh yang paling panjang berada pada sekitar 10000 mile.
   - Konsumsi bahan bakar (mil per galon) dari mobil Ford yang terjual sebagian besar pada rentang 55 mpg hingga 68 mpg, dan konsumsi bahan bakar dari mobil Ford paling banyak hingga 68 mpg.
   - Ukuran mesin yang paling banyak adalah 1.0.
   - Pendapatan pajak yang paling besar adalah 145.
   
6. **Multivariate Analysis**  
   Melakukan visualisasi distribusi data pada fitur-fitur numerik dari *dataframe* `epower`. Visualisasi dilakukan dengan bantuan *library* `seaborn` `pairplot` menggunakan parameter `diag_kind`, yaitu `kde`, untuk melihat perkiraan distribusi probabilitas antar fitur numerik.
   
   <img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Multivariate%20Analysis.png" alt="Multivariate Analysis" title="Multivariate Analysis">
   
7. **Correlation Matrix with Heatmap**  
   Melakukan pengecekan korelasi antar fitur numerik dengan menggunakan visualisasi diagram *heatmap* *correlation matrix*.
   
   <img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Correlation%20Matrix%20with%20Heatmap.png" alt="Correlation Matrix with Heatmap" title="Correlation Matrix with Heatmap">
   
   Dapat dilihat pada diagram *heatmap* di atas memiliki *range* atau rentang angka dari 1.0 hingga 0.4 dengan keterangan sebagai berikut,
   - Jika semakin mendekati 1, maka korelasi antar fitur numerik semakin kuat bernilai positif.
   - Jika semakin mendekati 0, maka korelasi antar fitur numerik semakin rendah.
   - Jika semakin mendekati -1, maka korelasi antar fitur numerik semakin kuat bernilai negatif.
   
   Jika korelasi bernilai positif, berarti nilai kedua fitur numerik cenderung meningkat bersama-sama.  
   
   Jika korelasi bernilai negatif, berarti nilai salah satu fitur numerik cenderung meningkat ketika nilai fitur numerik yang lain menurun.

8. **Analisis Korelasi Antar Fitur**  
   Fitur `price` memiliki korelasi yang cukup kuat dengan fitur `mileage`, `mpg` dan `engineSize`. Sehingga, fitur `tax` tidak memiliki korelasi dengan fitur `price`. Dengan begitu, dapat dilakukan *drop* (menghapus) fitur-fitur tersebut. Dengan begitu, dapat dilakukan *drop* (menghapus) fitur-fitur tersebut.
   
   <img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Analisis%20Korelasi%20Antar%20Fitur.PNG" alt="Analisis Korelasi Antar Fitur" title="Analisis Korelasi Antar Fitur">

## Data Preparation

Pada tahap persiapan data atau *data preparation* dilakukan berdasarkan penjelasan yang sudah dipaparkan pada bagian [Solution Statements](#solution-statements "Solution Statements"). Tahap ini penting dilakukan untuk mempersiapkan data sehingga dapat digunakan untuk melatih model *machine learning* dengan baik. Berikut adalah dua tahapan data preparation yang dilakukan, yaitu,

1. **Split Data**  
   Pembagian data dilakukan untuk memisahkan data keseluruhan menjadi dua (2) bagian, yaitu data latih (*training data*) dan data uji (*testing data*) dengan perbandingan rasio sebesar 90 : 10 menggunakan `train_test_split`.
   
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)
    ```
    
   Kemudian diperoleh hasil pembagian data masing-masing, yaitu sebagai berikut,
   
    ```python
    Total seluruh sampel : 8401
    Total data train     : 7560
    Total data test      : 841
    ```

2. **Standarisasi pada Fitur Numerik**  
   Standarisasi fitur numerik menggunakan `StandardScaler` untuk mencegah terjadinya penyimpangan nilai data yang cukup besar. Proses standarisasi tersebut dilakukan dengan mengurangkan nilai rata-rata, lalu membaginya dengan standar deviasi atau simpangan baku untuk menggeser distribusi. Proses standarisasi akan menghasilkan distribusi dengan nilai rata-rata menjadi 0, dan nilai standar deviasi menjadi 1.
   
    ```python
    scaler = StandardScaler()
    scaler.fit(X_train[numerical_features])
    X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
    ```
   
   <img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Standarisasi%20pada%20Fitur%20Numerik.PNG" alt="Standarisasi pada Fitur Numerik" title="Standarisasi pada Fitur Numerik">

    ```python
    X_train[numerical_features].describe().round(4)
    ```
   
   <img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Deskripsi%20Statistik%20setelah%20Standarisasi.PNG" alt="Deskripsi Statistik setelah Standarisasi" title="Deskripsi Statistik setelah Standarisasi">

## Modelling

Setelah dilakukannya tahap *data preparation*, selanjutnya adalah melakukan tahap persiapan model terlebih dahulu sebelum mengembangkan model menggunakan algoritma yang telah ditentukan.

Tahap persiapan *dataframe* untuk analisis model menggunakan parameter `index`, yaitu train_mse dan test_mse, serta parameter `columns` yang merupakan algoritma yang akan digunakan untuk melakukan prediksi, yaitu algoritma K-Nearest Neighbor (KNN), Random Forest, dan Adaptive Boosting (AdaBoost).

```python
models = pd.DataFrame(
    index   = ['train_mse', 'test_mse'],
    columns = ['KNN', 'RandomForest', 'Boosting']
)
```

Kemudian terapkan ketiga algoritma ke dalam model tersebut.

1. **K-Nearest Neighbor (KNN) Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `n_neighbors` dengan nilai k = 10 tetangga dan `metric` bawaan, yaitu Euclidean.
   
   ```python
   knn = KNeighborsRegressor(n_neighbors=10)
   ```
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)
   
2. **Random Forest Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `n_estimator` dengan jumlah 50 *trees* (pohon), `max_depth` dengan nilai kedalaman atau panjang pohon 16, `random_state` dengan nilai 55, dan `n_jobs` yang bernilai -1 (pekerjaan dilakukan secara paralel).
   
   ```python
   rf = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
   ```
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)
   
3. **Adaptive Boosting (AdaBoost) Algorithm**  
   Pada algoritma K-Nearest Neighbor digunakan parameter `learning_rate` dengan nilai bobot setiap *regressor* adalah 0.05, dan `random_state` dengan nilai 55.
   
   ```python
   boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
   ```
   
   Kemudian akan dilakukan analisis prediksi *error* menggunakan *Mean Squared Error* (MSE) pada data latih (*training data*) dan data uji (*testing data*)

Ketiga model yang telah dibangun di atas, akan dilakukan pengujian kinerja untuk masing-masing model yang menggunakan algoritma K-Nearest Neighbor, algoritma Random Forest, dan algoritma Adaptive Boosting. Dari ketiga model tersebut akan diperoleh satu (1) model dengan hasil prediksi yang paling baik dan tingkat *error* yang paling rendah.

## Evaluation

Pada tahap evaluasi model, akan dilakukan pengujian untuk melihat algoritma mana yang memberikan hasil prediksi paling baik dan dengan tingkat *error* yang paling rendah. Sebelumnya, akan dilakukan proses standarisasi atau *scaling* pada fitur numerik data uji (*testing data*) agar nilai rata-rata (*mean*) bernilai 0, dan varians bernilai 1.

```python
xTest.loc[:, numericalFeatures] = scaler.transform(xTest[numericalFeatures])
```

Kemudian evaluasi dari ketiga model, yaitu algoritma K-Nearest Neighbor, Random Forest, dan Adaptive Boosting (AdaBoost) untuk masing-masing data latih (*training data*) dan data uji (*testing data*) dengan melihat tingkat *error*-nya menggunakan *Mean Squared Error* (MSE),

$$MSE=\frac{1}{N}\sum_{i=1}^{N} (y_i-y\\_pred_i)^2$$

di mana, nilai $N$ adalah jumlah *dataset*, nilai $y_i$ merupakan nilai sebenarnya, dan $y\\_pred$ yaitu nilai prediksinya.

Penggunaan metode metrik *Mean Squared Error* (MSE) memiliki kelebihan, yaitu cukup sederhana dan mudah dipahami dalam melakukan perhitungan. Meskipun begitu, terdapat kelemahan pada metrik ini, yaitu hasil akurasi prediksi yang kecil karena tidak dapat membandingan hasil peramalan tersebut dengan kenyataannya. []

```python
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN', 'RF', 'Boosting'])
modelDict = {'KNN': knn, 'RF': rf, 'Boosting': boosting}
for name, model in modelDict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=yTrain, y_pred=model.predict(xTrain))/1e3
    mse.loc[name, 'test']  = mean_squared_error(y_true=yTest,  y_pred=model.predict(xTest))/1e3
```

<img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Evaluation.PNG" alt="Evaluation" title="Evaluation">

Dari data tabel tersebut dapat divisualisasikan pada grafik batang berikut.

<img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Evaluation%20Graph.png" alt="Evaluation Graph" title="Evaluation Graph">

Dari visualisasi diagram di atas dapat disimpulkan bahwa,
1. Model dengan algoritma Random Forest memberikan nilai *error* yang paling kecil, yaitu sebesar 498.5 pada *training error*, dan 1542.5 pada *testing error*.
2. Model dengan algoritma Adaptive Boosting memiliki tingkat *error* yang sedang di antara dua algoritma lainnya.
3. Model dengan algoritma K-Nearest Neighbor mengalami *error* yang paling beser dengan nilai *training error* sebesar 9976.8, dan nilai *testing error* sebesar 13319.4.

Selanjutnya adalah pengujian prediksi model dengan menggunakan beberapa nilai harga (*price*) dari data uji (*testing*)

<img src="https://raw.githubusercontent.com/fathrrohmn/predictive_analytics_ford/main/ford%20gambar/Testing%20Model.PNG" alt="Testing Model" title="Testing Model">

Dapat dilihat prediksi pada model dengan algoritma Random Forest memberikan hasi yang paling mendekati dengan nilai `y_true` jika dibandingkan dengan algoritma model yang lainnya.

Nilai `y_true` sebesar **9750** dan nilai prediksi `Random Forest` sebesar **10032.9**.

Kesimpulannya adalah model yang digunakan untuk melakukan prediksi harga mobil bekas produsen mobil Ford (*Used Car*) menghasilkan **tingkat *error* yang paling rendah** dengan menggunakan **algoritma Random Forest** pada model yang telah dibangun.

---

## Referensi

[1] Yusdiansya Haryadi Arsyaf, "Prediksi Harga Mobil Bekas Menggunakan Metode Multivariate Linear Regression", *Skripsi*, 2022, Retrieved from: https://repository.nusaputra.ac.id/id/eprint/590/1/YusdiansyaHaryadiArsyaf_TI22.pdf

[2] N Fardillah, 2019, Retrieved from: http://repo.unand.ac.id/907/3/bab%25201.pdf

[3] Gaikindo, "Ford kembali ke Indonesia, Hadirkan Ranger dan Everest", *Gaikindo*, 2022, Retrieved from: https://www.gaikindo.or.id/ford-kembali-ke-indonesia-hadirkan-ranger-dan-everest

[4] S. Hussein, "Mengenal K-Nearest Neighbor: Algoritma Populer untuk Machine Learning", *GEOSPASIALIS*, 2021, Retrieved from: https://geospasialis.com/k-nearest-neighbor

[5] scikit-learn, "sklearn.neighbors.KNeighborsRegressor", Retrieved from: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

[6] A. Yanuar, "Random Forest", *Universitas Gadjah Mada Menara Ilmu Machine Learning*, 2018, Retrieved from: https://machinelearning.mipa.ugm.ac.id/2018/07/28/random-forest

[7] "Random Forest", Retrieved from: https://en.wikipedia.org/wiki/Random_forest#Bagging

[8] "AdaBoost", Retrieved from: https://en.wikipedia.org/wiki/AdaBoost#Training
