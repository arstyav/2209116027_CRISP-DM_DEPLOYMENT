import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import joblib 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# def page1():
    

#     # Load dataset

#     def load_data():
#         # Ganti dengan path file dataset Anda
#         data = pd.read_csv("Data-Before-Mapping.csv")
#         return data

#     data = load_data()

#     # Header
#     st.title("NBA Player Clustering Dashboard")

#     # Sidebar
#     st.sidebar.title("Clustering Settings")
#     n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=4)

#     # Data preprocessing
#     numeric_data = data.select_dtypes(include=['float64', 'int64'])
#     X = numeric_data.dropna()

#     # Fit KMeans clustering model
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(X)

#     # Add cluster labels to original data
#     data['Cluster'] = kmeans.labels_

#     # Pilar 1: Statistik Pemain Secara Keseluruhan
#     st.header("Player Statistics Overview")
#     st.write(data)

#     # Pilar 2: Visualisasi Klastering
#     st.header("Cluster Visualization")
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.scatterplot(data=data, x='PTS', y='AST', hue='Cluster', palette='viridis', legend='full', ax=ax)
#     ax.set_xlabel("Points")
#     ax.set_ylabel("Assists")
#     st.pyplot(fig)


#     # Pilar 3: Detail Klaster
#     st.header("Cluster Details")
#     for cluster in range(n_clusters):
#         st.subheader(f"Cluster {cluster}")
#         cluster_data = data[data['Cluster'] == cluster]
#         st.write(cluster_data)

#     # Pilar 4: Analisis Statistik Klaster
#     st.header("Cluster Statistics Analysis")
#     cluster_stats = data.groupby('Cluster').mean()
#     st.write(cluster_stats)

# # Panggil fungsi page1()
# page1()

st.title('NBA Player Clustering Analysis')
# Menampilkan foto di bagian bawah
st.image('basket.jpg', caption='NBA Player CLustering Analysis', use_column_width=True)


URL = 'Data_Cleaning.csv'
df = pd.read_csv(URL)

# file_path_clf = 'kmeans.pkl'
# with open(file_path_clf, 'rb') as f:
#     clf = joblib.load(f)

data_before_mapping = 'https://raw.githubusercontent.com/arstyav/Mini-Project-Data-Mining1/main/2021-2022%20NBA%20Player%20Stats%20-%20Playoffs%20(1).csv'
dfa = pd.read_csv(data_before_mapping)

selected_page = st.sidebar.selectbox(
    'Select Page',
    ['🏠 Introducing','📊 Data Distribution','🔗 Relationship Analysis','🧩 Composition & Comparison','🌐 Clustering']
)

if selected_page == '🏠 Introducing':
    data = pd.read_csv(URL)
    st.subheader("NBA Player Analysis")
    st.write("""
   Selamat datang di Analisis Statistik Pemain NBA! Di sini, kami hadirkan alat interaktif untuk membantu Anda memahami performa pemain NBA selama musim reguler 2021-2022. Dengan data statistik permainan yang terperinci, tujuan kami adalah memberikan wawasan yang penting bagi Anda untuk mengenal lebih dalam soal dataset ini dan bertujuan juga untuk pengambilan keputusan yang lebih efektif. Mari kita telusuri data dan temukan cerita menarik yang tersimpan di dalamnya!
    """)

    st.markdown("[Sumber : Kaggle.com](https://www.kaggle.com/datasets/vivovinco/nba-player-stats)")

    st.title('NBA Player Clustering Analysis')
    st.write('Berikut merupakan tampilan dataset setelah di cleaning')
    URL = 'Data_Cleaning.csv'
    df = pd.read_csv(URL)
    st.write(df)

elif selected_page == "📊 Data Distribution":
    st.subheader("Bagian Data Distribution")

    feature_options = ['Jumlah Pemain Berdasarkan Posisi', 'Distribusi Umur Pemain', 'Perbandingan Jumlah Pemain Per Tim']
    selected_feature = st.selectbox('Pilih Opsi dibawah ini', feature_options)

    if selected_feature == 'Jumlah Pemain Berdasarkan Posisi':
        # Bar Plot Jumlah Pemain berdasarkan Posisi
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Pos', data=dfa, palette='viridis')
        plt.title('Jumlah Pemain berdasarkan Posisi')
        plt.xlabel('Posisi')
        plt.ylabel('Jumlah Pemain')
        st.pyplot(plt)

        st.write("""Jumlah Pemain berdasarkan Posisi: Judul grafik menyatakan tujuan dari visualisasi ini, yaitu menampilkan distribusi jumlah pemain berdasarkan posisi.

Axe X (Posisi): Pada sumbu x, kita memiliki kategori posisi pemain (misalnya, C untuk Center, SG untuk Shooting Guard, SF untuk Small Forward, PG untuk Point Guard, PF untuk Power Forward). Setiap batang pada grafik mewakili satu posisi.

Axe Y (Jumlah Pemain): Pada sumbu y, kita memiliki jumlah pemain untuk setiap posisi. Tinggi batang menunjukkan seberapa banyak pemain yang memiliki posisi tertentu.

Palette 'viridis': Warna batang yang digunakan diambil dari palet warna 'viridis'. Palet warna ini sering digunakan untuk visualisasi data kuantitatif karena memberikan kontrast yang baik dan mudah dibaca.""")


    elif selected_feature == 'Distribusi Umur Pemain':
        # Visualisasi distribusi umur pemain
        plt.figure(figsize=(10, 6)) 
        sns.histplot(dfa['Age'], bins=20, kde=True)
        plt.title('Distribusi Umur Pemain NBA')
        plt.xlabel('Umur')
        plt.ylabel('Frekuensi')
        st.pyplot(plt)

        st.write("""Axe X (Umur): Pada sumbu x, kita memiliki variabel umur pemain. Interval umur dibagi menjadi bins, dan setiap batang mewakili sejumlah pemain dalam bin tersebut.

Axe Y (Frekuensi): Pada sumbu y, kita memiliki frekuensi atau jumlah pemain untuk setiap bin umur. Tinggi batang menunjukkan seberapa sering umur tertentu muncul dalam dataset.

KDE (Kernel Density Estimation): Grafik ini juga menunjukkan kurva KDE yang mendekati distribusi probabilitas dari data umur. Ini memberikan perkiraan visual tentang distribusi data secara keseluruhan.

Banyak Bins (20): Data umur dibagi menjadi 20 bins, yang memberikan gambaran rinci tentang sebaran umur pemain.

Dari grafik ini, kita dapat melihat sebaran umur pemain dalam dataset.
Puncak atau puncak tertinggi pada kurva menunjukkan di mana umur pemain paling sering muncul.
Dengan melihat keseluruhan distribusi, kita dapat mendapatkan gambaran tentang apakah dataset cenderung memiliki pemain yang lebih muda, lebih tua, atau memiliki distribusi umur yang merata.""")

    
    elif selected_feature == 'Perbandingan Jumlah Pemain Per Tim':
        sns.countplot(x='Tm', data=dfa)
        plt.title('Jumlah Pemain per Tim')
        plt.xticks(rotation=45)
        st.pyplot(plt)

        st.write("""
                - Grafik ini memberikan gambaran tentang distribusi jumlah pemain di setiap tim dalam dataset.
                - Tim dengan batang lebih tinggi memiliki lebih banyak pemain, sementara tim dengan batang lebih pendek memiliki jumlah pemain yang lebih sedikit.
                - Count plot berguna untuk melihat sebaran keseimbangan atau ketidakseimbangan dalam distribusi pemain di seluruh tim, dan ini dapat memberikan wawasan awal tentang komposisi pemain di berbagai tim.""")



elif selected_page == "🔗 Relationship Analysis":
    data = pd.read_csv(data_before_mapping)

    # Pilih hanya kolom numerik
    numeric_data = data.select_dtypes(include='number')

    # Hitung korelasi antar variabel numerik
    correlation_matrix = numeric_data.corr()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Custom color map with high contrast (contoh menggunakan palet "coolwarm")
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlation_matrix, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, annot=False, ax=ax)

    # Add a title
    ax.set_title('Matriks Korelasi', fontsize=16)

    # Customize the tick labels
    ax.tick_params(labelsize=12)

    # Menampilkan visualisasi dengan Streamlit
    st.pyplot(fig)

    st.write("""Warna dingin (biru) menunjukkan korelasi negatif, sementara warna hangat (merah) menunjukkan korelasi positif.

- Korelasi antar variabel dapat dilihat dari heatmap. Semakin dekat nilai warna dengan 1 atau -1, semakin tinggi korelasinya.
- Korelasi positif terlihat dengan warna yang lebih terang, sedangkan korelasi negatif terlihat dengan warna yang lebih gelap.
- Heatmap membantu mengidentifikasi pola dan hubungan antar variabel dalam dataset, yang dapat digunakan untuk analisis lebih lanjut atau pemilihan fitur.""")


# elif selected_page == "Composition & Comparison":
#     data = pd.read_csv(URL)

#     def compositionAndComparison(df):
#     # Hitung rata-rata fitur untuk setiap kelas
#         class_composition = df.groupby('kmeans_cluster').mean()
    
#     # Plot komposisi kelas
#         plt.figure(figsize=(10, 6))
#         sns.heatmap(class_composition.T, annot=True, cmap='YlGnBu')
#         plt.title('Composition for each cluster')
#         plt.xlabel('Cluster')
#         plt.ylabel('Feature')
#         st.pyplot(plt)

#         st.markdown('''
#         As you can see the bar plot above shows the composition of each cluster taken from the average of each feature (column) used in the clustering process.
#         ''')

#     # Panggil fungsi compositionAndComparison dengan DataFrame yang sesuai
#     # df harus berisi kolom-kolom yang diinginkan untuk analisis clustering
#     compositionAndComparison(df[['Pos', 'Age', 'G', 'GS', 'MP', 'FG', '3P', '2P', 'eFG%', 'FT', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Age_Group', 'kmeans_cluster']])

# elif selected_page == "Composition & Comparison":
#     data = pd.read_csv(URL)

#     def compositionAndComparison(df):
#     # Memilih fitur-fitur yang akan digunakan untuk clustering
#         selected_features = ['Pos', 'Age', 'G', 'GS', 'MP', 'FG', '3P', '2P', 'eFG%', 'FT', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Age_Group']

#     # Memisahkan data menggunakan fitur-fitur yang dipilih
#         X = df[selected_features]

#     # Normalisasi data
#         scaler = MinMaxScaler()
#         X_scaled = scaler.fit_transform(X)

#     # Melakukan clustering dengan KMeans
#         kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
#         kmeans.fit(X_scaled)

#     # Tambahkan kolom 'kmeans_cluster' ke DataFrame
#         df['kmeans_cluster'] = kmeans.labels_

#     # Hitung rata-rata fitur untuk setiap kelas
#         class_composition = df.groupby('kmeans_cluster').mean()

#     # Plot komposisi kelas
#         plt.figure(figsize=(10, 6))
#         sns.heatmap(class_composition.T, annot=True, cmap='YlGnBu')
#         plt.title('Composition for each cluster')
#         plt.xlabel('Cluster')
#         plt.ylabel('Feature')
#         st.pyplot(plt)

#         st.markdown('''
#         As you can see, the heatmap above shows the composition of each cluster taken from the average of each feature used in the clustering process.
#         ''')

#     # Panggil fungsi compositionAndComparison dengan DataFrame yang sesuai
#     compositionAndComparison(data)

elif selected_page == "🧩 Composition & Comparison":
    data = pd.read_csv(URL)

    def compositionAndComparison(df):
    # Memilih fitur-fitur yang akan digunakan untuk clustering
        selected_features = ['Pos', 'Age', 'G', 'GS', 'MP', 'FG', '3P', '2P', 'eFG%', 'FT', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Age_Group']

    # Memisahkan data menggunakan fitur-fitur yang dipilih
        X = df[selected_features]

    # Normalisasi data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

    # Melakukan clustering dengan KMeans
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)

    # Tambahkan kolom 'kmeans_cluster' ke DataFrame
        df['kmeans_cluster'] = kmeans.labels_

    # Hitung rata-rata fitur untuk setiap kelas
        class_composition = df.groupby('kmeans_cluster').mean()

    # Plot komposisi kelas
        plt.figure(figsize=(10, 6))
        sns.heatmap(class_composition.T, annot=True, cmap='YlGnBu')
        plt.title('Composition for each cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Feature')
        st.pyplot(plt)
        st.write('''
                 Visualisasi heatmap di atas menampilkan komposisi dari setiap klaster yang dihasilkan dari proses clustering. Heatmap ini didasarkan pada rata-rata setiap fitur yang digunakan dalam proses clustering. 
                 Dengan demikian, kita dapat melihat bagaimana karakteristik rata-rata dari setiap klaster dalam hal fitur-fitur yang dipilih. Misalnya, kita dapat melihat bagaimana rata-rata fitur seperti usia, jumlah permainan (G), jumlah menit bermain (MP), dan lainnya bervariasi di antara klaster-klaster yang terbentuk. 
                 Ini membantu kita memahami pola dan perbedaan antara klaster yang ada dalam dataset.
                 ''')

    # Plot perbandingan umur antara kluster menggunakan box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='kmeans_cluster', y='Age')
        plt.title('Comparison of Age between Clusters')
        plt.xlabel('Cluster')
        plt.ylabel('Age')
        st.pyplot(plt)

        st.write('''Box plot memungkinkan kita untuk membandingkan distribusi umur antara klaster yang dihasilkan oleh algoritma KMeans. 
                 Setiap box plot menampilkan distribusi umur di dalam klaster dalam bentuk box dan garis median. 
                 Dengan box plot ini, kita dapat melihat apakah ada perbedaan yang signifikan dalam distribusi umur di antara klaster dan memahami bagaimana umur terdistribusi di setiap klaster.
                 ''')

    # Panggil fungsi compositionAndComparison dengan DataFrame yang sesuai
    compositionAndComparison(data)



elif selected_page == "🌐 Clustering":    
    data = pd.read_csv(URL)

    # Memilih fitur-fitur yang akan digunakan untuk clustering
    selected_features = ['Age', 'G', 'MP', 'FG', '3P', '2P', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']

    # Memisahkan data menggunakan fitur-fitur yang dipilih
    X = data[selected_features]

    # Normalisasi data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Menambahkan sidebar
    st.sidebar.subheader("Cluster Input")

    # Input fitur dari pengguna
    age = st.sidebar.slider('Age', min_value=0, max_value=50, step=1)
    g = st.sidebar.slider('Games Played', min_value=0, max_value=82, step=1)
    mp = st.sidebar.slider('Minutes Played', min_value=0, max_value=48, step=1)
    fg = st.sidebar.slider('Field Goals', min_value=0, max_value=20, step=1)
    three_p = st.sidebar.slider('3-Point Field Goals', min_value=0, max_value=10, step=1)
    two_p = st.sidebar.slider('2-Point Field Goals', min_value=0, max_value=20, step=1)
    trb = st.sidebar.slider('Total Rebounds', min_value=0, max_value=30, step=1)
    ast = st.sidebar.slider('Assists', min_value=0, max_value=20, step=1)
    stl = st.sidebar.slider('Steals', min_value=0, max_value=10, step=1)
    blk = st.sidebar.slider('Blocks', min_value=0, max_value=10, step=1)
    tov = st.sidebar.slider('Turnovers', min_value=0, max_value=10, step=1)
    pts = st.sidebar.slider('Points', min_value=0, max_value=50, step=1)

    # Button untuk melakukan clustering
    button_cluster = st.sidebar.button("Cluster")

    # Jika tombol ditekan, lakukan clustering
    if button_cluster:
    # Menyimpan input pengguna ke dalam array
        user_input = [[age, g, mp, fg, three_p, two_p, trb, ast, stl, blk, tov, pts]]
    
    # Normalisasi data input pengguna
        user_input_scaled = scaler.transform(user_input)
    
    # Melakukan clustering dengan KMeans
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
    
    # Prediksi klaster untuk input pengguna
        predicted_cluster = kmeans.predict(user_input_scaled)
    
    # Menampilkan hasil klaster
        st.write("Hasil Klaster untuk Nilai Fitur yang Dimasukkan Pengguna:", predicted_cluster[0])

        st.write('''
                - Cluster 0 All-Around Performers: Kelompok ini mungkin terdiri dari pemain yang memiliki kontribusi merata di berbagai aspek permainan, termasuk poin, assist, rebound, dan lain-lain.
                - Cluster 1 Scoring Specialists: Cluster ini mungkin berisi pemain-pemain yang terutama unggul dalam mencetak poin, dengan statistik seperti field goals, 3-point field goals, dan total points menjadi yang terpenting.
                - Cluster 2 Defensive Powerhouses: Cluster ini mungkin terdiri dari pemain-pemain yang menjadi andalan dalam pertahanan, dengan statistik seperti steals, blocks, dan total rebounds menjadi sorotan utama.  
                 ''')
    
    st.write("""
    Aplikasi ini menggunakan algoritma KMeans untuk clustering data pemain NBA berdasarkan fitur-fitur tertentu. Anda dapat menyesuaikan nilai fitur melalui sidebar dan melihat hasil klaster dengan menekan tombol "Cluster".
    Fitur-fitur dipilih berdasarkan relevansinya dalam memahami performa pemain NBA. Ini mencakup usia, keterlibatan dalam permainan, waktu bermain, kemampuan menyerang, pertahanan, dan kontribusi poin, sesuai dengan tujuan analisis (business understanding) untuk membantu pengambilan keputusan strategis tim dan evaluasi pemain.
    """)
