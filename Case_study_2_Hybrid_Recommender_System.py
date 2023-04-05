import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#### İş problemi

'''

ID'si verilen kullanıcı için item-based ve user-based recommender yöntemlerini kullanarak 10 film önerisi yapınız.

'''


#### Veri seti & Değişkenler

'''

Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır. İçerisinde filmler ile birlikte bu filmlere yapılan
derecelendirme puanlarını barındırmaktadır. 27.278 filmde 2.000.0263 derecelendirme içermektedir. Bu veri seti ise 17 Ekim 2016
tarihinde oluşturulmuştur. 138.493 kullanıcı ve 09 Ocak 1995 ile 31 Mart 2015 tarihleri arasında verileri içermektedir. Kullanıcılar
rastgele seçilmiştir. Seçilen tüm kullanıcıların en az 20 filme oy verdiği bilgisi mevcuttur.

movie.csv

movieId:  Eşsiz film numarası. 
title:  Film adı
genres:  Tür


rating.csv

userid:  Eşsiz kullanıcı numarası. (UniqueID)
movieId:  Eşsiz film numarası. (UniqueID)
rating:  Kullanıcı tarafından filme verilen puan
timestamp:  Değerlendirme tarihi

'''


##############################################
## Görev 1: Veriyi Hazırlama
##############################################

### Adım 1: movie, rating veri setlerini okutunuz.
movie = pd.read_csv("Miuul_Course_1/Recommendation-Systems/Datasets/movie.csv")
rating = pd.read_csv("Miuul_Course_1/Recommendation-Systems/Datasets/rating.csv")


movie.head()
rating.head()

### Adım 2: rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.

df = rating.merge(movie[["title", "movieId"]], how="left", on="movieId")

### Adım 3: Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri setinden çıkartınız.

df_count = pd.DataFrame(df.groupby("movieId").agg({"title": "count"}))
df_count = df_count[df_count["title"] > 1000]
df_count.reset_index(inplace=True)

df = df.loc[df["movieId"].isin(df_count["movieId"])]

### Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturunuz.

df_pivot = df.pivot_table(index=["userId"], columns=["title"], values="rating")

### Adım 5: Yapılan tüm işlemleri fonksiyonlaştırınız.

def data_prep(rat, movie):
    # rating veri setine Id’lere ait film isimlerini ve türünü movie veri setinden ekleyiniz.
    rat = rat.merge(movie[["title", "movieId"]], how="left", on="movieId")

    # Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini listede tutunuz ve veri setinden çıkartınız.
    df_counts = pd.DataFrame(rat.groupby("movieId").agg({"title": "count"}))
    df_counts = df_counts[df_counts["title"] > 1000]
    df_counts.reset_index(inplace=True)

    rat = rat.loc[rat["movieId"].isin(df_counts["movieId"])]

    # index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin bulunduğu dataframe için pivot table oluşturunuz.
    df_pivot = df.pivot_table(index=["userId"], columns=["title"], values="rating")

    return df_pivot

df_pivot = data_prep(rating, movie)



##############################################
## Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
##############################################

### Adım 1: Rastgele bir kullanıcı id’si seçiniz.

random_user = 82739

### Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.

random_user_df = df_pivot[df_pivot.index == random_user]

### Adım 3: Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye atayınız

movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()



##############################################
## Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
##############################################

### Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları dataframenizden seçiniz ve movies_watched_df adında yeni bir dataframe oluşturunuz.

movies_watched_df = df_pivot[movies_watched]

### Adım 2: Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğini bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.

user_movie_count = movies_watched_df.T.notna().sum()

### Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı id’lerinden users_same_movies adında bir liste oluşturunuz.

users_same_movies = user_movie_count[user_movie_count > (len(movies_watched)*60/100)].index


##############################################
## Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
##############################################

### Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.

filtered_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]

### Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.

corr_df = filtered_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

### Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.

top_users = corr_df.loc[((corr_df["user_id_1"] == random_user) & (corr_df["corr"] > 0.65)), ["user_id_2", "corr"]].reset_index(drop=True)

### Adım 4: top_users dataframe’ine rating veri seti ile merge ediniz.

top_users_rating = top_users.merge(df, how="inner")
top_users_rating = top_users_rating[top_users_rating["userId"] != random_user]



##############################################
## Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
##############################################

### Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.

top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]
top_users_rating.head()

### Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir dataframe oluşturunuz

recommendation_df = top_users_rating.groupby("movieId").agg({"weighted_rating": "mean"})
recommendation_df.head()

### Adım 3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.

movies_to_be_recommend  = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values(by="weighted_rating", ascending=False)

### Adım 4: movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz.

reccomend = movies_to_be_recommend.merge(movie[["movieId", "title"]])
reccomend.head(5)

