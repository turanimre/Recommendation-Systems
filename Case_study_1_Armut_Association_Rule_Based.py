import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#### İş problemi

'''

Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
ulaşılmasını sağlamaktadır.

Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak 
Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.

'''


#### Veri seti & Değişkenler

'''

Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır. Alınan her hizmetin tarih ve saat
bilgisini içermektedir.

UserId:  Müşteri numarası
ServiceId:  Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
            Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
            (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
CategoryId:  Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
CreateDate:  Hizmetin satın alındığı tarih

'''



##############################################
## Görev 1: Veriyi Hazırlama
##############################################


### Adım 1: armut_data.csv dosyasını okutunuz.

df = pd.read_csv("Miuul_Course_1/Recommendation-Systems/Datasets/armut_data.csv")
df.head()
df.info()




### Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID’yi "_" ile birleştirerek bu hizmetleri temsil edecek yeni bir değişken oluşturunuz

df["hizmet"] = df["ServiceId"].astype("string") + "_" + df["CategoryId"].astype("string")



### Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır. Association Rule
# Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir. Burada sepet tanımı her bir müşterinin aylık aldığı
# hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti; 2017’in 10.ayında aldığı 9_4, 38_4 hizmetleri
# başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir. Bunun için öncelikle sadece yıl ve ay içeren yeni bir
# date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_" ile birleştirirek ID adında yeni bir değişkene atayınız.


df["new_date"] = [row[:7] for row in df["CreateDate"]]


df["sepetId"] = df["UserId"].astype("string") + "_" + df["new_date"]



df.nunique()

df["hizmet"].nunique()

##############################################
## Görev 2: Birliktelik Kuralları Üretiniz ve Öneride bulununuz
##############################################


### Adım 1: Aşağıdaki gibi sepet, hizmet pivot table’i oluşturunuz.


pvt_df = df.groupby(["sepetId", "hizmet"]).agg({"hizmet": "count"}).\
    unstack().\
    fillna(0).\
    applymap(lambda x: 1 if x > 0 else 0)

pvt_df.columns = pvt_df.columns.droplevel(0)



### Adım 2: Birliktelik kurallarını oluşturunuz.


frequent_itemsets = apriori(pvt_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values(by="support", ascending=False)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)


### Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

arl_recommender(rules, "9_4", 4)