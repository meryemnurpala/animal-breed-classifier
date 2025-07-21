# Evcil Hayvan Cinsi Tanıma Web Uygulaması

Bu proje, Oxford-IIIT Pet Dataset kullanılarak eğitilmiş bir derin öğrenme modeliyle kedi ve köpek cinslerini tahmin eden modern bir web uygulamasıdır. Kullanıcılar, web arayüzü üzerinden evcil hayvan fotoğrafı yükleyerek, modelin tahmin ettiği cins ve olasılıkları anında görebilir.

## Özellikler
- 37 farklı kedi ve köpek cinsini tanıyan model
- Flask tabanlı, modern ve responsive web arayüzü
- Gerçek zamanlı tahmin ve görsel önizleme
- Arka planda pets.jpg ile estetik tasarım

## Kurulum
1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
2. Oxford-IIIT Pet Dataset'i ve modeli (oxford_pets_model.h5) oluşturun veya eğitin. (Veri ve model dosyaları repoya dahil değildir.)
3. Sunucuyu başlatın:
   ```bash
   python app.py
   ```
4. Tarayıcıda `http://127.0.0.1:5000` adresine gidin.

## Kullanım
- Ana sayfada evcil hayvan fotoğrafı yükleyin.
- Model, görseldeki hayvanın cinsini ve en yüksek olasılıklı diğer cinsleri tahmin eder.
- Sonuçlar, modern bir kart yapısında ve progress bar ile görselleştirilir.

## Dizin Yapısı
```
Pet/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── static/
│   └── pets.jpg
├── templates/
│   └── index.html
└── data/         # (Git'e dahil edilmez)
```

## Notlar
- `data/` klasörü ve model dosyaları `.gitignore` ile hariç tutulmuştur.
- Modeli eğitmek için `train_model.py` dosyasını kullanabilirsiniz.
- Web arayüzü, sadece model dosyası ve gerekli kodlar ile çalışır.

## Örnek Çıktı

---
MIT Lisansı 
