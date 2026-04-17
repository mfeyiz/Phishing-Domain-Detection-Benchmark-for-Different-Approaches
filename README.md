# Phishing Alan Adı Tespit Projesi

Bu proje, şüpheli alan adlarının (domainlerin) bilinen markaları taklit edip etmediğini tespit etmek için Algoritmik, Makine Öğrenmesi ve Derin Öğrenme yöntemlerini kullanan kapsamlı bir araç setidir.

## Proje Genel Bakış

Proje, bir hedef alan adı ile şüpheli bir alan adını karşılaştırarak, aralarındaki benzerlikleri ve potansiyel phishing tekniklerini analiz eder.

### Temel Özellikler:
- **Algoritmik Analiz:** Levenshtein mesafesi, Jaro-Winkler, klavye yakınlığı ve homograf (homoglyph) tespiti gibi metriklerle hızlı analiz.
- **Makine Öğrenmesi (ML):** Entropi, özel karakter sayımı ve leksikal sözcüksel özellikler gibi çıkarılmış nitelikler üzerinden risk puanlaması.
- **Derin Öğrenme (DL):** Transformer tabanlı modeller (BERT) ve özel Encoder mimarileri ile anlamsal ve yapısal benzerlik tespiti.
- **Benchmark Aracı:** Farklı dedektörlerin performansını (doğruluk, hız) ölçmek için kapsamlı test senaryoları.

## Proje Yapısı

| Dosya | Açıklama |
|-------|----------|
| `algorithmic_detector.py` | Kural tabanlı ve metrik bazlı (Levenshtein, Jaro vb.) tespit motoru. |
| `ml_detector.py` | Karakter istatistikleri ve leksikal özellikler kullanarak risk puanı hesaplayan motor. |
| `dl_detector.py` | PyTorch ve Hugging Face (DomURLs_BERT) kullanan derin öğrenme modeli. |
| `utils.py` | Tüm modeller tarafından kullanılan yardımcı fonksiyonlar ve veri yapıları (homograf sözlükleri, klavye düzeni vb.). |
| `benchmark.py` | Modelleri çeşitli senaryolarda (Typosquatting, Combosquatting vb.) test eden araç. |

## Kurulum

1. Depoyu klonlayın:
   ```bash
   git clone <repo-url>
   cd Phising
   ```

2. Gerekli paketleri yükleyin:
   ```bash
   pip install torch transformers scikit-learn
   ```

## Kullanım

### Modellerin Manuel Test Edilmesi
Her bir modelin `predict` fonksiyonu şu şekilde kullanılabilir:

```python
from algorithmic_detector import predict

target = "google.com"
suspicious = "g00gle.com"

label, score = predict(target, suspicious)
print(f"Sonuç: {label}, Güven Skoru: {score}")
```

### Benchmark Çalıştırma
Tüm modellerin performansını çeşitli saldırı tiplerine karşı test etmek için:

```bash
python scripts/benchmark.py
```

Tek bir modeli benchmark etmek ve dashboard için JSON raporu üretmek için:

```bash
python scripts/benchmark.py --samples 500 --output-json benchmark_results.json --output-md benchmark_results.md
```

Kullanılabilir model anahtarları:

- `algorithmic`
- `random_forest`
- `xgboost`
- `url_bert`
- `bi_encoder_canine` (model mevcutsa)
- `crossencoder_canine` (model mevcutsa)

### Benchmark Dashboard

Proje kökünde yer alan `benchmark_dashboard.html`, benchmark sonuçlarını tek model odaklı bir arayüzde gösterir.

1. Benchmark çalıştırın (JSON ve MD çıktısı üretilecek).
2. Proje kökünde basit bir HTTP sunucu açın:
   ```bash
   python -m http.server 8000
   ```
3. Tarayıcıdan `http://localhost:8000/benchmark_dashboard.html` adresine gidin.

Dashboard önce `benchmark_results.json` dosyasını, bulunamazsa `benchmark_results.md` dosyasını okur.

## Tespit Edilen Saldırı Tipleri

- **Typosquatting:** Yazım hatalarından yararlanan alan adları (örn: `gogle.com`).
- **Combosquatting:** Marka isminin yanına anahtar kelimeler eklenmesi (örn: `paypal-login.com`).
- **Homoglyph Attacks:** Benzer görünen karakterlerin (Görsel benzerlik) kullanımı (örn: `googIe.com`).
- **Subdomain manipulation:** Alt alan adları ile yanıltma.

