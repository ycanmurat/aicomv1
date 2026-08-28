# AICOM v1

AICOM, çağrı senaryolarına bağlı olmayan; Türkçe konuşmayı anlayan, yerel bir dil
modeliyle cevap üreten ve cevabı daha tamamlanmadan cümle cümle seslendiren bir masaüstü
sesli iletişim uygulamasıdır. Çalışma anında bulut API'sine ihtiyaç duymaz. Model kurulumu
bir kez internet gerektirir.

Bu proje özellikle Apple Silicon ve 16 GB birleşik bellek için dengelenmiştir. Hedef,
“dev bir modeli tek başına çalıştırmak” değil; iyi konuşma modeli, doğru ses hattı, yerel
araçlar ve düşük gecikmeyi birlikte kullanarak güçlü bir bütün oluşturmaktır.

## Şu anda çalışan özellikler

- Tarayıcı mikrofonundan kesintisiz 16 kHz PCM ses hattı
- Gürültü tabanını öğrenen yerel VAD ve konuşma sonunu otomatik algılama
- Konuşurken asistanı kesme (barge-in), devam eden model yanıtını ve ses kuyruğunu iptal etme
- Doğruluk odaklı `whisper.cpp`; tek ayarla düşük gecikmeli `Nemotron 3.5 ASR 0.6B`
- Ollama üzerinde `Qwen3.5 9B` akış yanıtı
- Tam cevabı beklemeden cümle bazlı TTS
- FreyaTTS-small hazırsa Türkçe Freya; değilse internetsiz macOS Yelda sesi
- Oturum hafızası, uzun konuşmaları özetleme ve yerel SQLite FTS5 bilgi tabanı
- Güvenli yerel saat ve hesap makinesi araçları
- Sağlık ekranı, gecikme ölçümleri, yazılı giriş ve sesli çıkış kuyruğu

## Hızlı başlangıç

Gerekli sistem araçları: macOS Apple Silicon, `uv`, `ffmpeg`, `ollama` ve `whisper-cli`.
Ollama uygulaması/servisi açık olmalı.

```bash
cd /Users/murat/Desktop/Projects/aicomv1
./scripts/bootstrap.sh
./scripts/run.sh
```

Sonra [http://127.0.0.1:7870](http://127.0.0.1:7870) adresini açın ve mikrofon düğmesine
bir kez dokunun. Temel kurulum Qwen ve Whisper yolunu hazırlar.

En iyi açık kaynak ses hattını da kurmak için:

```bash
./scripts/bootstrap.sh full
```

`full`, NVIDIA NeMo-Speech.cpp Metal çalışma zamanını ve Nemotron modelini proje içine;
FreyaTTS kodunu ve model önbelleğini de yerel kurulum alanına indirir. Bu seçenek daha uzun
sürer ve birkaç GB ek alan kullanır. Her iki mod da tekrar çalıştırılabilir.

Kurulumu denetlemek için:

```bash
uv run aicom-doctor
uv run aicom-benchmark --voice
uv run aicom-smoke /tam/yol/16khz-mono.wav
```

## Mimari

```text
Mikrofon → AudioWorklet 16 kHz → tarayıcı VAD/endpoint
         → kalıcı WebSocket → Whisper veya Nemotron
         → yerel araçlar + oturum hafızası → Qwen3.5/Ollama akışı
         → cümle ayırıcı → Freya veya macOS TTS → kesilebilir ses kuyruğu
```

Buradaki önemli ayrım, STT → LLM → TTS zincirinin üç uzun ve ardışık iş olarak
çalışmamasıdır. Model token üretirken metin ekrana akar; ilk anlamlı cümle biter bitmez TTS
başlar. Sonraki cümle üretilirken önceki cümle çalabilir.

Ayrıntılı teknik kararlar [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) dosyasındadır.

## Yerel bilgi ekleme

Uygulama yalnız eklediğiniz yerel belgelerde arama yapar. Örnek:

```bash
curl -X POST http://127.0.0.1:7870/api/knowledge \
  -H 'content-type: application/json' \
  -d '{"title":"Proje notu","body":"AICOM tamamen yerel çalışır.","source":"kişisel"}'
```

## Gizlilik ve gerçekçi sınır

Ses dosyaları `data/audio`, bilgi tabanı `data/knowledge`, STT/TTS modelleri `models`
altında kalır; Qwen ağırlığı Ollama'nın cihazdaki yerel model deposundadır. Bu veriler Git'e
alınmaz. Sunucu varsayılan olarak yalnız `127.0.0.1` üzerinde dinler. Kodda bulut model
çağrısı yoktur; Ollama da loopback adresindedir.

Qwen3.5 9B iyi bir yerel genel asistandır fakat GPT-5.6 Sol kapasitesini yüzde yüz yerelde,
16 GB bir cihazda eşitlemek fiziksel olarak mümkün değildir. AICOM belirsiz veya güncel
bilgi isteyen sorularda bunu söyleyecek şekilde yönlendirilmiştir. “Her konuda uzman ve hiç
hata yapmaz” iddiası yerine ölçülebilen hız, sağlam ses akışı, yerel bilgi ve dürüst sınırlar
hedeflenir.

## Geliştirme

```bash
uv sync --extra dev --python 3.11
uv run ruff check .
uv run pytest
```

Freya geliştirme profilini korumak için `uv sync --extra dev --extra freya --python 3.11`
kullanın. Temel profil bilinçli olarak yalnız macOS çevrimdışı ses geri dönüşünü kurar.

Lisans: MIT. Model ve çalışma zamanı bileşenlerinin kendi lisansları geçerlidir.
