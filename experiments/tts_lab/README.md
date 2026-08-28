# Türkçe TTS Deney Alanı

Bu klasör, ana uygulamaya model bağlamadan önce Türkçe sesleri karşılaştırmak içindir.
Model kodları, sanal ortamlar, ağırlıklar ve üretilen sesler `.runtime/voice-lab` altında tutulur ve Git'e eklenmez.

## MOSS-TTS-Nano kurulumu

```bash
./experiments/tts_lab/setup_moss.sh
```

## Yerel test ekranı

```bash
./experiments/tts_lab/run_moss.sh
```

Ardından `http://127.0.0.1:18083` adresini açın. Türkçe bir kadın sesi değerlendirmek için temiz, tek konuşmacılı bir referans WAV dosyası yükleyin.

## Streaming ölçümü

```bash
.runtime/voice-lab/.venv/bin/python experiments/tts_lab/benchmark_moss.py \
  --reference-audio /tam/yol/referans.wav
```

Ölçüm, dosyanın tamamlanma süresi yerine ilk gerçek PCM parçasının geliş süresini, parça sayısını, üretilen ses süresini ve RTF değerini raporlar.

## Temizleme

Deney alanının bütün indirilenlerini kaldırmak için, çalışan test sunucusunu durdurduktan sonra yalnızca şu klasörü silmek yeterlidir:

```text
.runtime/voice-lab
```

Bu işlem ana projenin veya başka projelerin modellerine dokunmaz.
