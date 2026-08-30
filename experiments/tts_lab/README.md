# Türkçe MOSS-TTS Ses Laboratuvarı

Bu alan, **MOSS-TTS v1.5 Qwen3-8B** modelinin Türkçe doğallığını ana uygulamaya
bağlamadan önce karşılaştırmalı olarak dinlemek içindir. Local 4B veya Nano'ya
otomatik geçiş yapılmaz.

## Kurulum

```bash
./experiments/tts_lab/setup_moss_v15_flagship.sh
```

Kurulum; CrispASR `v0.8.30`, 8B Q4_K omurga, v1.5 codec ve yalnızca laboratuvara
ait Python ortamını `.runtime/voice-lab/moss-v15-flagship` altında tutar. Dosyalar
SHA-256 ile doğrulanır.

## Laboratuvarı çalıştırma

```bash
./experiments/tts_lab/run_moss_v15_lab.sh
```

Ardından `http://127.0.0.1:18083` adresini açın. İlk açılışta model bir kez
yüklenir ve gerçek Türkçe üretimle ısıtılır; sonraki denemelerde bellekte kalır.
Ekranda şunlar bulunur:

- zorunlu `Turkish` dil istemi;
- preset ve kullanım hakkı doğrulanmış WAV klonları;
- aynı Türkçe test cümleleriyle A/B dinleme geçmişi;
- istekten WAV'a gecikme, ses süresi, RTF ve gerçek backend RSS;
- resmî sampling varsayılanları ve isteğe bağlı temperature override.

Ctrl-C yalnızca laboratuvarın sahip olduğu CrispASR sürecini kapatır. Başka
projelerdeki model veya servisler değiştirilmez.

## Doğrulanan mevcut ölçümler

M4 / 16 GB üzerinde bu 8B Q4_K + codec süreci boşta yaklaşık **8.6 GB RSS**
kullanmıştır. Aynı 51 karakterlik karşılama cümlesinde sıcak preset denemeleri
15.64–45.82 saniye, doğru uzunluktaki sıcak klon denemesi 20.78 saniye sürmüştür.
Sonuçlar arayüz geçmişinde gerçek ses süresi ve RTF ile birlikte saklanır.

CrispASR v0.8.30, WAV klon referansını her istekte yeniden kodlar. Model karelerini
gerçek zamanlı aktarmak yerine tamamlanmış WAV döndürür ve gecikme denemeler arasında
belirgin değişebilir. Bu nedenle bu kurulum şu an gerçek zamanlı ürün motoru değil;
Türkçe doğallığı ve klon uygunluğunu dürüstçe değerlendirmek için bir **kalite
referansıdır**. Kontrolden çıkan üretimler yaklaşık 12.8 saniyelik model-frame
üst sınırıyla durdurulur.

## Tek dosya üretimi

Arayüz olmadan hızlı kontrol için:

```bash
./experiments/tts_lab/run_moss_v15_flagship.sh \
  "Merhaba, ben Fatma. Size nasıl yardımcı olabilirim?"
```

Eski Nano deneyi korunmuştur ancak flagship kalite değerlendirmesinde kullanılmaz.
