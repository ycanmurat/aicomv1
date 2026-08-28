# Mimari ve araştırma kararları

## Donanıma uygun ana yol

Hedef makine Apple M4 ve 16 GB birleşik bellektir. Bu sınıfta kaliteyi yalnız parametre
sayısıyla büyütmek, konuşma gecikmesini ve bellek baskısını hızla bozar. Seçilen denge:

| Katman | Birincil | Güvenli geri dönüş | Seçim nedeni |
|---|---|---|---|
| STT | Whisper large-v3-turbo q8 | Nemotron 3.5 ASR 0.6B / Metal | Canlı Türkçe denemede Whisper daha doğru; Nemotron yaklaşık 0,8 sn daha hızlı |
| LLM | Qwen3.5 9B Q4 / Ollama | Qwen3.5 4B | 16 GB'da genel yetenek ile hız dengesi |
| TTS | FreyaTTS-small 183M | macOS Yelda | Türkçe-first açık model; sistem sesi her zaman çevrimdışı hazır |
| Hafıza | SQLite FTS5 + özet | yalnız yakın geçmiş | Küçük, denetlenebilir, ağsız |

Kaynak projeler: [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp),
[Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
[whisper.cpp](https://github.com/ggml-org/whisper.cpp),
[FreyaTTS](https://github.com/freyavoiceai/FreyaTTS),
[Qwen3.5/Ollama](https://ollama.com/library/qwen3.5).

## Neden önceki tipik prototipler kötü hissediyor?

Bir sesli asistanı kötü hissettiren şey çoğu zaman yalnız model kalitesi değildir:

1. Her turda bağlantı ve model yeniden kurulursa ilk yanıt gecikir.
2. Konuşma sonu kaba bir sessizlik süresiyle ölçülürse asistan ya söz keser ya bekletir.
3. LLM cevabının tamamı beklenip TTS'e tek parça verilirse etkileşim konuşma gibi olmaz.
4. Sabit cümle önbelleği genel sohbette işe yaramaz.
5. Ses çalarken mikrofon kapatılırsa kullanıcı asistanı kesemez.

AICOM kalıcı WebSocket, tarayıcıda 20 ms ses kareleri, ön tampon, cümle tabanlı TTS kuyruğu
ve tur iptali kullanır. İlk sürümde endpoint kararı enerji tabanlı VAD'dir. Nemotron hazır
olduğunda ASR akış yolu seçilir; ileride Silero VAD/Smart Turn eklemek adaptör sınırlarını
değiştirmez.

## Güven ve uzmanlık

Yerel LLM genel bilgiyi taşır; doğruluk gerektiren kişisel/kurumsal olgular SQLite bilgi
tabanından getirilir. Hesap ve saat gibi deterministik işler modelin tahminine bırakılmaz.
Güncel internet bilgisi varsayılan mimarinin dışında tutulmuştur. Bu nedenle asistan,
kanıtı olmayan güncel bir iddiayı uydurmak yerine sınırını belirtir.

Bir sonraki araç katmanı için güvenli sınır: yalnız açıkça izin verilmiş yerel işlevler,
şemalı girdiler, zaman aşımı ve kullanıcıya görünür sonuç. Serbest kabuk erişimi sesli
asistanın varsayılan yetkisi değildir.

## Kaynak ve veri sınırları

- Web istemcisi yalnız aynı origin REST/WebSocket uçlarına bağlanır.
- Model dosyaları ve Hugging Face önbelleği proje `models/` dizinindedir.
- Oturum sesleri rastgele kimlikli, izinleri daraltılmış dizinlerde tutulur.
- Ses sunucusu yol geçişini gerçek yol ve üst dizin denetimiyle reddeder.
- Git yalnız kodu taşır; model, ses, `.env` ve kişisel bilgi tabanı dışarıda kalır.
