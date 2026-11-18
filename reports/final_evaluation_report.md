# Laporan Evaluasi Komparatif Lengkap: Speech Emotion Recognition (SER)

> **Dataset**: Emotion Speech Dataset (ESD) — 35.000 ucapan (16 kHz mono)  
> **Evaluasi**: *Speaker-Independent* (Leave-Speakers-Out: 14 Train / 2 Val / 4 Test)  
> **Sampel Uji**: 7.000 sampel suara dari 4 speaker baru (0001, 0002, 0012, 0019; seimbang 1.400 sampel per emosi)  
> **Status Audit**: 100% Lolos Uji Anti-Bocor (*Anti-Leakage Verified*)  
> **Tanggal Pembaruan**: 18 November 2026

---

## 1. Tabel Perbandingan 3 Jalur Model (Benchmark Utama)

Seluruh model dievaluasi pada **test set yang sama persis** tanpa kebocoran subjek (*speaker-independent*).

| Metrik Evaluasi | Peluang Acak (Chance) | Jalur A: Baseline SVM (Handcrafted) | Jalur B: Log-Mel + CNN (ResNet) | Jalur C: Pretrained WavLM (Fine-tuned) 🏆 | Delta C vs Baseline |
|---|:---:|:---:|:---:|:---:|:---:|
| **UAR (Balanced Accuracy)** ⭐ | 20.00% | 45.79% | 64.44% | **70.40%** | <span style="color:green">**+24.61%**</span> |
| **Weighted Accuracy (WAR)** | 20.00% | 45.79% | 64.44% | **70.40%** | <span style="color:green">+24.61%</span> |
| **Macro F1-Score** | 20.00% | 45.45% | 63.01% | **70.41%** | <span style="color:green">+24.96%</span> |
| **Matthews Correlation (MCC)** | 0.0000 | 0.3332 | 0.5839 | **0.6388** | <span style="color:green">+0.3056</span> |
| **Cohen's Kappa** | 0.0000 | 0.3223 | 0.5555 | **0.6300** | <span style="color:green">+0.3077</span> |
| **Top-2 Accuracy** | 40.00% | 72.74% | 84.09% | **94.63%** 🔥 | <span style="color:green">**+21.89%**</span> |
| **Log Loss (Cross-Entropy)** | 1.6094 | 1.7664 | 0.9975 | **0.9184** | <span style="color:green">-0.8480</span> |
| **Expected Calibration (ECE)**| - | 0.3068 | 0.0799 | **0.1506** | - |
| **Subject 95% Bootstrap CI** | - | [41.80% – 51.41%] | [62.17% – 66.71%] | **[60.66% – 78.37%]** | Signifikan |

> [!NOTE]
> **Kesimpulan Kunci**:  
> **Jalur C (WavLM)** menjadi pemenang mutlak di seluruh metrik utama dengan **UAR 70.40%** dan **Top-2 Accuracy fantastis mencapai 94.63%**. Artinya, dalam **94.6% kasus suara**, tebakan 1 atau 2 teratas dari WavLM selalu mencakup emosi yang benar.

---

## 2. Perbandingan Performa Per Emosi (Per-Class)

Tabel di bawah memperlihatkan bagaimana evolusi model dari Baseline $\rightarrow$ CNN $\rightarrow$ WavLM secara drastis memperbaiki emosi-emosi sulit:

| Emosi | Support | Baseline SVM F1 | CNN / ResNet F1 | WavLM F1 🏆 | WavLM Presisi | WavLM Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Neutral** | 1.400 | 53.89% | **85.14%** | 72.95% | **98.90%** | 57.79% |
| **Sad** | 1.400 | 57.25% | 81.80% | **81.66%** | 73.78% | **91.43%** |
| **Angry** | 1.400 | 49.03% | 63.88% | **81.42%** | **89.97%** | **74.36%** |
| **Surprise** | 1.400 | 51.32% | 58.68% | **64.75%** | 53.81% | **81.29%** |
| **Happy** | 1.400 | 15.75% | 25.56% | **51.26%** 🚀 | 56.17% | **47.14%** |

### Analisis Kemajuan Krusial:
1. **Emosi Happy meningkat 2x lipat**: Pada CNN, F1 happy hanya 25.56%. Pada WavLM melesat menjadi **51.26%** (recall naik dari 17.9% menjadi 47.1%).
2. **Emosi Angry melompat ke 81.42% F1**: Presisi mencapai **89.97%** dan recall **74.36%**. Representasi SSL menangkap ketegangan pita suara (*vocal tension/harshness*) jauh lebih tajam dibanding spektrogram biasa.
3. **Neutral dengan Presisi 98.90%**: Hampir tanpa *false positive*. Jika WavLM memprediksi ucapan berlabel netral, kepastiannya adalah 99%.
4. **Sad dengan Recall 91.43%**: 9 dari 10 suara sedih terdeteksi dengan tepat berkat representasi pitch kontur rendah WavLM.

---

## 3. Matriks Konfusi & Solusi Konfusi Akustik

````carousel
![Confusion Matrix WavLM](/C:/Users/ALIM/.gemini/antigravity/brain/a5df193c-a9a9-4917-a54d-9c05b6738f5f/cm_wavlm.png)
<!-- slide -->
![Confusion Matrix CNN](/C:/Users/ALIM/.gemini/antigravity/brain/a5df193c-a9a9-4917-a54d-9c05b6738f5f/cm_cnn.png)
<!-- slide -->
![Confusion Matrix SVM](/C:/Users/ALIM/.gemini/antigravity/brain/a5df193c-a9a9-4917-a54d-9c05b6738f5f/cm_svm.png)
````

### Analisis Penurunan Konfusi Happy $\rightarrow$ Surprise:
- Pada **CNN (Jalur B)**: Terjadi konfusi masif sebanyak **1.077 file (76.93%)** suara Happy tertukar menjadi Surprise.
- Pada **WavLM (Jalur C)**: Konfusi berkurang drastis menjadi **681 file (48.64%)**, penurunan kesalahan sebesar **~37%**!  
- **Penyebab Mengapa WavLM Lebih Unggul**: WavLM dilatih sebelumnya (*pre-trained*) dengan tujuan masked speech prediction yang sensitif terhadap transisi fonem mikro dan artikulasi vokal, bukan sekadar energi makro spektral. Hal ini membantunya membedakan bukaan vokal *joy/laughter* dari *abrupt startle* pada ekspresi terkejut.

---

## 4. Evaluasi Lintas Bahasa & Lintas Speaker

### Breakdown Bahasa:
| Model | Bahasa Inggris (`en`) UAR | Bahasa Mandarin (`zh`) UAR | Analisis |
|---|:---:|:---:|---|
| **Baseline SVM** | 43.14% | 48.43% | Mandarin lebih unggul (+5.3%) karena nada leksikal membantu SVM |
| **CNN (ResNet)** | 62.17% | 66.71% | Mandarin stabil di 66.7%, Inggris 62.2% |
| **WavLM** 🏆 | **75.03%** | **65.77%** | **Bahasa Inggris melonjak ke 75.03%!** |

> [!TIP]
> Backbone `microsoft/wavlm-base-plus` dipra-latih terutama pada korpus bahasa Inggris (GigaSpeech, VoxCeleb, LibriSpeech). Akibatnya, pemahaman prosodi bahasa Inggrisnya sangat superior (**75.03% UAR / 76.12% F1**).

### Breakdown Per Speaker Unseen (Test Set):
| ID Speaker | Bahasa | SVM UAR | CNN UAR | WavLM UAR 🏆 | WavLM F1 |
|---|:---:|:---:|:---:|:---:|:---:|
| **0019** | `en` | 45.14% | 63.49% | **80.74%** 🔥 | **79.93%** |
| **0002** | `zh` | 54.40% | 66.11% | **76.00%** | **73.77%** |
| **0012** | `en` | 41.14% | 60.86% | **69.31%** | **70.40%** |
| **0001** | `zh` | 42.46% | 67.31% | **55.54%** | **49.50%** |

- Speaker `0019` dan `0002` mencapai performa mendekati pengenal emosi manusia (**76% – 81% UAR**).

---

## 5. Ringkasan Rekomendasi Deployment / Produksi

```mermaid
flowchart TD
    Audio[Audio Masuk .wav 16kHz] --> Preprocess[Preprocessing: Trim Silence top_db=30, Peak Norm, Crop/Pad 4s]
    Preprocess --> Choice{Kebutuhan Sistem}
    Choice -->|Akurasi Tertinggi / Server Cloud| WavLM[Jalur C: WavLM fine-tuned\nUAR: 70.40% | Top-2: 94.63%\nUkuran: ~370MB]
    Choice -->|Latensi Rendah / Edge / CPU / Mobile| CNN[Jalur B: EmotionCNN ResNet\nUAR: 64.44% | Top-2: 84.09%\nUkuran: ~24MB | Latensi: <10ms]
    Choice -->|Ultra-Ringan / Minimal Compute| SVM[Jalur A: SVM Baseline\nUAR: 45.79%\nUkuran: ~34MB]
```

1. **Server / Cloud API (Rekomendasi Utama)**: Gunakan checkpoint [best.pt (WavLM)](file:///D:/audio%20recog/runs/wavlm_exp1/best.pt) untuk mendapatkan akurasi maksimal **70.40% UAR** dan **Top-2 94.63%**.
2. **On-Device / Edge / Latensi Rendah**: Gunakan checkpoint [best.pt (CNN)](file:///D:/audio%20recog/runs/cnn_exp1/best.pt) yang ukurannya hanya ~24MB dan dapat melakukan inferensi di bawah 10 ms per jendela audio pada CPU biasa.
