# Blueprint: Pipeline Training Speech Emotion Recognition (SER)

> Dataset utama: **Emotion Speech Dataset (ESD)**, lokasi lokal `D:\datasets\archive\Emotion Speech Dataset\`
> Tujuan: klasifikasi 5 emosi dari suara (`neutral, happy, angry, sad, surprise`) dengan evaluasi yang jujur (speaker-independent).
>
> **Versi 2** menambahkan: generator CSV split (`make_split.py`), prosedur anti-bocor antar subjek, dan modul metrik evaluasi lengkap (`metrics.py`).

---

## 1. Ringkasan

| Item | Keputusan |
|---|---|
| Tugas | Klasifikasi emosi kategorikal dari audio (1 ucapan → 1 label) |
| Input | `.wav` 16 kHz mono, crop/pad 4 detik |
| Kelas | neutral, happy, angry, sad, surprise (5 kelas, seimbang) |
| Jalur model | A: fitur manual + SVM (baseline) · B: log-mel + CNN · C: fine-tune WavLM/wav2vec2 |
| Split | **Per subjek (speaker)**, dibuat otomatis jadi `train.csv / val.csv / test.csv`. Tiga mode: `fixed` (mis. train 0001–0015), `stratified`, `kfold` |
| Anti-bocor | Pengecekan otomatis (speaker, path, md5, kelas, bahasa) + aturan prosedural (bagian 5.3) |
| Metrik utama | **UAR** + macro-F1, ditambah MCC, kappa, ECE, per kelas/speaker/bahasa, dan CI bootstrap per subjek |
| Uji generalisasi | Cross-corpus: latih di ESD, uji di RAVDESS / CREMA-D (label irisan) |
| Keluaran | CSV split, checkpoint terbaik, `label_map.json`, laporan metrik, skrip inferensi |

Emosi dikenali dari **akustik** (pitch, energi, tempo, kualitas suara), bukan dari teks. ESD memakai kalimat yang sama untuk semua emosi, jadi isi kata tidak membantu model.

---

## 2. Dataset

### 2.1 Struktur folder (dari screenshot)

```
Emotion Speech Dataset/
├── 0001/
│   ├── Angry/
│   ├── Happy/
│   ├── Neutral/
│   ├── Sad/
│   ├── Surprise/
│   └── 0001.txt          # transkrip + label per ucapan
├── 0002/ ...
└── 0020/
```

- Nama folder level-1 = **ID speaker**, level-2 = **label emosi**.
- Di dalam folder emosi, versi ESD asli biasanya masih punya subfolder `train/`, `evaluation/`, `test/`. Kode membaca `.wav` secara rekursif, jadi struktur dengan atau tanpa subfolder tetap terbaca.
- `0001.txt` berisi transkrip. Dipakai untuk verifikasi label (opsional), bukan sebagai fitur.

### 2.2 Karakteristik (verifikasi di salinanmu)

- 20 speaker, 5 emosi, 350 ucapan per emosi per speaker, total sekitar 35.000 ucapan (~29 jam, rata-rata ~3 detik per ucapan).
- Asumsi yang dipakai kode: **0001–0010 = penutur Mandarin, 0011–0020 = penutur Inggris**. Cek dulu di datamu. Bila berbeda, ubah dengan `--zh-max-id`.
- Data **akting** (bukan emosi spontan), direkam di studio. Hasil bagus di ESD belum tentu bagus di dunia nyata.
- Lisensi: untuk riset/non-komersial. Cek ketentuan sebelum dipakai di luar kuliah/penelitian.

---

## 3. Keputusan desain

| Aspek | Pilihan | Alasan |
|---|---|---|
| Sample rate | 16 kHz, mono | Standar SSL (WavLM/wav2vec2) dan cukup untuk prosodi |
| Durasi | Crop/pad **4 s** (train: random crop, eval: center crop) | Rata-rata ucapan ~3 s, 4 s menutup hampir semuanya |
| Trim silence | `top_db=30` di awal/akhir | Panjang hening bisa jadi pintasan bocoran label |
| Normalisasi | RMS/peak per ucapan | Menghilangkan perbedaan gain rekaman |
| Split | Per speaker, lewat CSV (5.2) | Mencegah model menghafal suara |
| Loss | Cross-entropy, label smoothing 0.1 | Stabil, kelas sudah seimbang |
| Pemilihan model | Early stopping pada **val UAR** | Metrik sesuai tugas |
| Seed | Tetap (42) dan dicatat | Reproduksibel |

---

## 4. Arsitektur pipeline

```
 wav mentah (ESD)
      │
      ▼
 [1] Audit + manifest.csv        ← make_split.py (scan, validasi, md5 opsional)
      │
      ▼
 [2] Split per subjek + CSV      ← train.csv / val.csv / test.csv / split_report.json
      │
      ▼
 [2b] Cek anti-bocor             ← otomatis saat membuat split DAN di awal training
      │
      ▼
 [3] Preprocessing               ← resample 16k, mono, trim, normalisasi, crop 4s
      │
      ├──► [4A] Fitur manual (MFCC+F0+energi / eGeMAPS) ─► SVM / RF
      ├──► [4B] Log-mel + SpecAugment ─────────────────► CNN
      └──► [4C] Waveform ─► WavLM/wav2vec2 + head ────► fine-tune
                                   │
                                   ▼
 [5] Training loop (AdamW, scheduler, AMP, early stop, checkpoint pada val UAR)
                                   │
                                   ▼
 [6] predictions.csv ─► metrics.py ─► laporan (UAR, F1, MCC, CM, per speaker/bahasa, CI, ECE)
                                   │
                                   ▼
 [7] Export + inferensi (best.pt, label_map.json, predict.py)
```

---

## 5. Tahapan detail

### 5.0 Environment

- Python 3.10+, GPU disarankan untuk jalur C (WavLM base muat di ±8 GB VRAM dengan batch kecil + AMP).
- Paket: `torch`, `torchaudio`, `transformers`, `librosa`, `soundfile`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pyyaml`, `tensorboard`, `pytest`. Opsional: `opensmile` (eGeMAPS), `audiomentations`, `onnx`.
- `make_split.py` tetap jalan tanpa `soundfile` (memakai modul `wave` bawaan Python untuk file WAV PCM seperti ESD).

### 5.1 Audit dan manifest

`make_split.py` membaca seluruh `.wav` dan membuat `manifest.csv` dengan kolom:

| Kolom | Isi |
|---|---|
| `path` | Path file |
| `speaker` | ID speaker (string, mis. `0001`) |
| `lang` | `zh` atau `en` (turunan dari ID speaker) |
| `emotion`, `label_id` | Nama dan indeks kelas (urutan: neutral, happy, angry, sad, surprise) |
| `utt_id` | Nomor ucapan dari nama file |
| `orig_split` | Subfolder bawaan ESD (`train/evaluation/test`) bila ada |
| `sr`, `dur` | Sample rate dan durasi (detik) |
| `md5` | Hash isi file (hanya dengan `--hash`) |

File rusak dilewati dengan peringatan. Cek tambahan yang tetap dilakukan manual: durasi ekstrem (< 0,5 s atau > 10 s), sample rate selain 16 kHz, dan beberapa label di manifest cocok dengan `0001.txt`.

### 5.2 Split per subjek dan CSV train/val/test

Kode: `src/data/make_split.py`. Satu perintah menghasilkan semua CSV.

```bash
# (a) Split tetap berdasar ID: train 0001-0015, val 0016-0017, test 0018-0020
python src/data/make_split.py --root "D:/datasets/archive/Emotion Speech Dataset" --out data \
       --mode fixed --train 1 15 --val 16 17 --test 18 20

# (b) Acak seimbang per bahasa (disarankan): 14 train / 2 val / 4 test
python src/data/make_split.py --root "D:/datasets/archive/Emotion Speech Dataset" --out data \
       --mode stratified --seed 42

# (c) Group K-Fold per subjek (angka akhir yang stabil)
python src/data/make_split.py --root "D:/datasets/archive/Emotion Speech Dataset" --out data \
       --mode kfold --k 5 --seed 42

# Opsi tambahan: --hash (deteksi duplikat konten), --strict (peringatan dianggap error)
# Verifikasi CSV yang sudah ada:
python src/data/make_split.py --verify data
```

| Mode | Cara kerja | Kapan dipakai |
|---|---|---|
| `fixed` | Range ID speaker kamu tentukan sendiri (default 1–15 / 16–17 / 18–20). Range yang tumpang tindih ditolak. | Ingin split yang sama persis dengan rancanganmu |
| `stratified` | Pilih speaker val/test acak dengan seed, **seimbang per bahasa** (2 test + 1 val per bahasa) | Hasil utama yang disarankan |
| `kfold` | Speaker dibagi ke k fold. Fold i = test, fold i+1 = val, sisanya train. Tiap speaker jadi test tepat satu kali | Angka akhir (mean ± std) |

> **Catatan penting untuk mode `fixed`.** Jika 0001–0010 adalah penutur Mandarin dan 0011–0020 penutur Inggris, maka val (0016–0017) dan test (0018–0020) **hanya berisi penutur Inggris**, sedangkan semua penutur Mandarin ada di train. Model tidak pernah diuji pada speaker Mandarin yang belum dilihat. Skrip mengeluarkan peringatan untuk ini. Pilihan: pakai `stratified` untuk hasil utama, atau tetap pakai `fixed` dan tulis di laporan bahwa generalisasi hanya diukur pada penutur Inggris.

**Keluaran (folder `data/`):**

| File | Isi |
|---|---|
| `manifest.csv` | Semua file tanpa kolom split |
| `split.csv` | Semua file + kolom `split` (`train/val/test`) |
| `train.csv`, `val.csv`, `test.csv` | Satu CSV per split, siap dibaca oleh Dataset |
| `split_report.json` | Jumlah file/speaker/jam, distribusi emosi dan bahasa per split, hasil pengecekan |
| `kfold/fold{i}/{train,val,test}.csv`, `kfold/folds.csv` | Hanya mode `kfold` |

Contoh ringkasan yang dicetak (pada data tiruan, bukan ESD asli):

```
train:    225 file | 15 speaker | bahasa {'en': 75, 'zh': 150} | speaker ['0001', ..., '0015']
val  :     30 file |  2 speaker | bahasa {'en': 30} | speaker ['0016', '0017']
test :     45 file |  3 speaker | bahasa {'en': 45} | speaker ['0018', '0019', '0020']
[PERINGATAN] Split 'test' hanya berisi bahasa ['en']; bahasa ['zh'] tidak terukur generalisasinya.
Pengecekan anti-bocor: LULUS (tidak ada error)
```

Memakai CSV di training (baca `speaker` sebagai string supaya `0001` tidak menjadi `1`):

```python
import sys, pandas as pd
sys.path.append("src/data")
from make_split import verify_split_files

verify_split_files("data/train.csv", "data/val.csv", "data/test.csv")   # berhenti bila ada kebocoran
train_df = pd.read_csv("data/train.csv", dtype={"speaker": str})
val_df   = pd.read_csv("data/val.csv",   dtype={"speaker": str})
test_df  = pd.read_csv("data/test.csv",  dtype={"speaker": str})
```

Split bawaan ESD (train/evaluation/test yang berbagi speaker) boleh dilatih sebagai pembanding, dengan label "speaker-dependent". Angkanya hampir pasti lebih tinggi dan tidak boleh dipakai untuk klaim generalisasi.

### 5.3 Prosedur anti-bocor antar subjek

**A. Pengecekan otomatis** (`check_leakage`, dijalankan saat membuat split dan oleh `verify_split_files`):

| Cek | Jenis | Isi |
|---|---|---|
| Speaker di >1 split | **Error** | Inti anti-bocor: satu subjek hanya boleh ada di satu split |
| Path duplikat | **Error** | File yang sama muncul dua kali atau di dua split |
| md5 identik lintas split | **Error** (dengan `--hash`) | Konten audio sama dengan nama file berbeda |
| Split kosong | **Error** | train/val/test harus terisi |
| Kelas hilang di sebuah split | **Error** | Setiap emosi harus ada di train, val, dan test |
| Bahasa tidak terwakili di val/test | Peringatan | Generalisasi bahasa itu tidak terukur |
| Split tidak seimbang antar kelas | Peringatan | Rasio max/min > 1,5 |
| Nomor ucapan sama di train dan test | Info | Wajar di ESD (lihat catatan teks paralel) |
| Tiap speaker jadi test tepat 1× | Assert (kfold) | Memastikan tidak ada speaker terlewat atau ganda |

**B. Aturan prosedural** (tidak bisa dicek otomatis, harus kamu patuhi):

1. **Split dulu, baru yang lain.** Statistik normalisasi, `StandardScaler`, PCA, pemilihan fitur, dan trim threshold yang dipelajari dari data harus di-*fit* hanya pada **train**, lalu diterapkan ke val/test.
2. **Augmentasi hanya pada train**, dan hanya setelah split dibuat.
3. **Val untuk semua keputusan** (hyperparameter, early stopping, pemilihan model/epoch). **Test dibuka sekali** di akhir. Jangan ubah apa pun setelah melihat hasil test.
4. **Mode k-fold:** fit ulang scaler dan pilih hyperparameter di dalam tiap fold (train/val fold itu saja), bukan memakai test fold.
5. **Backbone pretrained:** WavLM/wav2vec2 aman karena dilatih tanpa label ESD. Jangan memakai model SER pretrained yang sudah dilatih di ESD (atau di dataset uji cross-corpus), karena itu bocor ke test.
6. **Cross-corpus:** corpus uji (RAVDESS, CREMA-D) tidak boleh dipakai untuk memilih epoch, threshold, atau hyperparameter.
7. **Normalisasi per speaker** (mis. CMVN per speaker) memakai statistik dari semua ucapan speaker itu. Ini tidak membocorkan label, tetapi di dunia nyata statistik itu belum tersedia di awal. Hindari, atau tulis di laporan.
8. **Teks paralel di ESD.** Nomor ucapan yang sama (teks yang sama) diucapkan oleh banyak speaker dalam satu bahasa (verifikasi lewat file `.txt`). Model akustik tidak memakai teks, jadi ini bukan kebocoran subjek, tetapi sebutkan di laporan. Bila ingin lebih ketat, pakai split yang juga memisahkan nomor ucapan.
9. **Catat bukti.** Simpan `split_report.json` dan versi hasil `verify_split_files` bersama tiap eksperimen.

### 5.4 Preprocessing

Urutan: muat → mono → resample 16 kHz → trim silence → normalisasi → crop/pad 4 s.

- Train: random crop. Val/test: center crop atau pad dengan nol.
- Simpan semua parameter di `configs/base.yaml` dan salin ke folder hasil percobaan.

### 5.5 Tiga jalur model

**A. Baseline klasik (cepat, mudah dijelaskan)**
- Fitur: MFCC (40) + delta + delta-delta, F0, energi; ringkas jadi statistik (mean, std, min, max) per ucapan. Alternatif: eGeMAPS via `opensmile`.
- `StandardScaler` (fit di train saja) → SVM RBF / Random Forest / Logistic Regression.
- Fungsi: batas bawah performa. Jika model dalam tidak mengalahkan ini, ada yang salah.

**B. CNN di log-mel**
- Log-mel 64–80 band, `n_fft=1024`, `win=400` (25 ms), `hop=160` (10 ms).
- SpecAugment (time mask + freq mask).
- CNN kecil atau blok ResNet ringan → global average pooling → linear 5 kelas.

**C. Fine-tune self-supervised (kemungkinan terbaik)**
- Backbone: `microsoft/wavlm-base-plus` (default). Karena ESD berisi Mandarin dan Inggris, bandingkan dengan `facebook/wav2vec2-xls-r-300m` yang multibahasa. Pastikan ID model di Hugging Face sebelum dipakai.
- Head: mean pooling atas `last_hidden_state` → dropout → linear.
- Tahap 1: bekukan feature encoder CNN, latih head + lapisan atas. Tahap 2 (opsional): buka sebagian lapisan transformer dengan LR kecil.
- Hyperparameter awal: LR 2e-5 (backbone) dan 1e-3 (head), batch 16, grad accumulation 2, AMP fp16, 5–10 epoch.

Sketsa (belum diuji):

```python
import torch, torch.nn as nn
from transformers import AutoModel

class SERModel(nn.Module):
    def __init__(self, name="microsoft/wavlm-base-plus", n_cls=5, p=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.backbone.freeze_feature_encoder()
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(h, n_cls))

    def forward(self, wav):                        # wav: (B, T) float32 16 kHz
        x = self.backbone(wav).last_hidden_state   # (B, frames, H)
        return self.head(x.mean(dim=1))
```

### 5.6 Augmentasi

| Boleh | Hati-hati |
|---|---|
| Noise latar (mis. MUSAN), gain acak, RIR/reverb ringan, SpecAugment | **Pitch shift dan time-stretch besar** mengubah sinyal emosi itu sendiri. Batasi sangat kecil atau jangan dipakai. |

### 5.7 Training

- Optimizer AdamW, scheduler cosine/OneCycle dengan warmup.
- Pilih checkpoint terbaik berdasarkan **val UAR** (`balanced_accuracy_score`), early stopping patience 5–8 epoch.
- Log ke TensorBoard: loss, UAR, confusion matrix per epoch.
- Satu eksperimen = satu folder `runs/<nama>/` berisi `config.yaml`, `best.pt`, `split_report.json`, `report/` (dari `metrics.py`).

### 5.8 Evaluasi dan metrik

Kode: `src/metrics.py`. Langkahnya: (1) jalankan model pada **test** (dan val) dan simpan `predictions.csv`, (2) jalankan `metrics.py` atau panggil dari kode.

**Format `predictions.csv`:**

| Kolom | Wajib | Keterangan |
|---|---|---|
| `y_true`, `y_pred` | Ya | Indeks kelas (int) atau nama kelas |
| `speaker`, `lang`, `path` | Disarankan | Untuk metrik per speaker/bahasa dan analisis kesalahan |
| `p_neutral`, `p_happy`, `p_angry`, `p_sad`, `p_surprise` | Opsional | Probabilitas softmax, untuk log-loss, ECE, AUC, top-2 |

```bash
python src/metrics.py --pred runs/exp1/predictions.csv --out runs/exp1/report
```

Atau dari kode:

```python
import sys; sys.path.append("src")
from metrics import compute_metrics, save_report

CLASSES = ["neutral", "happy", "angry", "sad", "surprise"]
res = compute_metrics(y_true, y_pred, CLASSES, y_prob=probs, speakers=spk, langs=lang)
print(save_report(res, "runs/exp1/report", y_true, y_pred, CLASSES, y_prob=probs,
                  paths=paths, speakers=spk, langs=lang))
```

Sketsa pengumpul prediksi (belum diuji):

```python
@torch.no_grad()
def collect(model, loader, device):
    model.eval(); Y, P = [], []
    for wav, y in loader:
        P.append(torch.softmax(model(wav.to(device)), -1).cpu()); Y.append(y)
    P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy()
    return Y, P.argmax(1), P
```

**Metrik yang dihitung:**

| Metrik | Arti dan cara membaca |
|---|---|
| **UAR** | Rata-rata recall per kelas. **Metrik utama SER** karena adil antar kelas |
| Accuracy (WAR) | Proporsi benar secara keseluruhan |
| Macro-F1 / Weighted-F1 | Keseimbangan precision-recall antar kelas (macro) atau berbobot jumlah sampel |
| Macro precision / recall | Rata-rata antar kelas |
| Per kelas | Precision, recall, F1, support untuk tiap emosi (`per_class.csv`) |
| Confusion matrix | Hitungan dan versi ternormalisasi per baris, plus gambar `.png` |
| Top confusions | 5 pasangan yang paling sering tertukar (biasanya neutral↔sad, happy↔surprise) |
| **MCC** dan **Cohen's kappa** | Koreksi terhadap kebetulan. 0 ≈ acak, 1 = sempurna |
| Log loss | Kualitas probabilitas (makin kecil makin baik) |
| **ECE** | Kalibrasi: selisih confidence dan akurasi nyata. Di atas ~0,1 berarti model terlalu percaya diri |
| Top-2 accuracy | Label benar ada di dua tebakan teratas |
| ROC-AUC (OvR, macro) | Kemampuan memisahkan tiap kelas dari sisanya |
| **Per speaker** | Akurasi, UAR, macro-F1 tiap subjek (`per_speaker.csv`). Variasi besar = model tidak stabil |
| **Per bahasa** | Perbandingan zh vs en (`per_lang.csv`) |
| **CI 95% bootstrap per subjek** | Interval kepercayaan untuk UAR, accuracy, macro-F1 dengan sampling ulang subjek, bukan file |

Catatan membaca hasil:
- Bandingkan dengan peluang acak (20% untuk 5 kelas) dan dengan baseline A.
- Dengan hanya 4 subjek test, CI akan lebar. Itu jujur. Pakai `kfold` dan laporkan mean ± std antar fold untuk angka yang lebih stabil.
- Laporkan UAR **dan** MCC. Jangan hanya accuracy.

**Cross-corpus** (latih di ESD, uji di corpus lain tanpa fine-tune ulang) dengan label irisan:

```python
from metrics import restrict_labels, compute_metrics
# CREMA-D tidak punya 'surprise' -> 4 kelas. RAVDESS punya kelima kelas.
yt, yp, pr, names = restrict_labels(y_true, y_prob, CLASSES, keep=["neutral", "happy", "angry", "sad"])
res = compute_metrics(yt, yp, names, pr)
```

`restrict_labels` membuang baris dengan label di luar irisan dan menormalkan ulang probabilitas pada kelas irisan. Tulis di laporan kelas mana yang dibuang.

**Ablasi:** A vs B vs C; dengan vs tanpa augmentasi; panjang crop 3 s vs 4 s; speaker-dependent vs speaker-independent; `fixed` vs `stratified`.

### 5.9 Export dan inferensi

- Simpan `best.pt`, `label_map.json`, dan parameter preprocessing (SR, panjang crop, normalisasi).
- `predict.py`: muat wav → preprocess yang sama → softmax → label + confidence.
- Audio panjang: jendela 4 s, hop 2 s, rata-ratakan probabilitas.
- Opsional: export ONNX untuk model CNN dan ukur latensi di CPU.

---

## 6. Struktur proyek

```
ser-pipeline/
├── configs/
│   └── base.yaml
├── data/
│   ├── manifest.csv
│   ├── split.csv
│   ├── train.csv  val.csv  test.csv
│   ├── split_report.json
│   └── kfold/fold0..fold4/{train,val,test}.csv      # bila mode kfold
├── src/
│   ├── data/
│   │   ├── make_split.py        # manifest + split + cek anti-bocor (CLI)
│   │   ├── dataset.py
│   │   └── augment.py
│   ├── features/    # handcrafted.py, melspec.py
│   ├── models/      # cnn.py, ssl_head.py
│   ├── metrics.py               # semua metrik + laporan (CLI)
│   ├── train.py
│   ├── evaluate.py              # kumpulkan prediksi -> predictions.csv
│   └── predict.py
├── tests/
│   └── test_split_metrics.py    # 16 tes untuk make_split.py dan metrics.py
├── runs/            # satu folder per eksperimen
│   └── exp1/{config.yaml, best.pt, predictions.csv, report/}
├── reports/
├── requirements.txt
└── README.md
```

Contoh `configs/base.yaml`:

```yaml
seed: 42
data:
  root: "D:/datasets/archive/Emotion Speech Dataset"
  train_csv: data/train.csv
  val_csv: data/val.csv
  test_csv: data/test.csv
  sr: 16000
  crop_sec: 4.0
  trim_db: 30
  classes: [neutral, happy, angry, sad, surprise]
split:
  mode: stratified      # fixed | stratified | kfold
  n_test_per_lang: 2
  n_val_per_lang: 1
model:
  type: wavlm           # handcrafted | cnn | wavlm
  name: microsoft/wavlm-base-plus
  freeze_encoder: true
train:
  epochs: 10
  batch_size: 16
  grad_accum: 2
  lr_backbone: 2.0e-5
  lr_head: 1.0e-3
  label_smoothing: 0.1
  amp: true
  patience: 6
```

---

## 7. Risiko dan jebakan

| Risiko | Dampak | Penanganan |
|---|---|---|
| Kebocoran speaker | Akurasi palsu tinggi | Split per speaker, `check_leakage`, `verify_split_files` di awal training |
| Split `fixed` (0001–0015) tidak mewakili kedua bahasa di val/test | Generalisasi Mandarin tidak terukur | Pakai `stratified`, atau akui batasannya di laporan |
| Statistik normalisasi/scaler dipelajari dari seluruh data | Kebocoran halus ke test | Fit hanya di train (aturan 5.3-B nomor 1) |
| Test dipakai untuk tuning | Estimasi terlalu optimistis | Tuning di val, test dibuka sekali |
| Hening/trim berbeda antar emosi | Model membaca durasi hening, bukan emosi | Trim silence, cek distribusi durasi per kelas |
| Data akting dan kalimat paralel | Tidak mewakili emosi spontan | Uji cross-corpus, sebut keterbatasan di laporan |
| Test hanya 3–4 speaker | Angka berfluktuasi, CI lebar | Group K-Fold, laporkan mean ± std |
| Augmentasi pitch/tempo | Merusak sinyal emosi | Batasi atau matikan |
| Lisensi dataset | Masalah penggunaan | Cek ketentuan ESD sebelum publikasi |

---

## 8. Rencana kerja (perkiraan 5 hari)

| Hari | Target |
|---|---|
| 1 | Audit data, jalankan `make_split.py` (fixed dan stratified), verifikasi bahasa speaker, `pytest` |
| 2 | Baseline A (SVM) + jalur B (CNN log-mel), simpan `predictions.csv` |
| 3 | Jalur C (WavLM) latih dan tuning singkat di val |
| 4 | `metrics.py` untuk semua model, Group K-Fold, cross-corpus |
| 5 | Ablasi, export + `predict.py`, laporan |

---

## 9. Checklist deliverable

- [ ] `manifest.csv`, `split.csv`, `train/val/test.csv`, `split_report.json`
- [ ] Bukti anti-bocor: keluaran `make_split.py` dan `--verify` lulus, tanpa speaker yang sama di dua split
- [ ] `pytest tests/` lulus di mesinmu
- [ ] Hasil baseline A, B, dan C dalam satu tabel perbandingan
- [ ] Untuk tiap model: UAR (+CI), macro-F1, MCC, kappa, ECE pada test speaker-independent
- [ ] Confusion matrix + hasil per speaker dan per bahasa
- [ ] Hasil cross-corpus dengan label irisan yang dijelaskan
- [ ] `best.pt`, `label_map.json`, `predict.py` yang bisa dijalankan
- [ ] Catatan keterbatasan (data akting, jumlah speaker, lisensi, teks paralel)

---

## 10. Pengembangan lanjutan

- Fine-tune pada data bahasa Indonesia (mis. IndoWaveSentiment) untuk domain lokal. Pastikan kelas yang dipetakan benar-benar sepadan.
- Tambah kelas emosi dari dataset lain dengan pemetaan label yang konsisten.
- Fusi dengan teks (hasil STT) untuk valence pada ucapan natural.
- Model dimensi (valence/arousal) bila datanya tersedia.
