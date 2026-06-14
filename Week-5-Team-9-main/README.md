# 🏥 Multi-Modal Agentic AI System for Medical Diagnosis and Triage

> **University of Hertfordshire — Final Year Group Project**  
> An autonomous AI agent that diagnoses chest and cardiovascular conditions by sequentially analysing patient symptoms, chest X-ray images, and structured clinical records — then triages and books appropriate specialist appointments.

---

## 📌 Project Status

| Phase   | Description                                  | Status         |
| ------- | -------------------------------------------- | -------------- |
| Phase 1 | Exploratory Data Analysis (EDA)              | ✅ Complete    |
| Phase 2 | Data Preprocessing                           | ✅ Complete    |
| Phase 3 | Feature Engineering                          | ✅ Complete    |
| Phase 4 | Model Building — CNN (ChestVision)           | ✅ Complete    |
| Phase 4 | Model Building — Hybrid EHR (ClinicalFusion) | ✅ Complete    |
| Phase 4 | Model Building — BioClinical QA (Fine-tune)  | 🔄 In Progress |
| Phase 5 | Evaluation & Optimisation                    | ⏳ Pending     |
| Phase 6 | App Deployment (Next.js + LangGraph)         | ⏳ Pending     |

---

## 👥 Team

| Name    | Role                    | GitHub                                                 | Technical Focus                                  |
| ------- | ----------------------- | ------------------------------------------------------ | ------------------------------------------------ |
| Austin  | Team Leader             | [@augustineakauba](https://github.com/augustineakauba) | Model 1 — ChestVision (ViT-B/16)                 |
| Girish  | Scrum Master            | [@GirishGowd](https://github.com/GirishGowd)           | Model 2 — ClinicalFusion training & evaluation   |
| Abishek | GitHub Document Manager | [@abhishek7112000](https://github.com/abhishek7112000) | Data pipeline, ONNX export & quantisation        |
| Israel  | Communications Lead     | [@AjayiIsrael](https://github.com/AjayiIsrael)         | RAG pipeline, ChromaDB, Medical Encyclopedia     |
| Victor  | Code Review Manager     | [@victoran0](https://github.com/victoran0)             | AI Agent — LangGraph.js orchestration            |
| Karan   | Test Plan Manager       | [@k-jay23](https://github.com/k-jay23)—                | Application — Next.js, Google Calendar/Gmail API |

> **Repository:** https://github.com/Professional-Project-Team-9/Week-5-Team-9

---

## 🏗️ System Architecture

```
Patient Input (symptoms + X-ray + clinical record)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│                  LangGraph.js Agent                       │
│  intake_node → route_node → xray_node + clinical_node    │
│  → synthesise_node (MedGemma-27B-IT)                     │
│  → [clarify_node if confidence < 0.60]                   │
│  → rag_node (ChromaDB) → response_node (BioClinical QA)      │
│  → triage_node → appointment booking                     │
└──────────────────────────────────────────────────────────┘
        │                    │
        ▼                    ▼
  Model 1               Model 2
  ChestVision           ClinicalFusion
  (ViT-B/16 ★)         (MLP + BioClinical-
  NIH CXR14             ModernBERT-large)
  14-label ONNX         EHR → 28 labels ONNX
```

### Model Stack

| Component                    | Technology                                 | Dataset                  | Output                               |
| ---------------------------- | ------------------------------------------ | ------------------------ | ------------------------------------ |
| **Model 1 — ChestVision**    | ViT-B/16 fine-tuned (timm) ★               | NIH ChestX-ray14         | 14-label X-ray classification (ONNX) |
| **Model 2 — ClinicalFusion** | MLP + frozen BioClinical-ModernBERT-large  | Synthea Synthetic EHR    | 28-label condition prediction (ONNX) |
| **Model 3 — BioClinical QA** | BioClinical fine-tuned (transfer learning) | Medical QA Dataset (HF)  | Patient-facing clinical response     |
| **Agent LLM**                | MedGemma-27B-IT                            | —                        | DiagnosisReport JSON synthesis       |
| **RAG Layer**                | ChromaDB + BioClinical embeddings          | A-Z Medical Encyclopedia | Clinical QA enrichment               |

---

## 📊 Model Results (Current)

### Model 1 — ChestVision (NIH ChestX-ray14)

Three architectures trained and compared on 112,120 chest X-rays (78,566 train / 17,063 val / 16,491 test):

| Architecture                            | Training     | Val AUC (best) | Test AUC   | Test Macro-F1 | Status       |
| --------------------------------------- | ------------ | -------------- | ---------- | ------------- | ------------ |
| ResNet-50 (from scratch, baseline)      | 30 epochs    | 0.7955         | 0.7893     | 0.2439        | Baseline     |
| **ResNet50V2** (fine-tuned, timm)       | 20 ep (5+15) | 0.7685         | 0.7666     | 0.2517        | Comparison   |
| **EfficientNetV2B0** (fine-tuned, timm) | 20 ep (5+15) | 0.8014         | 0.7913     | 0.3000        | Comparison   |
| **ViT-B/16** (fine-tuned, timm) ★       | 20 ep (5+15) | **0.8253**     | **0.8175** | **0.3265**    | **Selected** |

> **ViT-B/16 selected** as the final deployed model — highest validation and test AUC across all four architectures.

**Training configuration:**

- Loss: `BCEWithLogitsLoss` with per-class `pos_weight = (N⁻/N⁺)`, capped at 20
- Two-phase schedule: Phase 1 head-only `lr=1e-3` × 5 epochs → Phase 2 full fine-tune `lr=1e-5` × 15 epochs
- Augmentation: `RandomResizedCrop`, `HorizontalFlip`, `ColorJitter`, `RandomAffine`
- Optimiser: AdamW · Early stopping patience=6

---

### Model 2 — ClinicalFusion (Synthea EHR)

Hybrid model trained on 1,376 synthetic patients, 80 tabular features, 28 condition labels:

| Metric                     | Score      |
| -------------------------- | ---------- |
| **Best Val Macro AUC-ROC** | **0.8524** |
| F1 Macro (threshold=0.5)   | 0.4387     |
| F1 Micro (threshold=0.5)   | 0.4636     |

**Top per-label AUC-ROC:**

| Condition                      | AUC-ROC |
| ------------------------------ | ------- |
| Coronary Heart Disease         | 1.0000  |
| Diabetes                       | 1.0000  |
| Normal Pregnancy               | 0.9902  |
| Neuropathy (T2DM)              | 0.9897  |
| Hypertension                   | 0.9849  |
| Prediabetes                    | 0.9845  |
| Chronic Obstructive Bronchitis | 0.98+   |

**Architecture:**

- Tabular MLP branch: 80 features → `Linear(256) → BN → ReLU → Dropout → Linear(128) → Linear(64)`
- Text branch: BioClinical-ModernBERT-large (frozen) → 1024-dim CLS → projected 256-dim
- Fusion: `concat[256 ‖ 256]` = 512-dim → classification heads
- Clinical notes: SOAP-format summaries generated via Groq LLM, cached for training

---

## 🗂️ Datasets

### Dataset 1 — NIH ChestX-ray14

- **Source:** NIH Clinical Center
- **Size:** ~45 GB · 112,120 PNG chest X-rays · 30,805 patients
- **Labels:** 14 thoracic diseases (Cardiomegaly, Pneumonia, Effusion, Atelectasis, etc.)
- **Download:** https://nihcc.app.box.com/v/ChestXray-NIHCC
- **Licence:** NIH Public Use (free for research/education; cite Wang et al., 2017)
- **Ethics:** Fully de-identified (HIPAA Safe Harbor); no registration required

### Dataset 2 — Synthea Synthetic EHR

- **Source:** MITRE Synthea / Kaggle
- **Size:** ~500 MB compressed · 1,376 patients used
- **Tables:** patients, conditions, observations, medications, encounters, procedures, careplans, allergies, immunizations, claims
- **Download:** https://www.kaggle.com/datasets/imtkaggleteam/synthetic-medical-dataset
- **Licence:** Apache 2.0 (open source; no real patients)
- **Ethics:** Entirely synthetic — no GDPR obligations

### Dataset 3 — Medical QA (HuggingFace)

- **Source:** Malikeh1375 / HuggingFace Hub
- **Size:** ~2.3 GB · MedQA + MedMCQA + PubMedQA + HealthSearchQA
- **Access:** `load_dataset("Malikeh1375/medical-question-answering-datasets")`
- **Licence:** Apache 2.0
- **Ethics:** Aggregates public NLP corpora; no private patient records

### Knowledge Base (RAG — not a training dataset)

- **Source:** A-Z Family Medical Encyclopedia (Internet Archive)
- **Usage:** Chunked → ChromaDB embeddings → RAG QA enrichment at runtime

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.11+
- Node.js 20+
- CUDA-capable GPU (recommended for model training)
- 100 GB+ storage for datasets

### Python Environment

```bash
# Clone the repository
git clone https://github.com/Professional-Project-Team-9/Week-5-Team-9.git
cd Week-5-Team-9

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Key Python Dependencies

```txt
torch>=2.2.0
torchvision>=0.17.0
timm>=0.9.16
transformers>=4.40.0
datasets>=2.19.0
xgboost>=2.0.0
lightgbm>=4.3.0
scikit-learn>=1.4.0
imbalanced-learn>=0.12.0
optuna>=3.6.0
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
onnx>=1.16.0
onnxruntime>=1.18.0
groq>=0.5.0
```

---

## 🚀 Running the Notebooks

### Model 1 — ChestVision (Image Classification)

```bash
# 1. Download NIH ChestX-ray14 to data/raw/nih_chestxray/
#    Place images in:  data/raw/nih_chestxray/images/
#    Place CSVs in:    data/raw/nih_chestxray/

# 2. Run ViT-B/16 fine-tuned (selected model)
jupyter notebook notebooks/ChestXray_Competition_timm_ViT_ResNet.ipynb

# 3. Run EfficientNetV2B0 (comparison model)
jupyter notebook notebooks/ChestXray_Competition_EfficientNet.ipynb

# 4. Run ResNet-50 from scratch (baseline)
jupyter notebook notebooks/ChestXray_Model_Training_ResNet_scratch.ipynb
```

### Model 2 — ClinicalFusion (EHR Hybrid Model)

```bash
# 1. Download Synthea dataset to data/raw/synthea/

# 2. Set your Groq API key (for clinical note generation)
export GROQ_API_KEY=your_key_here

# 3. Run the hybrid EHR model
jupyter notebook notebooks/hybrid_ehr_model.ipynb
```

> **Note:** Clinical notes are generated via Groq LLM and cached to `clinical_notes_cache.csv` on first run. Subsequent runs load from cache — no repeated API calls.

---

## 🧪 Training Details

### Model 1 — Two-Phase Fine-Tuning (timm)

```python
# Phase 1: Train classification head only
for param in model.parameters():
    param.requires_grad = False
for param in model.get_classifier().parameters():
    param.requires_grad = True
optimizer = AdamW(model.parameters(), lr=1e-3)
# Train for 5 epochs

# Phase 2: Full model fine-tuning
for param in model.parameters():
    param.requires_grad = True
optimizer = AdamW(model.parameters(), lr=1e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=15)
# Train for 15 epochs
```

### Model 2 — Hybrid Architecture

```python
# Text branch: BioClinical-ModernBERT-large (frozen)
BIOCLINICAL_MODEL = 'thomas-sounack/BioClinical-ModernBERT-large'
# hidden_size = 1024 → projected to 256-dim

# Tabular branch: deep MLP
# Linear(80, 256) → BN → ReLU → Dropout(0.3) → Linear(256,128)
# → ReLU → Dropout(0.3) → Linear(128, 64) → projection Linear(64, 256)

# Fusion: concat[256 ‖ 256] = 512-dim → 28 label outputs
# Loss: BCE with per-label pos_weight + label smoothing
```

---

## 📈 Evaluation Targets (KPIs)

| Model                  | Metric                  | Target      | Achieved                                 |
| ---------------------- | ----------------------- | ----------- | ---------------------------------------- |
| ChestVision (ViT-B/16) | Mean AUC-ROC            | ≥ 0.82      | ✅ **0.8175**                            |
| ChestVision (ViT-B/16) | Per-class AUC           | ≥ 0.75      | 🔄 In progress                           |
| ChestVision (ViT-B/16) | ONNX CPU latency        | < 50 ms/img | ⏳ Pending export                        |
| ClinicalFusion         | Macro AUC-ROC           | ≥ 0.85      | ✅ **0.8524**                            |
| ClinicalFusion         | F1 Macro                | ≥ 0.80      | 🔄 **0.4387** (threshold tuning pending) |
| ClinicalFusion         | False-low rate (safety) | < 2%        | ⏳ Pending safety evaluation             |

---

## 🔄 Next Steps

- [ ] Export ViT-B/16 and ClinicalFusion to ONNX and apply INT8 quantisation
- [ ] Fine-tune BioClinical QA model on `Malikeh1375/medical-question-answering-datasets`
- [ ] Build ChromaDB vector store from A-Z Medical Encyclopedia
- [ ] Implement LangGraph.js agent with all nodes
- [ ] Scaffold Next.js application (Clerk auth + NeonDB)
- [ ] Connect Google Calendar / Gmail API for appointment booking
- [ ] Run full test plan (unit tests, model KPI gates, UAT)

---

## 🔐 Ethical Statement

This project uses only de-identified public data (NIH ChestX-ray14) and entirely synthetic records (Synthea, Medical QA). No real patient information is used at any stage. The system is a **decision-support tool only** — not a replacement for clinical judgement. All outputs include a confidence score and a disclaimer directing users to seek in-person evaluation.

| Dataset          | GDPR                                 | Permission                | Collected Ethically               |
| ---------------- | ------------------------------------ | ------------------------- | --------------------------------- |
| NIH ChestX-ray14 | ✅ De-identified (HIPAA Safe Harbor) | ✅ NIH Public Use Licence | ✅ IRB-approved (NIH, 1992–2015)  |
| Synthea EHR      | ✅ Synthetic — no real individuals   | ✅ Apache 2.0             | ✅ Computer-generated (MITRE)     |
| Medical QA       | ✅ Public NLP corpora                | ✅ Apache 2.0             | ✅ Public sources (PubMed, MedQA) |

---

## 📚 Key References

- Wang, X. et al. (2017). _ChestX-ray8: Hospital-scale Chest X-ray Database_. IEEE CVPR.
- Walone, J. et al. (2017). _Synthea: Synthetic patient data_. JAMIA, 25(3):230–238.
- Google Research (2025). _MedGemma Technical Report_. arXiv:2507.05201.
- Rajpurkar, P. et al. (2017). _CheXNet: Radiologist-level pneumonia detection_. arXiv:1711.05225.
- Strick, D. et al. (2025). _Reproducing and Improving CheXNet_. arXiv:2505.06646.
- Li, D. et al. (2024). _MEDIQ: Question-asking LLMs for medical QA_. NeurIPS 2024.
- Sounack, T. et al. (2025). _BioClinical ModernBERT: A State-of-the-Art Long-Context Encoder for Biomedical and Clinical NLP_. arXiv:2506.10896.
- Malikeh1375. (2023). Medical question answering datasets [Dataset]. HuggingFace. https://huggingface.co/datasets/Malikeh1375/medical-question-answering-datasets

---

## 🤝 Contributing

1. Branch off `develop`: `git checkout -b feature/your-feature develop`
2. Commit with prefix: `[DATA]`, `[MODEL]`, `[APP]`, `[DOCS]`, `[FIX]`
3. Open a Pull Request to `develop` — requires one reviewer sign-off (Victor — Code Review Manager)
4. All CI checks must be green before merge (formatting, linting, ONNX parity)

---

## 📄 Licence

Code: **MIT Licence** — see `LICENSE` file.  
Datasets retain their original licences (NIH Public Use; Apache 2.0).  
Model weights: subject to respective model provider terms of use.

---

<div align="center">
  <sub>University of Hertfordshire · Final Year Project · 2025–2026</sub>
</div>
