# Week-2-Team-9

# Multi-Modal Agentic AI System for Medical Diagnosis and Triage

A full-stack AI-powered medical triage platform that combines chest X-ray analysis, structured Electronic Health Records (EHR), and clinical text processing to generate diagnosis recommendations and triage decisions.

The system integrates three machine learning models, a LangGraph.js agent workflow, MedGemma-27B-IT reasoning, Retrieval-Augmented Generation (RAG), and automated appointment booking through Google services.

---

# Project Objective

The project aims to provide a real-time diagnostic and triage assistant for chest and cardiovascular conditions by combining:

- Chest X-ray image classification
- Structured EHR analysis
- Clinical Named Entity Recognition (NER)
- Agent-based reasoning
- Medical knowledge retrieval
- Automated appointment booking

The final system returns a structured diagnosis report with one of four triage outcomes:

- EMERGENCY
- URGENT
- ROUTINE
- SELF_CARE

---

# System Architecture

## Model 1 — ChestVision

**Type:** Custom Neural Network

**Technology:**
- ResNet-50
- ViT-B/16

**Purpose:**
- Multi-label classification of 14 thoracic conditions from chest X-rays

**Output:**
- Disease predictions
- Explainability heatmaps (Grad-CAM)

---

## Model 2 — ClinicalFusion

**Type:** Hybrid Model

**Technology:**
- XGBoost
- OpenMed NER Encoder Fusion

**Purpose:**
- Diagnosis classification (5 categories)
- Severity prediction (0–3)

**Output:**
- Diagnosis category
- Severity score

---

## Model 3 — OpenMed NER

**Type:** Pretrained + Fine-Tuned Model

**Technology:**
- OpenMed-NER-DiseaseDetect-SuperClinical-434M

**Purpose:**
- Extract clinical entities from medical notes

**Entities:**
- Disease
- Anatomy
- Pharmaceutical terms

---

## Agent Layer

**Technology:**
- LangGraph.js
- MedGemma-27B-IT

**Purpose:**
- Coordinate all model outputs
- Generate final diagnosis reasoning
- Produce triage recommendations

---

## Knowledge Retrieval Layer

**Technology:**
- ChromaDB
- A-Z Medical Encyclopedia

**Purpose:**
- Retrieval-Augmented Generation (RAG)
- Post-diagnosis knowledge enrichment

---

## Application Layer

**Technology:**
- Next.js 14
- Clerk Authentication
- NeonDB
- Google Calendar API
- Gmail API

**Purpose:**
- Patient-facing application
- Appointment scheduling
- Real-time diagnostic workflow

---

# Datasets

## 1. NIH ChestX-ray14

Used for Model 1 (ChestVision)

- 112,120 chest X-ray images
- 30,805 patients
- 14 thoracic disease labels

---

## 2. Synthea Synthetic EHR

Used for Model 2 (ClinicalFusion)

Contains:

- Patient records
- Conditions
- Observations
- Medications
- Procedures
- Clinical notes

---

## 3. Medical Question Answering Dataset

Used for Model 3 (OpenMed NER)

Sources include:

- MedQA
- MedMCQA
- PubMedQA
- HealthSearchQA

---

# Project Workflow

## Step 1 — Data Preparation

### NIH ChestX-ray14 Processing

- Download dataset
- Validate files
- Create train/validation/test splits
- Resize and normalize images
- Prepare labels

### Synthea EHR Processing

- Load patient records
- Engineer structured features
- Create diagnosis labels
- Scale numerical features
- Save encoders

### NER Dataset Preparation

- Extract clinical notes
- Generate BIO tags
- Create train/validation/test splits

### Knowledge Base Creation

- Chunk medical encyclopedia
- Generate embeddings
- Store in ChromaDB

---

## Step 2 — Train OpenMed NER

### Fine-Tuning

- Train OpenMed NER on Synthea notes
- Apply transfer learning
- Evaluate using seqeval

### Evaluation Targets

- Entity F1 ≥ 0.82
- Disease Recall ≥ 0.85
- Transfer Gain ≥ 3 F1 points

### Export

- Convert to ONNX
- Apply INT8 quantization

---

## Step 3 — Train ClinicalFusion

### XGBoost Baseline

- Train structured EHR classifier
- Perform GridSearchCV

### Fusion Model

Combine:

- XGBoost leaf features
- OpenMed CLS embeddings
- NER entity vectors

### Evaluation Targets

- Macro F1 ≥ 0.80
- False-low rate < 2%

### Export

- Convert to ONNX
- Validate latency and parity

---

## Step 4 — Train ChestVision

### Model Training

- Train ResNet-50 and/or ViT-B/16
- Apply Optuna hyperparameter optimization
- Monitor validation metrics

### Explainability

- Generate Grad-CAM heatmaps
- Validate anatomical relevance

### Evaluation Targets

- Mean AUC ≥ 0.82
- Per-class AUC ≥ 0.75

### Export

- Convert to ONNX
- Apply INT8 quantization

---

## Step 5 — Build Application

### Authentication

- Clerk integration
- User account management

### Database

- NeonDB configuration
- Schema creation

### Inference APIs

Build API routes for:

- ChestVision
- ClinicalFusion
- OpenMed NER

### Agent Workflow

Create LangGraph.js workflow:

1. Patient Intake
2. NER Analysis
3. X-ray Analysis
4. Clinical Analysis
5. Medical Reasoning
6. RAG Retrieval
7. Diagnosis Generation
8. Triage Decision

### Appointment Services

- Google Calendar integration
- Gmail notifications

---

## Step 6 — Testing

### Data Validation

Verify:

- Dataset integrity
- Split correctness
- Label quality
- Feature quality

### Unit Testing

Test:

- Preprocessing functions
- Metric calculations
- ONNX parity
- Triage logic

### Integration Testing

Test:

- Agent workflow
- API routes
- Model interaction

### End-to-End Testing

Validate:

- Complete patient journey
- Diagnosis generation
- Triage assignment
- Appointment booking

---

# Triage Levels

| Level | Action |
|---------|---------|
| EMERGENCY | Immediate A&E guidance |
| URGENT | Fast specialist referral |
| ROUTINE | Standard appointment |
| SELF_CARE | Guidance without referral |

---

# Repository Structure

```text
.
├── data
│   ├── raw
│   ├── processed
│   ├── metadata
│   └── splits
│
├── notebooks
│   ├── 01_eda
│   ├── 02_preprocessing
│   ├── 03_training
│   └── 04_evaluation
│
├── src
│   ├── data_processing
│   ├── models
│   ├── agent
│   ├── api
│   └── app
│
├── models
│   ├── onnx
│   └── quantised
│
├── docs
├── tests
│
├── README.md
├── requirements.txt
├── package.json
├── .env.example
└── .gitignore
```

---

# Development Workflow

## Branch Structure

```text
main
develop
feature/*
model/*
```

---

## Pull Request Process

1. Create feature branch
2. Implement changes
3. Run tests
4. Submit Pull Request
5. Pass CI checks
6. Complete peer review
7. Merge into main

---

# Coding Standards

## Python

- PEP 8
- Black
- isort
- mypy
- Google-style docstrings

## TypeScript

- ESLint
- Prettier
- Strict TypeScript
- JSDoc/TSDoc

---

# Deployment

## Production Stack

### Frontend
- Next.js 14

### Database
- NeonDB

### Authentication
- Clerk

### Vector Database
- ChromaDB

### Deployment
- Vercel

### CI/CD
- GitHub Actions

---

# Success Criteria

## Model 1

- Mean AUC-ROC ≥ 0.82
- Per-class AUC ≥ 0.75

## Model 2

- Macro F1 ≥ 0.80
- False-low rate < 2%

## Model 3

- Entity F1 ≥ 0.82
- Disease Recall ≥ 0.85

## System

- Complete diagnostic workflow
- Successful ONNX deployment
- UAT completed
- End-to-end testing passed

---

# Team

| Member | Role |
|----------|----------|
| Augustine | Team Leader |
| Girish | Scrum Master |
| Abhishek | GitHub Document Manager |
| Israel | Communications Lead |
| Victor | Code Review Manager |
| Karan | Test Plan Manager |

---
