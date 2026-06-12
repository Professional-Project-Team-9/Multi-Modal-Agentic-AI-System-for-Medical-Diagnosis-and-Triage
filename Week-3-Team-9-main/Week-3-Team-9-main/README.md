# Week-3-Team-9

# Multi-Modal Agentic AI System for Medical Diagnosis and Triage

## Project Overview

This project develops a Multi-Modal Agentic AI System designed to support preliminary diagnosis and triage of chest and cardiovascular conditions. The system combines medical imaging, structured Electronic Health Records (EHR), clinical text analysis, Retrieval-Augmented Generation (RAG), and Agentic AI orchestration to generate explainable diagnosis reports and triage recommendations.

The solution integrates multiple machine learning models into a unified workflow capable of processing patient symptoms, chest X-rays, and clinical records before producing a structured diagnosis report.

---

## Project Objective

To develop a deployable AI-powered triage assistant capable of analysing multi-modal patient data and generating clinically grounded diagnostic recommendations through the integration of imaging, EHR, and natural language processing models.

---

## System Components

### Model 1 — ChestVision

**Type:** Custom Neural Network

**Dataset:** NIH ChestX-ray14

**Purpose:**
- Analyse chest X-ray images
- Detect 14 thoracic conditions
- Generate imaging-based diagnostic predictions

---

### Model 2 — ClinicalFusion

**Type:** Hybrid Machine Learning Model

**Dataset:** Synthea Synthetic EHR

**Purpose:**
- Analyse structured patient records
- Predict diagnosis categories
- Generate severity scores

**Output Categories:**
- Cardiac
- Respiratory
- Overlap
- Hypertensive Crisis
- Other

---

### Model 3 — OpenMed QA

**Type:** Pre-trained Model with Transfer Learning

**Dataset:** Medical QA Dataset

**Purpose:**
- Process medical language
- Generate patient-friendly responses
- Improve clinical explanation quality

---

### Agent Layer

**Framework:** LangGraph.js

**Reasoning Model:** MedGemma-27B-IT

**Responsibilities:**
- Coordinate all model outputs
- Perform clinical reasoning
- Generate Diagnosis Reports
- Produce triage recommendations

---

### Retrieval-Augmented Generation (RAG)

**Knowledge Base:** A-Z Medical Encyclopedia

**Vector Database:** ChromaDB

**Purpose:**
- Retrieve supporting medical knowledge
- Validate findings
- Enrich diagnostic responses

---

## Workflow

### Step 1 — Patient Intake

Patient provides:
- Symptoms
- Clinical information
- Optional chest X-ray image

---

### Step 2 — Data Processing

The system processes:

#### Imaging Data
- Chest X-ray preprocessing
- Feature extraction
- Image classification

#### Clinical Data
- Structured EHR processing
- Feature engineering
- Severity scoring

#### Medical Text
- Clinical note processing
- Medical language understanding

---

### Step 3 — Model Inference

#### ChestVision
Processes chest X-ray images and predicts thoracic conditions.

#### ClinicalFusion
Processes EHR records and predicts diagnosis category and severity level.

#### OpenMed QA
Generates medically appropriate explanations and responses.

---

### Step 4 — AI Agent Reasoning

LangGraph.js orchestrates all model outputs and passes results to MedGemma-27B-IT for clinical synthesis.

---

### Step 5 — Knowledge Retrieval

The RAG pipeline retrieves supporting information from the Medical Encyclopedia stored within ChromaDB.

---

### Step 6 — Diagnosis Report Generation

The system generates a structured diagnosis report containing:

- Predicted condition(s)
- Confidence level
- Supporting evidence
- Clinical explanation
- Recommended action

---

### Step 7 — Triage Recommendation

The system assigns one of four triage levels:

- EMERGENCY
- URGENT
- ROUTINE
- SELF_CARE

---

## Technology Stack

### Frontend

- Next.js 14
- TypeScript
- Clerk Authentication

### Backend

- Node.js
- LangGraph.js

### Machine Learning

- PyTorch
- ResNet-50
- ViT-B/16
- XGBoost
- OpenMed
- MedGemma-27B-IT

### Data Storage

- NeonDB
- ChromaDB

### Deployment

- Vercel
- HuggingFace Inference Endpoints

### APIs

- Google Calendar API
- Gmail API

---

## Datasets

### NIH ChestXray14

- 112,120 chest X-ray images
- 30,805 patients
- 14 thoracic disease labels

Used for:
- ChestVision model training

---

### Synthea Synthetic EHR

Contains:
- Patient demographics
- Clinical observations
- Medications
- Conditions
- Procedures
- Encounter records

Used for:
- ClinicalFusion model training

---

### Medical QA Dataset

Contains:
- Medical question-answer pairs
- Biomedical terminology
- Clinical knowledge

Used for:
- OpenMed QA fine-tuning

---

## Repository Structure

```text
Week-3-Team-9/

├── data/
│   ├── raw/
│   ├── processed/
│   ├── metadata/
│   └── splits/
│
├── notebooks/
│   ├── 01_eda/
│   ├── 02_preprocessing/
│   ├── 03_training/
│   └── 04_evaluation/
│
├── src/
│   ├── data_processing/
│   ├── models/
│   ├── agent/
│   ├── api/
│   └── app/
│
├── models/
│   ├── onnx/
│   ├── quantised/
│   ├── docs/
│   └── tests/
│
├── README.md
├── requirements.txt
├── package.json
└── .env.example
```

---

## Team Members

| Team Member | Role | Responsibility |
|------------|------|---------------|
| Girish | Scrum Master | Agile sprint management, team planning, ClinicalFusion training and evaluation |
| Augustine | Team Leader | Project direction, project timeline management, ChestVision model training |
| Abhishek | GitHub Document Manager | Repository management, project specifications, ONNX export and quantisation |
| Israel | Communications Lead | Data management planning, ChromaDB implementation, Medical Encyclopedia embedding |
| Victor | Code Review Manager | Pipeline workflow design, code review, LangGraph.js orchestration, MedGemma integration |
| Karan | Test Plan Manager | Test planning, unit testing, UAT coordination, performance benchmarking, frontend integration |

---

## Project Deliverables

- Multi-modal diagnostic pipeline
- Custom neural network for chest X-ray classification
- Hybrid EHR diagnosis and severity prediction model
- Transfer learning medical language model
- LangGraph.js AI Agent
- Retrieval-Augmented Generation pipeline
- Full-stack patient-facing application
- Automated triage recommendation workflow

---

## Ethical Considerations

- No real patient data is used.
- NIH ChestXray14 is de-identified.
- Synthea EHR consists entirely of synthetic patient records.
- Medical QA datasets contain publicly available biomedical content.
- The system is intended as a clinical decision-support tool and not a replacement for professional medical judgement.

---

## University

University of Hertfordshire

Module Project: Multi-Modal Agentic AI System for Medical Diagnosis and Triage
