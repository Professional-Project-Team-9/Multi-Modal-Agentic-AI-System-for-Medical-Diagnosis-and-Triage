# Reproduction-Week-4-Team-9

# Multi-Modal Agentic AI System for Medical Diagnosis and Triage

## Overview

The Multi-Modal Agentic AI System for Medical Diagnosis and Triage is designed to assist with preliminary diagnosis and triage of chest and cardiovascular conditions. The system combines medical imaging, structured Electronic Health Records (EHR), and clinical free-text notes within a unified AI pipeline.

The solution integrates three machine learning models, a reasoning Large Language Model (LLM), and a Retrieval-Augmented Generation (RAG) layer to generate a structured diagnosis report and triage recommendation.

---

## Project Objectives

- Process chest X-ray images for thoracic disease classification.
- Analyse structured Electronic Health Records (EHR) for diagnosis prediction and severity scoring.
- Extract biomedical entities from clinical notes using Named Entity Recognition (NER).
- Combine outputs from all models through an AI agent.
- Generate diagnosis reports with triage recommendations.
- Deploy the complete system as a patient-facing web application.

---

## System Architecture

### Model 1 – ChestVision
Chest X-ray classification model trained using transfer learning.

Architectures evaluated:
- ResNet50V2
- ViT-B/16
- EfficientNetV2B0

Purpose:
- Multi-label classification of 14 thoracic conditions.

Output:
- ONNX Model

---

### Model 2 – ClinicalFusion

Hybrid machine learning model consisting of:

- MLP tabular branch
- Frozen BioClinical-ModernBERT-large text branch

Purpose:
- Diagnosis prediction
- Severity scoring (0–3)

Output:
- ONNX Model

---

### Model 3 – OpenMed NER

Fine-tuned biomedical NER model.

Purpose:
- Extract disease entities
- Extract anatomy entities
- Extract pharmaceutical entities

Output:
- ONNX Model

---

### Agent LLM

- MedGemma-27B-IT

Purpose:
- Final diagnostic reasoning

---

### RAG Layer

Components:
- A-Z Medical Encyclopedia
- ChromaDB

Purpose:
- Context enrichment and evidence retrieval

---

### Application Layer

Technologies:
- Next.js 14
- LangGraph.js
- Clerk Authentication
- NeonDB

Purpose:
- Patient-facing application
- Agent orchestration
- Appointment booking

---

## Datasets

### Dataset 1 – NIH ChestX-ray14

Used for:
- Model 1 (ChestVision)

Contents:
- 112,120 chest X-ray images
- 30,805 patients
- 14 thoracic disease labels

---

### Dataset 2 – Synthea Synthetic EHR

Used for:
- Model 2 (ClinicalFusion)

Contents:
- Patients
- Encounters
- Conditions
- Observations
- Medications
- Procedures

---

### Dataset 3 – Medical Question Answering Dataset

Used for:
- Model 3 (OpenMed NER)

Includes:
- MedQA
- MedMCQA
- PubMedQA
- HealthSearchQA

---

## Team Members

| Name | Role | Technical Focus |
|--------|--------|--------|
| Augustine | Team Leader | Model 1 – ChestVision |
| Girish | Scrum Master | Model 2 – ClinicalFusion |
| Abhishek | GitHub Document Manager | Data Pipeline, ONNX Export & Quantisation |
| Israel | Communications Lead | ChromaDB & RAG |
| Victor | Code Review Manager | LangGraph Agent & MedGemma Integration |
| Karan | Test Plan Manager | Application Development & Testing |

---

## Project Workflow

### Step 1 – Data Preparation

- NIH ChestX-ray14 preprocessing
- Synthea EHR feature engineering
- NER dataset creation
- ChromaDB knowledge base setup

---

### Step 2 – OpenMed NER Training

- Fine-tune OpenMed NER
- Evaluate entity extraction performance
- Export ONNX model

---

### Step 3 – ClinicalFusion Training

- Train tabular baseline
- Integrate BioClinical-ModernBERT-large
- Evaluate hybrid model
- Export ONNX model

---

### Step 4 – ChestVision Training

- Fine-tune ResNet50V2
- Fine-tune ViT-B/16
- Fine-tune EfficientNetV2B0
- Generate Grad-CAM visualisations
- Evaluate model performance
- Export ONNX model

---

### Step 5 – Application Development

- Build Next.js application
- Configure Clerk authentication
- Configure NeonDB
- Create inference API routes
- Integrate LangGraph agent
- Integrate MedGemma-27B-IT
- Integrate ChromaDB
- Integrate Google Calendar and Gmail APIs

---

### Step 6 – Production Deployment

- End-to-end testing
- CI/CD configuration
- Production deployment

---

### Step 7 – Final Submission

- Final report preparation
- Presentation preparation
- Project submission

---

## Testing Strategy

### Data Validation

- Dataset integrity checks
- Data preprocessing validation
- Feature engineering validation
- Dataset split verification

### Unit Testing

- Image preprocessing
- Metric calculations
- ONNX parity testing
- Agent node testing

### Integration Testing

- Model integration
- Agent workflow validation
- API route testing

### End-to-End Testing

- Complete diagnostic workflow
- Diagnosis report generation
- Triage assignment

### User Acceptance Testing

Scenarios:

- Standard patient journey
- Cardiac referral workflow
- Emergency triage workflow
- Invalid file upload handling
- First-time user journey
- Low-confidence agent response workflow

---

## Repository Structure

```text
data/
├── raw/
├── processed/
├── metadata/
└── splits/

notebooks/
├── 01_eda/
├── 02_preprocessing/
├── 03_training/
└── 04_evaluation/

src/
├── data_processing/
├── models/
├── agent/
├── api/
└── app/

models/
├── onnx/
└── quantised/

docs/
tests/

README.md
requirements.txt
package.json
.env.example
.gitignore
```

---

## Development Process

### Sprint Structure

- Sprint Planning
- Daily Standups
- Weekly Team Review
- Sprint Retrospectives

### Communication Channels

- Microsoft Teams
- WhatsApp
- GitHub Issues
- GitHub Pull Requests
- Teams Calendar

### Code Review Process

1. Create feature branch
2. Open Pull Request
3. Run automated checks
4. Peer review
5. Approval
6. Merge into main branch

---

## Deployment Stack

### Frontend
- Next.js 14

### Authentication
- Clerk

### Database
- NeonDB

### Vector Database
- ChromaDB

### Agent Framework
- LangGraph.js

### LLM
- MedGemma-27B-IT

### Deployment Platforms
- Vercel
- Railway

---

## Project Goal

Deliver a deployable AI-powered medical triage assistant capable of processing multi-modal patient data and generating clinically grounded diagnosis reports and triage recommendations through the integration of imaging, EHR data, clinical text, AI reasoning, and retrieval-augmented generation.
