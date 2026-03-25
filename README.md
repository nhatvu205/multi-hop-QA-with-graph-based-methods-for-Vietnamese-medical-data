# Project Name: MULTI-HOP QA WITH GRAPH-BASED METHODS FOR VIETNAMESE MEDICAL DOMAIN 

## Description
This project is a part of the DS201 course. We research and build a Multi-hop Question Answering system specialized for the medical domain in Vietnamese. By combining a **Knowledge Graph (KG)** for logical reasoning and a **Transformer architecture (ViT5)** for language understanding and generation, the system is capable of solving complex medical questions that require linking information across multiple steps.
## Installation
Since this is a small project a subject, we don't recommend follow 100% what we have done as a scientific roadmap. For further references, please contact us via Contact below.
## Contributors and Contacts
1. Dinh-Nhat Vu - 23521104@gm.uit.edu.vn
2. Thu-Nhi Ho-Huynh - 23521107@gm.uit.edu.vn
3. Quynh-Nhu Le-Diem - 23521122@gm.uit.edu.vn
## 🏗️ System Architecture & Directory Structure

Our methodology for building the Vietnamese medical multi-hop Question Answering system consists of four main stages: retrieving context for QA pairs, recognizing medical entities, extracting relationships between these entities, and constructing Knowledge Graph Embeddings. These graph embeddings are then integrated into a Seq-to-Seq Transformer architecture, combining structured reasoning with natural language understanding.
![pipeline_qa](https://github.com/user-attachments/assets/70093087-be22-4be2-ae0b-afbfadf290d2)
```text
├── Báo cáo/                      # Contains the project report (PDF) and related documents
├── data_exploration/             # Initial Exploratory Data Analysis (EDA)
├── entity-canonicalization/      # Medical entity canonicalization module (mapping variants to standard entities)
├── relations_extraction/         # Relation Extraction to create triplets
│   ├── data/
│   ├── models/
│   ├── output/
│   └── src/
├── knowlegde-base/               # Medical knowledge base management
│   ├── entity_data/              # Original and canonicalized entity data
│   ├── entity_linking/           # Entity Linking module mapping entities to the KB
│   ├── kb_data/                  # Dictionary and base ontology data
│   └── notebooks/                # Jupyter notebooks for KB analysis and testing
├── KG-building/                  # Knowledge Graph construction and management
│   ├── build_KG/                 # Source code for graph initialization
│   ├── data/                     # Graph data formatted for Neo4j
│   ├── vncorenlp/                # Vietnamese Natural Language Processing tool
│   ├── hf_space/                 # Source code for Hugging Face Space (ViT5 Clone)
│   ├── model_hf/                 # Transformer model module (Hugging Face)
│   ├── model_module/             # Scripts for training, KG embedding, and evaluation
│   └── test_result/              # Prediction results and model evaluation logs
├── multi_hop_recognition/        # Multi-hop question recognition and classification
└── demo/                         # Source code for the system's Web Demo interface
```
## Our Contributions
Our primary contribution is the construction of a knowledge graph specifically tailored for the Vietnamese medical domain. This knowledge graph plays a crucial role in reducing the model's hallucinations when dealing with multi-hop questions that require retrieving and logically connecting multiple pieces of information across different entities.

## Our Limitations
Our main limitation lies in the data availability and quality. Currently, high-quality medical datasets created and verified by experts in the market do not contain many multi-hop questions. This contradicts the real-world scenario where patients' queries are often complex and require combining various pieces of information. Because we are not medical doctors or domain experts, we could not generate higher-quality, highly specific multi-hop data from scratch. Instead, we had to rely on a rule-based system to recognize and filter existing multi-hop questions from the available datasets.
