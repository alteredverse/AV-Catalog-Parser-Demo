# Alteredverse Product Catalog Parser

A Gradio app for generating and feeding structured common knowledge to the Inworld AI Studio API using data from product catalogs in JSON format. The app is hosted on Hugging Face Spaces and can also be run locally.

## 💼 Project Description: Automated PDF Product Catalog Processor for Virtual Sales Avatar

This project automates the extraction, processing, and structured output of product information from PDF-based catalogs, leveraging NVIDIA's powerful AI tools and libraries. The pipeline is designed to handle complex data structures, including tables and embedded text, and formats the output into JSON. Here's a step-by-step breakdown of the process:

### 1. PDF Parsing and Content Extraction
- **Libraries Used**: `fitz` (PyMuPDF) and `llama-index`
- The `PyMuPDFReader` extracts raw text and tables from each page of the PDF. This text is split into smaller, manageable chunks to comply with token limits for processing.

### 2. Text Normalization with NVIDIA NeMo
- **Tool**: `nemo`
- Extracted text chunks are normalized using NVIDIA's NeMo text processing capabilities, which ensure proper case formatting and language-specific corrections.

### 3. Text Embeddings and LLM Querying
- **Embedding Model**: NVIDIA's `NV-Embed-QA` for generating embeddings
- **Language Model**: NVIDIA’s `meta/llama3-70b-instruct` via `llama-index`
- Chunks are embedded using NVIDIA's embedding model, and a vector index is built. The embedded text is queried using an LLM to extract structured product information, following a predefined format.

### 4. Guardrails for Output Validation
- **Library**: `nemoguardrails`
- Guardrails are configured to validate and ensure the LLM's output adheres to specific quality and safety standards. Checks include fact validation, hallucination prevention, and format enforcement.

### 5. Data Structuring and Merging
- **Custom Classes**: `ProductInfo` and `ProductCatalog`
- Parsed product information is structured into dataclasses and merged to eliminate duplicates. The result is a consolidated list of product details with specifications and associated tables.

### 6. Output and Storage
- The final product information is saved as a JSON file, ready for further integration or submission to external APIs like ConvAI for knowledge enhancement.

### 7. Integration and Submission
- The structured data is optionally fed to a common knowledge API for further use in conversational AI systems.


## 🚀 Live Demo URL

**Check out the online demo on Hugging Face Spaces**: [Gradio App on Hugging Face](https://huggingface.co/spaces/Alteredverse/AV-Catalog-Parser-Demo)

## 📁 Project Structure

- **app.py**: The main Gradio app for running the interface.
- **main.py**: Script to process product catalogs and manage communication with the Inworld AI API.
- **common_knowledge_fill.py**: Script that handles creating and feeding structured common knowledge to the API.
- **requirements.txt**: A list of dependencies required for the project.

## 🛠️ Installation Guide

To set up the project on your local machine, follow these steps:

### 1. Create a Conda Environment

Ensure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed, and create a new environment:

```bash
conda create -n nvidia_llama python=3.10
conda activate nvidia_llama
```

### 2. Install Requirements

Use `pip` to install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## ▶️ How to Run the App
To run the Gradio app locally, prepare NVIDIA, LLAMA_CLOUD, CONVAI API Keys, and ConvAI characters. Then put the API keys in the scripts accordingly and use:

```bash
python app.py
```
This will launch the Gradio app in your default browser, where you can interact with the interface.

## 💡 Usage Guide
Input a product catalog: The app takes a .pdf file containing product details.
Generate common knowledge: The app extracts and formats the product information.
Feed to ConvAI API: The processed data is sent to the ConvAI Studio API.

