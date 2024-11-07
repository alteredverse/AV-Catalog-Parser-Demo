# Alteredverse Product Catalog Parser

A Gradio app for generating and feeding structured common knowledge to the Inworld AI Studio API using data from product catalogs in JSON format. The app is hosted on Hugging Face Spaces and can be run locally as well.

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


