import os
import shutil
import gradio as gr
import main


def predict_from_pdf(pdf_file):
    upload_dir = "./catalogue/"
    os.makedirs(upload_dir, exist_ok=True)

    dest_file_path = os.path.join(upload_dir, os.path.basename(pdf_file.name))
    if pdf_file.name != dest_file_path:
        shutil.copy(pdf_file.name, dest_file_path)

    df, response = main.retrieve_product_info_from_pdf(dest_file_path)
    return df, response


# Define PDF examples
pdf_examples = [
    ["catalogue/catalog_flexpocket"],
    ["catalogue/ASICS_Team_Catalog.pdf"],
]

# Define Gradio Interfaces
pdf_title = "Alteredverse::NVIDIA-LLAMA_Index DevCon MVP Demo"
pdf_desc = "NVIDIA-LLAMA_Index demo app for retrieving product info from e-commerce pdf catalogs."
pdf_long_desc = "Upload a product catalog in .pdf file to retrieve the product information in .json format."

demo = gr.Interface(
    fn=predict_from_pdf,
    inputs="file",
    outputs=["json", "textbox"],
    examples=pdf_examples,
    title=pdf_title,
    description=pdf_desc,
    article=pdf_long_desc
)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=8080)
