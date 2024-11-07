import os
import gradio as gr
import main


def predict_from_pdf(pdf_file):
    # Use a temporary upload directory
    upload_dir = "./catalogue/"
    os.makedirs(upload_dir, exist_ok=True)

    # Save the uploaded file to the upload directory
    dest_file_path = os.path.join(upload_dir, os.path.basename(pdf_file.name))
    with open(dest_file_path, "wb") as out_file:
        out_file.write(pdf_file.read())

    # Call the main function to process the PDF and return results
    df, response = main.retrieve_product_info_from_pdf(dest_file_path)
    return df, response


# Define PDF examples (ensure these examples exist or remove them)
pdf_examples = [
    ["catalogue/catalog_flexpocket.pdf"],
    ["catalogue/ASICS_Team_Catalog.pdf"],
]

# Define Gradio app details
pdf_title = "Alteredverse::NVIDIA-LLAMA_Index DevCon MVP Demo"
pdf_desc = "NVIDIA-LLAMA_Index demo app for retrieving product info from e-commerce pdf catalogs."
pdf_long_desc = "Upload a product catalog in .pdf format to retrieve product information in JSON format."

# Gradio Interface setup
demo = gr.Interface(
    fn=predict_from_pdf,
    inputs="file",
    outputs=["json", "textbox"],
    examples=pdf_examples,
    title=pdf_title,
    description=pdf_desc,
    article=pdf_long_desc
)

# Launch the app
if __name__ == "__main__":
    # demo.queue().launch(server_name="0.0.0.0", server_port=8080)
    demo.queue().launch()
