import os
import json
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict

import fitz  # PyMuPDF
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import Response, Settings

import nemo  # NVIDIA NeMo
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemoguardrails import LLMRails, RailsConfig
import common_knowledge_fill

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for API keys
os.environ["NVIDIA_API_KEY"] = "your-nvidia-api-key"
os.environ["LLAMA_CLOUD_API_KEY"] = "your-llama-cloud-api-key"


@dataclass
class ProductInfo:
    name: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    specifications: Dict[str, str] = None
    tables: List[Dict] = None

    def __post_init__(self):
        self.specifications = self.specifications or {}
        self.tables = self.tables or []

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProductCatalog:
    products: List[ProductInfo] = None
    tables: List[Dict] = None

    def __post_init__(self):
        self.products = self.products or []
        self.tables = self.tables or []

    def to_dict(self) -> Dict:
        return {
            "products": [product.to_dict() for product in self.products],
            "tables": self.tables
        }


class ProductCatalogProcessor:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = NVIDIAEmbedding(model_name="NV-Embed-QA")
        self.llm = NVIDIA(model="meta/llama3-70b-instruct")
        Settings.llm = self.llm
        self.pdf_reader = PyMuPDFReader()
        self.node_parser = SimpleNodeParser.from_defaults()
        self.max_chunk_size = 450

        # Initialize NVIDIA NeMo components
        self.text_normalizer = Normalizer(input_case='cased', lang='en')
        self.rails_config = RailsConfig.from_content(
            """
                rails:
                      # Input rails are invoked when a new message from the user is received.
                      input:
                        flows:
                          - check jailbreak
                          - check input sensitive data
                          - check toxicity
                          - check product information format
                          - ensure content is business-appropriate
                    
                      # Output rails are triggered after a bot message has been generated.
                      output:
                        flows:
                          - self check facts
                          - self check hallucination
                          - check output sensitive data
                          - validate product information
                          - ensure response is appropriate
                    
                      # Retrieval rails are invoked once `$relevant_chunks` are computed.
                      retrieval:
                        flows:
                          - check retrieval sensitive data
                          - ensure content accuracy
                          - verify relevance of information
            """)
        self.guardrails = LLMRails(self.rails_config)

    def split_into_smaller_chunks(self, text: str) -> List[str]:
        """Split text into chunks that respect the token limit."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) // 4 + 1
            if current_length + word_length > self.max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def process_text_with_nemo(self, text: str) -> str:
        """Use NeMo to normalize text."""
        return self.text_normalizer.normalize(text)

    def apply_guardrails(self, text: str) -> str:
        """Use NeMo Guardrails to enforce output rules."""
        return self.guardrails.generate(text)

    def process_tables(self, pdf_path: str) -> List[Dict]:
        tables = []
        pdf_document = fitz.open(pdf_path)
        for page_num, page in enumerate(pdf_document):
            try:
                tab = page.find_tables()
                if tab and tab.tables:
                    for table_idx, table in enumerate(tab.tables):
                        table_data = [[cell.strip() for cell in row if cell.strip()] for row in table.extract()]
                        if table_data:
                            tables.append({
                                "table_id": f"table_{page_num}_{table_idx}",
                                "page": page_num + 1,
                                "content": table_data
                            })
            except Exception as e:
                logger.warning(f"Failed to process tables on page {page_num}: {e}")
        return tables

    def merge_product_infos(self, all_products: List[List[ProductInfo]]) -> List[ProductInfo]:
        merged_products = {}
        for product_list in all_products:
            for product in product_list:
                if not product.name:
                    continue
                if product.name not in merged_products:
                    merged_products[product.name] = product
                else:
                    existing = merged_products[product.name]
                    if not existing.description and product.description:
                        existing.description = product.description
                    if not existing.price and product.price:
                        existing.price = product.price
                    existing.specifications.update(product.specifications or {})
        return list(merged_products.values())

    def parse_llm_response(self, response: Union[Response, str]) -> List[ProductInfo]:
        response_text = response.response if isinstance(response, Response) else str(response)
        try:
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            if start_idx == -1 or end_idx == 0:
                logger.warning("No valid JSON array found in response")
                return []
            json_text = response_text[start_idx:end_idx]
            products_data = json.loads(json_text)
            products = [
                ProductInfo(
                    name=product_data.get('name'),
                    description=self.process_text_with_nemo(product_data.get('description', "")),
                    price=product_data.get('price'),
                    specifications=product_data.get('specifications', {})
                )
                for product_data in products_data
            ]
            return products
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}\nResponse text: {response_text}")
            return []

    def create_structured_query(self, product_type: str) -> str:
        return f"""
        Analyze the provided text and extract detailed product information about {product_type}.
        Focus on finding the following details in the text:
        - Product name/model
        - Product description
        - Price (if mentioned)
        - Technical specifications and features

        Return the information in this exact JSON format:
        {{
            "name": "full product name",
            "description": "comprehensive product description",
            "price": numeric_price_or_null,
            "specifications": {{
                "key1": "value1",
                "key2": "value2"
            }}
        }}

        Guidelines:
        - Include ALL relevant specifications found
        - Convert prices to numeric values (e.g., 1299.99)
        - Use null for missing fields
        - Be precise and factual
        """

    def process_pdf(self, pdf_path: str, product_type: str = "product") -> Dict:
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            documents = self.pdf_reader.load(file_path=pdf_path)
            tables = self.process_tables(pdf_path)
            all_products = []

            for doc in documents:
                chunks = self.split_into_smaller_chunks(doc.text)
                for chunk in chunks:
                    try:
                        normalized_chunk = self.process_text_with_nemo(chunk)
                        index = VectorStoreIndex.from_documents(
                            [Document(text=normalized_chunk)],
                            embed_model=self.embedder,
                            llm=self.llm
                        )
                        query_engine = index.as_query_engine(similarity_top_k=1, response_mode="compact")
                        structured_query = self.create_structured_query(product_type)
                        response = query_engine.query(structured_query)
                        chunk_products = self.parse_llm_response(response)
                        if chunk_products:
                            all_products.append(chunk_products)
                    except Exception as e:
                        logger.warning(f"Skipping chunk due to error: {e}")
                        continue

            merged_products = self.merge_product_infos(all_products)
            catalog = ProductCatalog(products=merged_products, tables=tables)
            result_dict = catalog.to_dict()
            output_path = self.output_dir / f"{Path(pdf_path).stem}_processed.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved processed data to {output_path}")
            return result_dict
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise


def retrieve_product_info_from_pdf(pdf_path):
    processor = ProductCatalogProcessor()
    try:
        json_data = processor.process_pdf(pdf_path=pdf_path, product_type="product")
        print("Processing completed successfully!")
        print(f"Extracted information: {json.dumps(json_data, indent=2)}")

        # feed to convai API
        out_path = f"./output/{Path(pdf_path).stem}_processed.json"
        response = common_knowledge_fill.feed_common_knowledge_from_json(out_path)
        if response:
            convai_response = "Successfully submitted the common knowledge to convai character!"
        else:
            convai_response = "Failed to submit the common knowledge to convai character!"

        return json_data, convai_response
    except Exception as e:
        logger.error(f"Failed to process catalog: {e}")
        return [], ""


if __name__ == '__main__':
    pdf_path = "./catalogue/ASICS_Team_Catalog.pdf"
    retrieve_product_info_from_pdf(pdf_path)
