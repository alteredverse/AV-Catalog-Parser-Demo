import requests
from requests.auth import HTTPBasicAuth
import traceback
import json

BASE_URL = 'https://api.inworld.ai/studio/v1'
WORKSPACE_ID = 'testing-9njzo'
WORKSPACE_RESOURCE_NAME = f'workspaces/{WORKSPACE_ID}'
STUDIO_API_KEY = 'PeieHFCHCtCdd0fttKirBQeLZ8W8mI2s'
STUDIO_API_SECRET = 'QgAtDfUn9ta5mlY37lWMyIKciiEnzKUTymztXjvNcY5Xj8wWugMpIgopsyRnitmM'

headers = {
    'Content-Type': 'application/json',
    'Grpc-Metadata-X-Authorization-Bearer-Type': 'studio_api'
}


def create_common_knowledge_single(p_StrdisplayName, p_strArrMemoryRecords):
    ck_data = {
        "displayName": p_StrdisplayName,
        "description": p_StrdisplayName,
        "memory_records": p_strArrMemoryRecords
    }
    url = f"{BASE_URL}/{WORKSPACE_RESOURCE_NAME}/common-knowledge"
    response = requests.post(url=url, json=ck_data, headers=headers,
                             auth=HTTPBasicAuth(STUDIO_API_KEY, STUDIO_API_SECRET))
    print(response.text)
    return response.json()


def feed_common_knowledge_from_json(json_filepath, limit=50, name_key='name'):
    print("Feeding common knowledge to Inworld API from", json_filepath)
    try:
        with open(json_filepath) as data_file:
            data = json.load(data_file)
            products = data.get("products", [])

            for product in products:
                # Extract the product name
                product_name = product.get(name_key, '')
                if not product_name:
                    continue  # Skip this product if no name is found

                # Limit the name length if it exceeds the limit
                product_name = product_name[:limit]

                memory_record_array = []

                # Process the product's description and other fields
                description = product.get("description", "No description available")
                if description and description != 'null':
                    memory_record_array.append(f"{product_name} description is {description}")

                price = product.get("price", None)
                if price is not None:
                    memory_record_array.append(f"{product_name} price is {price}")

                # Process the product's specifications
                specifications = product.get("specifications", {})
                for spec_key, spec_value in specifications.items():
                    memory_record_array.append(f"{product_name} {spec_key} is {spec_value}")

                # Create common knowledge entry for this product
                create_common_knowledge_single(product_name, memory_record_array)

        print("Finished feeding common knowledge!")
        return True
    except FileNotFoundError:
        print(f"File not found: {json_filepath}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {json_filepath}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")
        traceback.print_exc()
        return False


if __name__ == '__main__':
    json_path = './output/ASICS_Team_Catalog_processed.json'
    feed_common_knowledge_from_json(json_path)
