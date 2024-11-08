import requests
import json
import os
import traceback

# Headers for Convai API
headers = {
    'CONVAI-API-KEY': 'your-convai-api-key',
    'Content-Type': 'application/json'
}

URL_UPLOAD = "https://api.convai.com/character/knowledge-bank/upload"
URL_KNOWLEDGE_CONNECT = "https://api.convai.com/character/update"


def create_knowledge_file(product_name, knowledge_content):
    """Create a text file for a product's knowledge content."""
    filename = f"{product_name.replace(' ', '_')}.txt"
    with open(filename, 'w') as file:
        file.write(knowledge_content)
    print(f"Created {filename}")
    return filename


def upload_knowledge_file_to_convai(filename):
    """Upload the knowledge file to Convai and return the knowledge ID."""
    HEADER_UPLOAD = {
        'CONVAI-API-KEY': 'your-convai-api-key'
    }

    with open(filename, "rb") as file:
        form_data = {
            "file_name": file.name,
            "file": file
        }
        response = requests.post(URL_UPLOAD, headers=HEADER_UPLOAD, files=form_data)
        if response.status_code == 200:
            ck_json_data = response.json()
            common_knowledge_id = ck_json_data.get("id", "")
            print(f"Uploaded {filename} and received ID: {common_knowledge_id}")
            return common_knowledge_id
        else:
            print(f"Error uploading {filename}: {response.text}")
            return None


def connect_knowledge_to_character(char_id, knowledge_id):
    """Connect the uploaded knowledge to a Convai character."""
    payload = {
        "charID": char_id,
        "docs": [
            {
                "id": knowledge_id,
                "status": "active"
            }
        ]
    }

    response = requests.post(URL_KNOWLEDGE_CONNECT, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        print("Successfully connected knowledge to character")
    else:
        print(f"Error connecting knowledge to character: {response.text}")


def feed_common_knowledge_from_json(json_filepath, char_id="your-convai-char-id", limit=50, name_key='name'):
    """Read JSON file and feed product information to Convai character."""
    print(f"Feeding common knowledge to Convai from {json_filepath}")
    try:
        with open(json_filepath) as data_file:
            data = json.load(data_file)
            products = data.get("products", [])

            for product in products:
                # Extract product name
                product_name = product.get(name_key, '')
                if not product_name:
                    continue

                # Limit the name length if it exceeds the limit
                product_name = product_name[:limit]

                # Build the knowledge content string
                knowledge_content = []

                description = product.get("description", "No description available")
                if description and description != 'null':
                    knowledge_content.append(f"{product_name} description is: {description}")

                price = product.get("price", None)
                if price is not None:
                    knowledge_content.append(f"{product_name} price is: {price}")

                specifications = product.get("specifications", {})
                for spec_key, spec_value in specifications.items():
                    knowledge_content.append(f"{product_name} {spec_key} is: {spec_value}")

                # Combine content into a single string
                knowledge_string_content = "\n".join(knowledge_content)

                # Create and upload the knowledge file
                filename = create_knowledge_file(product_name, knowledge_string_content)
                knowledge_id = upload_knowledge_file_to_convai(filename)

                # Connect the knowledge to the character
                if knowledge_id:
                    connect_knowledge_to_character(char_id, knowledge_id)

                # Clean up the created file
                os.remove(filename)
                print(f"Deleted {filename}")

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
    char_id = "your-convai-character-id"  # Replace with your Convai character ID
    json_path = './output/ASICS_Team_Catalog_processed.json'
    feed_common_knowledge_from_json(json_path, char_id)
