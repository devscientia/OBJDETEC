
import os
from dotenv import load_dotenv

# Load all key-value pairs from .env into environment variables
load_dotenv()

# Retrieve the variables using standard os.getenv
api_key = os.getenv("API_KEY")
db_url = os.getenv("DATABASE_URL")
# All env variables read as strings; convert types manually if needed
debug_mode = os.getenv("DEBUG_MODE") == "True" 

print(f"API Key: {api_key}")
print(f"Database: {db_url}")
print(f"Debug Active: {debug_mode}")

