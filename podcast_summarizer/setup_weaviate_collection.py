import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType

# Load environment variables
load_dotenv()

# Connect to Weaviate instance
client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
)

# Optional: delete old collection if it exists
if client.collections.exists("PodcastInsights"):
    client.collections.delete("PodcastInsights")

# Create collection
client.collections.create(
    name="PodcastInsights",
    properties=[
        Property(name="insight", data_type=DataType.TEXT)
    ],
    vectorizer_config=Configure.Vectorizer.text2vec_openai()
)
print("✅ Collection 'PodcastInsights' created successfully.")

# Insert some dummy insights
dummy_insights = [
    {"insight": "Pursuing a personal calling leads to long-term startup success."},
    {"insight": "Niche online bookstores can build loyal customer bases."},
    {"insight": "Risk-taking and embracing failure is key to entrepreneurship."},
    {"insight": "Jeff Bezos began with books but had a vision for everything."},
    {"insight": "Slower, deliberate progress often yields better long-term growth."}
]

collection = client.collections.get("PodcastInsights")
collection.data.insert_many(dummy_insights)
print("✅ Inserted dummy data into 'PodcastInsights'.")
