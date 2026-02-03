from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "google/medgemma-1.5-4b-it"
print(f"Downloading {model_id}...")

# This forces a clean download and cache
model = AutoModelForImageTextToText.from_pretrained(model_id,force_download=True)
processor = AutoProcessor.from_pretrained(model_id)

print("Download complete and verified.")