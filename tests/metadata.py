import json

# Initialize an empty list to store the metadata
metadata = []

# Iterate through the image files
for i in range(209):
    # Create a dictionary to store the metadata for the current image
    data = {}
    data["file_name"] = f"drone shot tracking around powerful geyser in rotorua FRAME{i}.jpg"
    data["text"] = f"drone shot tracking around powerful geyser in rotorua FRAME{i}"
    metadata.append(data)

# Write the metadata to a jsonl file
with open("datatest/metadata.jsonl", "w") as f:
    for item in metadata:
        json.dump(item, f)
        f.write("\n")
