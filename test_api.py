import requests
import os

# The API endpoint URL
url = "http://127.0.0.1:8000/hackrx/run"

# The path to your insurance policy file
# CORRECTED PATH: This has been updated with the path you provided.
file_path = "D:\\Bajaj Hackathon\\docs\\BAJHLIP23020V012223-1.pdf"

# The JSON payload for the questions
questions_data = {
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}

# Ensure the file exists before sending the request
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    # Prepare the multipart/form-data payload with explicit formatting
    files = {
        'file': (os.path.basename(file_path), open(file_path, 'rb'), 'application/pdf')
    }

    print("Sending request to API...")
    try:
        # Send the POST request with JSON body and file
        response = requests.post(
            url,
            files=files,
            json=questions_data
        )

        # Check the response status and content
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Success! Response:")
            response_data = response.json()
            print("API Response:")
            print(f"Status: {response_data.get('status')}")
            print(f"Document: {response_data.get('document')}")
            print("Answers:")
            for i, answer in enumerate(response_data.get('answers', []), 1):
                print(f"{i}. {answer}")
        else:
            print(f"Error: {response.status_code}")
            print("Response Body:")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")