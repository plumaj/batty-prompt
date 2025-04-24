import time
from openai import OpenAI, RateLimitError, OpenAIError
import pandas as pd
import json
import os

# Define the folder containing the text files
folder_path = "../txt_files"

# Get a list of all the text files in the folder 
file_names = [f for f in os.listdir(folder_path)] 

prompt = """Read the text passage below after these instructions. \
Search for all sentences containing comparisons, metaphors or metonymies for descriptions of war in the text. \
Consider only each sentence individually, do not go beyond the boundary of the sentence. \
Note that the texts are refering to WW 1. \
For each identified sentence, generate a JSON object with the following structure: \
{
  "sentences": [
    {
      "sentence_number": <number of the sentence in the text>,
      "sentence_text": "<text of the sentence>",
      "category": "<either comparison, metaphor or metonymy>",
      "explanation": "<explanation for your choice and which words/phrases describe war>",
      "confidence": "<your level of confidence regarding the categorisation (low, mid, or high)>",
      "sentiment": "<sentiment estimation for this sentence>"
    }
  ]
}
If no relevant sentences are found, return: {"sentences": []}.
Now, analyze the following passage:
"""

api_key = ''  # Replace with your OpenAI API key

client = OpenAI(
    api_key=api_key,
)

def run_prompts(file_names, prompt, folder_path):
    results = []
    json_results = {}
    retry_count = 5  # Number of retries for rate-limiting
    backoff_time = 5  # Time to wait before retrying (in seconds)

    for file_name in file_names:
        # Load the contents of the file
        with open(os.path.join(folder_path, file_name), "r") as f:
            content = f.read()
            print(content)
            date = file_name.split('.')[0]
            full_prompt = prompt + content

            success = False
            retries = 0
            while not success and retries < retry_count:
                try:
                    # Generate completion using OpenAI GPT-4
                    response = client.chat.completions.create(
                        messages=[{"role": "system", "content": "You are a scientific assistant and you reason carefully sentence by sentence your input prompt."},
                                  {"role": "user", "content": full_prompt}],
                        max_tokens=1000,
                        temperature=0.2,
                        response_format={"type": "json_object"},
                        model="gpt-4o"
                    )
                    
                    # Extract the generated completion
                    completion = response.choices[0].message.content
                    
                    # Add to a JSON object
                    json_results[date] = completion
                    success = True  # Mark as successful

                except RateLimitError:
                    retries += 1
                    print(f"Rate limit hit. Retrying in {backoff_time} seconds... (Attempt {retries}/{retry_count})")
                    time.sleep(backoff_time)  # Wait before retrying

                except OpenAIError as e:
                    # Catch other OpenAI API-related errors
                    print(f"OpenAI API error: {str(e)}")
                    retries += 1
                    time.sleep(backoff_time)  # Backoff and retry

                except Exception as e:
                    # Handle unexpected errors
                    print(f"Error processing file {file_name}: {str(e)}")
                    break  # Don't retry on unexpected errors

    # Write the JSON results to a file
    with open('gpt_responses_zero_shot_sentence.json', 'w') as f: # changed to gpt_zero.json
        json.dump(json_results, f)

results = run_prompts(file_names, prompt, folder_path)


