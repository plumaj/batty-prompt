import time
from openai import OpenAI, RateLimitError, OpenAIError
import pandas as pd
import json
import os

# Define the folder containing the text files
folder_path = "../txt_files"

# Get a list of all the text files in the folder with the prefix '1916_03_'
file_names = [f for f in os.listdir(folder_path)] #if f.startswith("1915-03-0")]

prompt = '''Read the text passage below after these instructions. \
Search for all sentences containing the three text-linguistic categories as there are "comparisons", "metaphors" and "metonymies" for descriptions of war in the text. \
Consider only each sentence individually, do not go beyond the boundary of the sentence. \
Note that the texts are refering to WW 1. \
 
To identify a comparison and how to analyse them, orient yourself to these examples.
 
Example 1:
"Der zweite August 1914 war für ihn darum *wie ein Keulenschlag* vor die Stirn."
Analysis:
The phrase *wie ein Keulenschlag vor die Stirn* compares the impact of the event to a blow to the forehead, indicating a sudden and violent shock of the onset of war.
 
Example 2:
"Ihr Pfiff ist *wie ein greller Tropfen Ton*, der in den schwarzen Weiher der Stille fällt."
Analysis:
The whistle ('Pfiff') is compared to a shrill drop of sound falling into a black pond of silence, suggesting the intrusion and disruption akin to war.
 
Example 3:
"In einem der armen, zerschossenen Wälder, in denen die schönsten Eichen nur noch verstümmelte Stümpfe sind, die im Frühjahr zu grünen versuchen, *wie ein todwundes Wild* noch den Kopf hebt oder mit den Läufen zuckt."
Analysis:
The sentence compares the damaged trees in the war-torn forest to a mortally wounded animal, using the phrase 'wie ein todwundes Wild'. This comparison highlights the devastation of war.
 
To identify a metaphor and how to analyse them, orient yourself to these examples.
 
Example 1:
"Der Druck ist *die hassende Seele* des Granatsplitters."
Analysis:
This sentence metaphorically describes the pressure as the 'hating soul' of the shell fragment, attributing human emotions to the forces of war.
 
Example 2:
"Der Krieg hat alle Sentimentalität aus der Welt geschmolzen."
Analysis:
The phrase 'Der Krieg hat alle Sentimentalität aus der Welt geschmolzen' uses the metaphor of war melting away sentimentality, likening the emotional impact of war to a physical process of melting.
 
Example 3:
"Diese Friedenstaube hat keinen Ölzweig im Schnabel."
Analysis:
The 'Friedenstaube' (peace dove) is used metaphorically, and the absence of an 'Ölzweig' (olive branch) suggests the lack of genuine peace offerings, relating to the war context. 
 
To identify a metonymy and how to analyse them, orient yourself to these examples.
 
Example 1:
"Der 42-Zentimeler-Mörser ist *der Bauernschreck Europas*."
Analysis:
The mortar is a metonymy called 'the farmer's terror of Europe,' highlighting its feared destruction and terror during the war.
 
Example 2:
"Meine Eingeweide waren Schützengräben und mein ganzes Innere ein Trommelfeuer."
Analysis:
The sentence uses metonymies to describe the internal turmoil of the narrator by comparing 'Eingeweide' (entrails) to 'Schützengräben' (trenches) and 'mein ganzes Innere' ('my inner self') to 'Trommelfeuer' (drumfire), both of which are elements associated with war.
 
Example 3:
"Seit den ersten Augusttagen 1914 hat das Blut zu strömen und hat das Elend zu fluten nicht aufgehört."
Analysis:
The sentence uses 'das Blut zu strömen' (blood to flow) and 'das Elend zu fluten' (misery to flood) as metanymies to describe the continuous and overwhelming nature of war, likening the effects of war to unstoppable natural forces.
 
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
'''

api_key = '***'  # Replace with your OpenAI API key

client = OpenAI(
    # This is the default and can be omitted
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
    with open('gpt_few_shot_sentence.json', 'w') as f: # changed to gpt_zero.json
        json.dump(json_results, f)

results = run_prompts(file_names, prompt, folder_path)


