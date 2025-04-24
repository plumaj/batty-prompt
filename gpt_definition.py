import time
from openai import OpenAI, RateLimitError, OpenAIError
import pandas as pd
import json
import os

# Define the folder containing the text files
folder_path = "../txt_files"

# Get a list of all the text files in the folder with the prefix '1916_03_'
file_names = [f for f in os.listdir(folder_path)] #if f.startswith("1915-03-0")]

# prompt with definitions and examples for metaphors, metonymies and comparisons related to war
prompt = """Read the text passage below after these instructions. \
Search for all sentences containing examples for the figures of speech "metaphors", "comparisons" or "metonymies" related to the descriptions of war in the text below. \
Consider only each sentence individually, do not go beyond the boundary of the sentence. \
Note that the text is a feuilleton text from a newspaper from the beginning of the 20th century. References to war mean  WW 1. \
To help you with this task, I am providing also definitions and examples for the figures of speech you are going to identify. \
Take also the following definitions into account! \
 
Definition of a "metaphor":
Eine Metapher ist eine Stilfigur, ein 'sprachliches Bild', das auf einer Ähnlichkeitsbeziehung zwischen zwei Gegenständen bzw. Begriffen beruht, d.h. auf Grund gleicher oder ähnlicher Bedeutungsmerkmale \
findet eine Bezeichungsübertragung statt (z.B. 'der Himmel weint' für 'es regnet'). Häufig wird eine Metapher auch als verkürzter Vergleich beschreiben, wobei der Vergleich als solcher jedoch nicht \
ausgedrückt wird. Metaphern können in substantivischer, adjektivischer und verbaler Form im Satzkontext auftreten (z.B. 'Fuchsschwanz' für 'Handsäge', 'spitze Bemerkung' für 'verletzende Bemerkung', 'sich zügeln' für 'sich zurückhalten'). \
 
Definition of a "metonymy":
Eine Metonymie ist eine Stilfigur, bei der Wort oder ein Konzept durch eine verwandte Bezeichnung ersetzt wird, die mit dem Gemeinten - im Gegensatz zur Metapher - durch einen sachlichen (z.B. räumlichen, zeitlichen, kausalen, instrumentalen) \
Zusammenhang bzw. durch semantische Kontiguität verknüpft ist. Beispiele: 'Seide tragen' für 'Kleidung aus Seide tragen', 'Goethe lesen' für 'ein Werk von Goethe lesen', 'das Weiße Haus schweigt' für 'die Regierung der USA schweigt', \
'Das Zepter niederlegen' für 'die Regierungsgewalt abgeben'. Eine Spezialform ist die 'Synekdoche'.
 
Definition of a "comparison":
Ein Vergleich ist ein Stilmittel, bei dem mindestens zwei unterschiedliche Dinge, Personen oder Situationen direkt gegenübergestellt werden, um eine Ähnlichkeit zu betonen. Oft wird die Vergleichspartikel 'wie' dafür verwendet. \
Der Vergleich wird also auch grammatisch direkt ausgedrückt.

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
    with open('gpt_definition_sentence.json', 'w') as f:
        json.dump(json_results, f)

results = run_prompts(file_names, prompt, folder_path)


