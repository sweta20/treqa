from google import genai
import os


assert (
    "GEMINI_API_KEY" in os.environ
), "Please set the GEMINI_API_KEY environment variable"

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def prompt_genai(system_prompt, user_prompt):
    config = genai.types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=32768,
        top_p=0.7,
        top_k=100,
        system_instruction=system_prompt,
        response_mime_type="text/plain",
        safety_settings=[
            genai.types.SafetySetting(
                category=category, threshold=genai.types.HarmBlockThreshold.BLOCK_NONE
            )
            for category in [
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            ]
        ],
    )

    while(True):
        response = client.models.generate_content(
                model="gemini-2.5-flash", contents=user_prompt, config=config
            )
        if response.candidates is None:
            continue

        if response.candidates[0].finish_reason == "MAX_TOKENS":
            continue
        
        return response.text
    

