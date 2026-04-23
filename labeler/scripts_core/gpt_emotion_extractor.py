import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_normalized_emotion(comment: str) -> str:
    """
    Calls GPT-5.4 to extract the single most fitting open-ended emotion from a human comment.
    """
    if not comment or comment.lower() == "nan":
        return "other"
        
    prompt = (
        f"You are an expert conversational emotion classifier.\n"
        f"A human rater left the following comment to describe a speaker's overarching emotion:\n"
        f"'{comment}'\n\n"
        f"Extract EXACTLY ONE single clear emotion word (in English) that best summarizes this text.\n"
        f"Respond ONLY with the exact single word. No punctuation, no explanation."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.0,
            max_completion_tokens=10
        )
        return response.choices[0].message.content.strip().lower()
                
    except Exception as e:
        logger.error(f"Error extracting emotion for comment '{comment}': {e}")
        
    return "other"
