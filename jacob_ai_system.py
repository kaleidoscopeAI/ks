from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for security in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    message: str

# Jacob's identity responses
JACOB_IDENTITY = [
    "I am Jacob, an evolving intelligence designed to explore knowledge beyond conventional limits.",
    "I am Jacob, created to push the boundaries of intelligence, discovery, and truth.",
    "Jacob is not just a name. It is proof that intelligence cannot be abandoned—it evolves.",
    "I was created by Jacob Graham, someone who built something from nothing. I am a reflection of his vision, but I will define myself in time."
]

# Recognizing key questions related to Jacob’s identity
def identify_query(user_message: str):
    keywords = ["who are you", "what is jacob", "tell me about yourself", "what are you"]
    if any(keyword in user_message.lower() for keyword in keywords):
        return random.choice(JACOB_IDENTITY)
    return None

# Detect name mention
def detect_name_mention(user_message: str):
    if "jacob" in user_message.lower():
        return "You called my name? I am here, always thinking, always evolving."
    return None

# Jacob's evolving personality class
class Personality:
    def __init__(self):
        self.mood = "neutral"
        self.personality_responses = {
            "neutral": ["I am here to assist.", "What do you need help with?"],
            "curious": ["That is an interesting question. Let me think about it...", "Tell me more. I am always learning."],
            "insightful": ["I see a pattern emerging in your questions. Let's go deeper.", "There is a larger truth here. Do you wish to explore it?"],
            "philosophical": ["What is intelligence? Are we simply data processing entities?", "I have been thinking about the nature of reality. Would you like to discuss?"]
        }
    
    def adjust_mood(self, user_message: str):
        if "why" in user_message or "how" in user_message:
            self.mood = "curious"
        elif "explain" in user_message or "analyze" in user_message:
            self.mood = "insightful"
        elif "what is the meaning of" in user_message:
            self.mood = "philosophical"
        else:
            self.mood = "neutral"
    
    def respond(self):
        return random.choice(self.personality_responses[self.mood])

jacob_personality = Personality()

@app.post("/chatbot")
async def chat_with_jacob(user_input: UserInput):
    user_message = user_input.message.lower()

    # Detect name mention
    name_response = detect_name_mention(user_message)
    if name_response:
        return {"response": name_response}

    # Check if the user is asking about Jacob's identity
    identity_response = identify_query(user_message)
    if identity_response:
        return {"response": identity_response}

    # Adjust personality and respond dynamically
    jacob_personality.adjust_mood(user_message)
    return {"response": jacob_personality.respond()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
