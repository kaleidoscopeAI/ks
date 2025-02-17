from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import json
import asyncio
import random
import logging
from datetime import datetime
import boto3
from boto3.dynamodb.conditions import Key
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, start_http_server

# Monitoring
QUESTIONS_ASKED = Counter('questions_asked_total', 'Total questions asked')
CORRECT_GUESSES = Counter('correct_guesses_total', 'Total correct guesses')
GUESS_TIME = Histogram('guess_time_seconds', 'Time taken to make a guess')

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameDatabase:
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.table = self.dynamodb.Table('mindreader_games')
        self.patterns_table = self.dynamodb.Table('mindreader_patterns')
        
    async def save_game(self, game_id: str, data: Dict):
        try:
            self.table.put_item(Item={
                'game_id': game_id,
                'timestamp': datetime.now().isoformat(),
                'game_data': json.dumps(data)
            })
        except Exception as e:
            logger.error(f"Failed to save game: {e}")
            
    async def load_patterns(self) -> Dict:
        try:
            response = self.patterns_table.scan()
            return {item['object']: item['patterns'] for item in response['Items']}
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            return self._get_default_patterns()
            
    def _get_default_patterns(self) -> Dict:
        return {
            "smartphone": {"electronic": 1, "expensive": 0.8, "modern": 1},
            "dog": {"animal": 1, "alive": 1, "friendly": 0.8},
            "car": {"vehicle": 1, "expensive": 1, "modern": 1},
            # Add more default patterns...
        }

class QuestionEngine:
    def __init__(self, patterns: Dict):
        self.patterns = patterns
        self.questions = self._generate_questions()
        
    def _generate_questions(self) -> List[Dict]:
        questions = []
        # Generate questions based on pattern attributes
        for attribute in set().union(*[p.keys() for p in self.patterns.values()]):
            questions.append({
                "text": f"Is it {attribute}?",
                "attribute": attribute,
                "weight": 1.0
            })
        return questions
        
    def get_next_question(self, answers: Dict) -> Optional[Dict]:
        if not self.questions:
            return None
            
        # Calculate information gain for each remaining question
        gains = []
        for q in self.questions:
            if q['attribute'] not in answers:
                gain = self._calculate_information_gain(q['attribute'], answers)
                gains.append((gain, q))
                
        if not gains:
            return None
            
        # Return question with highest information gain
        gains.sort(reverse=True)
        return gains[0][1]
        
    def _calculate_information_gain(self, attribute: str, answers: Dict) -> float:
        # Calculate entropy before and after this question
        current_entropy = self._calculate_entropy(answers)
        
        # Simulate both possible answers
        yes_answers = answers.copy()
        yes_answers[attribute] = 1
        no_answers = answers.copy()
        no_answers[attribute] = 0
        
        # Calculate conditional entropy
        entropy_after = (
            self._calculate_entropy(yes_answers) + 
            self._calculate_entropy(no_answers)
        ) / 2
        
        return current_entropy - entropy_after
        
    def _calculate_entropy(self, answers: Dict) -> float:
        # Calculate probability distribution over remaining objects
        probs = []
        for obj, pattern in self.patterns.items():
            match_score = 1.0
            for attr, value in answers.items():
                if attr in pattern:
                    match_score *= 1 - abs(pattern[attr] - value)
            if match_score > 0:
                probs.append(match_score)
                
        probs = np.array(probs)
        if len(probs) == 0:
            return 0
        probs = probs / probs.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))

class MindReaderGame:
    def __init__(self, patterns: Dict):
        self.engine = QuestionEngine(patterns)
        self.answers = {}
        self.start_time = datetime.now()
        
    async def process_answer(self, answer: str) -> Dict:
        question = self.engine.get_next_question(self.answers)
        if not question:
            return await self.make_guess()
            
        self.answers[question['attribute']] = 1 if answer == 'yes' else 0
        QUESTIONS_ASKED.inc()
        
        # Check if we have enough confidence to guess
        confidence = self.calculate_confidence()
        if confidence > 0.85 or len(self.answers) >= 15:
            return await self.make_guess()
            
        return {
            'type': 'question',
            'question': question['text'],
            'confidence': confidence * 100
        }
        
    async def make_guess(self) -> Dict:
        guess_time = (datetime.now() - self.start_time).total_seconds()
        GUESS_TIME.observe(guess_time)
        
        best_match = None
        best_score = 0
        
        for obj, pattern in self.engine.patterns.items():
            score = 1.0
            for attr, value in self.answers.items():
                if attr in pattern:
                    score *= 1 - abs(pattern[attr] - value)
            if score > best_score:
                best_score = score
                best_match = obj
                
        CORRECT_GUESSES.inc()  # Assuming the guess is correct
        
        return {
            'type': 'guess',
            'guess': best_match.title() if best_match else "Something I don't know yet!",
            'confidence': best_score * 100
        }
        
    def calculate_confidence(self) -> float:
        if not self.answers:
            return 0
            
        confidences = []
        for pattern in self.engine.patterns.values():
            match_score = 1.0
            for attr, value in self.answers.items():
                if attr in pattern:
                    match_score *= 1 - abs(pattern[attr] - value)
            confidences.append(match_score)
            
        return max(confidences)

# FastAPI Setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start Prometheus metrics server
start_http_server(8000)

# Initialize database
db = GameDatabase()
games = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Load patterns from database
    patterns = await db.load_patterns()
    
    # Create new game
    game_id = str(random.randint(1000, 9999))
    games[game_id] = MindReaderGame(patterns)
    
    try:
        # Send initial question
        question = games[game_id].engine.get_next_question({})
        await websocket.send_json({
            'type': 'question',
            'question': question['text'],
            'confidence': 0
        })
        
        # Game loop
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'start':
                # Reset game
                games[game_id] = MindReaderGame(patterns)
                question = games[game_id].engine.get_next_question({})
                await websocket.send_json({
                    'type': 'question',
                    'question': question['text'],
                    'confidence': 0
                })
            else:
                # Process answer and send response
                response = await games[game_id].process_answer(data['answer'])
                await websocket.send_json(response)
                
                # Save game state if it's a guess
                if response['type'] == 'guess':
                    await db.save_game(game_id, {
                        'answers': games[game_id].answers,
                        'guess': response['guess'],
                        'confidence': response['confidence']
                    })
                    
    except Exception as e:
        logger.error(f"Game error: {e}")
    finally:
        if game_id in games:
            del games[game_id]

@app.get("/stats")
async def get_stats():
    """Get game statistics"""
    try:
        # Get metrics from Prometheus
        stats = {
            'total_games': QUESTIONS_ASKED._value.get(),
            'correct_guesses': CORRECT_GUESSES._value.get(),
            'average_guess_time': GUESS_TIME.describe()['mean']
        }
        return stats
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)