import React, { useState, useEffect } from 'react';
import { Brain, ThumbsUp, ThumbsDown, Sparkles, Undo, Zap } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

const MindReaderInterface = () => {
  const [gameId, setGameId] = useState(null);
  const [question, setQuestion] = useState(null);
  const [questionType, setQuestionType] = useState(null);
  const [options, setOptions] = useState(null);
  const [guess, setGuess] = useState(null);
  const [history, setHistory] = useState([]);
  const [ws, setWs] = useState(null);
  const [thinking, setThinking] = useState(false);
  const [questionCount, setQuestionCount] = useState(0);

  useEffect(() => {
    const socket = new WebSocket('ws://localhost:8000/ws');
    
    socket.onopen = () => {
      console.log('Connected to Mind Reader AI');
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleGameState(data);
    };

    setWs(socket);

    return () => socket.close();
  }, []);

  const handleGameState = (data) => {
    setGameId(data.game_id);
    setThinking(false);

    if (data.type === 'category' || data.type === 'question') {
      setQuestion(data.question);
      setQuestionType(data.type);
      setOptions(data.options);
      setGuess(null);
      if (data.type === 'question') {
        setQuestionCount(prev => prev + 1);
      }
    } else if (data.type === 'guess') {
      setGuess(data.guess);
      setQuestion(null);
    }
  };

  const sendAnswer = (answer) => {
    if (ws?.readyState === WebSocket.OPEN) {
      setThinking(true);
      setHistory([...history, { question, answer }]);
      
      ws.send(JSON.stringify({
        game_id: gameId,
        question,
        answer
      }));
    }
  };

  const startNewGame = () => {
    setHistory([]);
    setGuess(null);
    setThinking(true);
    setQuestionCount(0);
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        game_id: gameId,
        question: "new_game",
        answer: "yes"
      }));
    }
  };

  return (
    <div className="w-full min-h-screen bg-black text-white p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent flex items-center gap-3">
          <Brain className="h-8 w-8" />
          Mind Reader AI
        </h1>
        <div className="flex gap-4">
          <Alert variant="default" className="bg-purple-900/30 border-purple-500">
            <Zap className="h-5 w-5 text-purple-500" />
            <AlertDescription className="ml-2">
              Questions: {questionCount}/20
            </AlertDescription>
          </Alert>
          <Alert variant="default" className="bg-purple-900/30 border-purple-500">
            <Sparkles className="h-5 w-5 text-purple-500" />
            <AlertDescription className="ml-2">
              {thinking ? "Thinking..." : "Ready"}
            </AlertDescription>
          </Alert>
        </div>
      </div>

      {/* Main Game Area */}
      <div className="max-w-2xl mx-auto">
        {/* Question Area */}
        {question && (
          <div className="bg-purple-900/10 rounded-lg p-6 border border-purple-500/30 mb-6 text-center">
            <h2 className="text-xl mb-4">{question}</h2>
            
            {questionType === 'category' ? (
              <div className="grid grid-cols-2 gap-4">
                {options.map((option) => (
                  <button
                    key={option}
                    onClick={() => sendAnswer(option)}
                    className="bg-purple-500 hover:bg-purple-600 text-white py-3 px-6 rounded transition-colors"
                  >
                    {option.charAt(0).toUpperCase() + option.slice(1)}
                  </button>
                ))}
              </div>
            ) : (
              <div className="flex justify-center gap-4">
                <button
                  onClick={() => sendAnswer('yes')}
                  className="bg-green-500 hover:bg-green-600 text-white py-3 px-8 rounded transition-colors flex items-center gap-2"
                >
                  <ThumbsUp className="h-5 w-5" />
                  Yes
                </button>
                <button
                  onClick={() => sendAnswer('no')}
                  className="bg-red-500 hover:bg-red-600 text-white py-3 px-8 rounded transition-colors flex items-center gap-2"
                >
                  <ThumbsDown className="h-5 w-5" />
                  No
                </button>
              </div>
            )}
          </div>
        )}

        {/* Guess Area */}
        {guess && (
          <div className="bg-purple-900/10 rounded-lg p-6 border border-purple-500/30 mb-6 text-center">
            <h2 className="text-2xl mb-4">I think it's...</h2>
            <div className="text-3xl font-bold text-purple-400 mb-6">
              {guess}
            </div>
            <button
              onClick={startNewGame}
              className="bg-purple-500 hover:bg-purple-600 text-white py-3 px-8 rounded transition-colors flex items-center gap-2 mx-auto"
            >
              <Undo className="h-5 w-5" />
              Play Again
            </button>
          </div>
        )}

        {/* History */}
        <div className="bg-purple-900/10 rounded-lg p-6 border border-purple-500/30">
          <h3 className="text-lg font-semibold mb-4">Question History</h3>
          <div className="space-y-2">
            {history.map((item, index) => (
              <div key={index} className="flex justify-between items-center text-sm">
                <span className="text-purple-300">{item.question}</span>
                <span className={item.answer === 'yes' ? 'text-green-400' : 'text-red-400'}>
                  {item.answer}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MindReaderInterface;