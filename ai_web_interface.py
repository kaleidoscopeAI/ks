<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Driven Discovery - Kaleidoscope Integration</title>
    <script>
        async function analyzeMolecule() {
            const smiles = document.getElementById("smiles").value;
            const useCase = document.getElementById("use-case").value;
            const response = await fetch(`http://localhost:8000/analyze?smiles=${encodeURIComponent(smiles)}&use_case=${encodeURIComponent(useCase)}`);
            const data = await response.json();
            document.getElementById("results").innerText = JSON.stringify(data, null, 2);
        }

        async function chatWithJacob() {
            const message = document.getElementById("chat-input").value;
            const response = await fetch("http://localhost:8000/chatbot", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({message: message})
            });
            const data = await response.json();
            document.getElementById("chat-output").innerText += "\nUser: " + message + "\nJacob: " + data.response;
        }
    </script>
</head>
<body>
    <h1>AI-Driven Molecular & Health Analysis - Kaleidoscope System</h1>
    <label for="smiles">Enter Molecular SMILES Code:</label>
    <input type="text" id="smiles" placeholder="CC(=O)OC1=CC=CC=C1C(=O)O">
    <br><br>
    <label for="use-case">Select Use Case:</label>
    <select id="use-case">
        <option value="drug_discovery">Drug Discovery</option>
        <option value="cosmetic_chemistry">Cosmetic Chemistry</option>
        <option value="nutraceuticals">Nutraceuticals</option>
        <option value="agriculture">Agriculture</option>
        <option value="personalized_health">Personalized Health</option>
        <option value="nutrition">Nutrition</option>
        <option value="mental_health">Mental Health</option>
        <option value="environmental_safety">Environmental Safety</option>
        <option value="disease_tracking">Disease Tracking</option>
        <option value="kids_learning_ai">Kids Learning AI</option>
    </select>
    <br><br>
    <button onclick="analyzeMolecule()">Analyze</button>
    <h2>Results:</h2>
    <pre id="results"></pre>
    
    <h1>Chat with Jacob</h1>
    <label for="chat-input">Ask Jacob a Question:</label>
    <input type="text" id="chat-input" placeholder="Type your question here...">
    <button onclick="chatWithJacob()">Chat</button>
    <pre id="chat-output"></pre>
</body>
</html>

