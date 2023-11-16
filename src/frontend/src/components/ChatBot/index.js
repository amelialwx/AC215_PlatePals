
//  const apiKey = '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'; // Replace with your actual OpenAI API key
//  const headers = {
//    'Authorization': `Bearer ${apiKey}`,

import React, { useState } from 'react';
import axios from 'axios';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';
const apiKey = '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'

function ChatBot() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');

    const sendMessage = async () => {
        if (input.trim()) {
            const newMessage = { role: 'user', content: input };
            setMessages([...messages, newMessage]);
            setInput('');
            const response = await fetchResponse(messages, newMessage);
            setMessages([...messages, newMessage, response]);
        }
    };

    const fetchResponse = async (conversation, newMessage) => {
        const payload = {
            messages: [...conversation, newMessage],
            model: 'gpt-3.5-turbo',
        };
        const headers = {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`,
        };
        try {
            const response = await axios.post('https://api.openai.com/v1/chat/completions', payload, { headers });
            return { role: 'assistant', content: response.data.choices[0].message.content };
        } catch (error) {
            console.error('Error fetching response:', error);
        }
    };

    const handleInputChange = (event) => {
        setInput(event.target.value);
    };

    const handleKeyPress = (event) => {
        if (event.key === 'Enter') {
            sendMessage();
        }
    };

    return (
//        <div>
//            <div>
//                {messages.map((msg, index) => (
//                    <p key={index}><strong>{msg.role}:</strong> {msg.content}</p>
//                ))}
//            </div>
//            <input
//                type="text"
//                value={input}
//                onChange={handleInputChange}
//                onKeyPress={handleKeyPress}
//            />
//            <button onClick={sendMessage}>Send</button>
//        </div>

        <div>
            <main>
                <Container maxWidth={false}>
                    <h1>&#129302; PlatePals AI ChatBot</h1>
                    <div>
                        {messages.map((msg, index) => (
                            <p key={index}><strong>{msg.role == 'user' ? 'You' : 'PlatePals AI Assistant'}:</strong> {msg.content}</p>
                        ))}
                    </div>
                    <input
                        type="text" 
                        size="50"
                        value={input}
                        onChange={handleInputChange}
                        onKeyPress={handleKeyPress}
                    />
                    <button onClick={sendMessage}>Send</button>

                </Container>
            </main>
        </div>
    );
}

export default ChatBot;
