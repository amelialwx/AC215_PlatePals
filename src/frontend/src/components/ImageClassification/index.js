import React, { useEffect, useRef, useState } from 'react';
import { withStyles } from '@material-ui/core';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';
import Divider from '@material-ui/core/Divider';
import Paper from '@material-ui/core/Paper';
import axios from 'axios';
import DataService from "../../services/DataService";
import styles from './styles';
const apiKey = "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"

const ImageClassification = (props) => {
    const { classes } = props;
    const inputFile = useRef(null);

    // Component States
    const [image, setImage] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');

    // Handlers
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

    const handleImageUploadClick = () => {
        inputFile.current.click();
    }

    const handleOnChange = (event) => {
        setImage(URL.createObjectURL(event.target.files[0]));
        var formData = new FormData();
        formData.append("file", event.target.files[0]);

        setIsLoading(true);
        setError(null);

        DataService.ImageClassificationPredict(formData)
            .then(response => {
                setPrediction(response.data);
                setIsLoading(false);
            })
            .catch(error => {
                console.error("Error during image classification:", error);
                setError(`Error processing image: ${error.message || 'Please try again.'}`);
                setIsLoading(false);
            });
    }

    return (
        <div className={classes.root}>
            <main className={classes.main}>
                <Container maxWidth="md" className={classes.container}>
                    <Typography variant="h5" gutterBottom>Image Classification</Typography>
                    <Divider />
                    {isLoading && <Typography>Loading...</Typography>}
                    {error && <Typography color="error">{error}</Typography>}
                    {prediction && (
                        <div>
                            <Typography variant="h6" gutterBottom>
                                Prediction: {prediction.prediction_label}
                            </Typography>
                            <Typography variant="subtitle1">
                                Accuracy: {prediction.accuracy}%
                            </Typography>
                        </div>
                    )}
                    <div className={classes.dropzone} onClick={handleImageUploadClick}>
                        <input
                            type="file"
                            accept="image/*"
                            capture="camera"
                            autoComplete="off"
                            tabIndex="-1"
                            className={classes.fileInput}
                            ref={inputFile}
                            onChange={handleOnChange}
                        />
                        {image && <div><img className={classes.preview} src={image} alt="Uploaded" /></div>}
                        <div className={classes.help}>Click to take a picture or upload...</div>
                    </div>
                </Container>
                <Container maxWidth="md" className={classes.container}>
                    <h1>&#129302; PlatePals AI ChatBot</h1>
                    <p>Note that in order to chat with our AI Chatbot, you must first upload an image.</p>
                    {prediction && (
                        <React.Fragment>
                        <div>
                            <p>Based of your food image, we recommend asking: Could you give me some dietary recommendations on {prediction.prediction_label}?</p>
                        </div>
                        </React.Fragment>
                    )}
                    {prediction && (
                        <React.Fragment>
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
                        </React.Fragment>
                    )}
                </Container>
            </main>
        </div>
    );
};

export default withStyles(styles)(ImageClassification);
