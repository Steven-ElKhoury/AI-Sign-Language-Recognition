

import React, { useState, useEffect } from 'react';
import socketIOClient from 'socket.io-client';
import './App.css'; // Import the CSS file for styling
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faComments, faGlobe, faUser } from '@fortawesome/free-solid-svg-icons';

import { BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function App() {
    const [data, setData] = useState(null);
    const [datap, setDatap] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [valueFromIndexHtml, setValueFromIndexHtml] = useState(window.myVariable);
    const modelStatsData = [
        { model: 'KNN', accuracy: 54.46 },
        { model: 'CNN followed by LSTM', accuracy: 92.3 },
        { model: '2* GRU followed by LSTM', accuracy: 85.38 },
        { model: 'LSTM followed by GRU', accuracy: 87.07 },
        { model: 'GRU followed by LSTM', accuracy: 84.15 },
        { model: 'BLSTM', accuracy: 87.69 },
        { model: 'Double GRU', accuracy: 94 },
        { model: 'Triple LSTM', accuracy: 90 }
    ];
    const modelsData = [
        {
            name: 'K-Nearest Neighbors (KNN)',
            description: 'K-Nearest Neighbors (KNN) is a simple and intuitive machine learning algorithm used for classification and regression tasks. It works by finding the \'k\' nearest data points in the training set to a given input data point and predicts the output based on the majority class or average of the \'k\' nearest neighbors.',
        },
        {
            name: 'Convolutional Neural Network (CNN)',
            description: 'Convolutional Neural Networks (CNNs) are deep learning models specifically designed for processing structured grid-like data, such as images. They consist of multiple layers of convolutional and pooling operations, followed by fully connected layers. CNNs are widely used in image classification, object detection, and image segmentation tasks.',
        },
        {
            name: 'Long Short-Term Memory (LSTM)',
            description: 'Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture designed to overcome the vanishing gradient problem in traditional RNNs. LSTMs are capable of learning long-term dependencies and are commonly used in sequence prediction tasks, such as natural language processing, speech recognition, and time series forecasting.',
        },
        {
            name: 'Gated Recurrent Unit (GRU)',
            description: 'Gated Recurrent Unit (GRU) is a type of recurrent neural network that is similar to LSTM but has a simpler structure. It has two gates (reset and update gates) and is capable of learning long-term dependencies. GRUs are used in tasks like natural language processing, speech recognition, and time series forecasting.',
        },
        {
            name: 'Bidirectional Long Short-Term Memory (BLSTM)',
            description: 'Bidirectional Long Short-Term Memory (BLSTM) is a type of recurrent neural network architecture that extends standard LSTM by introducing a second hidden layer of opposite direction. This allows the model to have both past and future context, which is beneficial for understanding context in sequence prediction tasks like natural language processing, speech recognition, and time series forecasting.',
        }
    ];
  const [selectedModel, setSelectedModel] = useState(null);

    const handleModelChange = (event) => {
        const selectedModelName = event.target.value;
        const model = modelsData.find(item => item.name === selectedModelName);
        setSelectedModel(model);
    };


    const handleVariableChange = (event) => {
        setValueFromIndexHtml(event.detail.newValue);
    };

    window.addEventListener('myVariableChanged', handleVariableChange);

    const fetchprediction = () => {
        setLoading(true);
        fetch('http://localhost:5000/api/prediction')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to fetch data');
                }
                return response.json();
            })
            .then(datap => {
                setDatap(datap);
                setLoading(false);
            })
            .catch(error => {
                setError(error);
                setLoading(false);
            });
    };

   
    const fetchData = () => {
        setLoading(true);
        setError(null);

        fetch('http://localhost:5000/api/data')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to fetch data');
                }
                return response.json();
            })
            .then(data => {
                setData(data);
                setLoading(false);
            })
            .catch(error => {
                setError(error);
                setLoading(false);
            });
    };

    return (
        <div>
            <div className="title">
                <h1>Sign Language Recognition</h1>
                <p>Zakhia Tayeh | Jude Soueid | Steven El Khoury | Anthony Fadel</p>
            </div>
            <div className="benefits-chart-container">
                <div className="benefits">
                    <div className="benefit">
                        <FontAwesomeIcon icon={faComments} size="3x" />
                        <h3>Improved Communication</h3>
                        <p>Facilitates communication between deaf individuals and the general public.</p>
                    </div>
                    <div className="benefit">
                        <FontAwesomeIcon icon={faGlobe} size="3x" />
                        <h3>Accessibility</h3>
                        <p>Makes information and services accessible to deaf communities.</p>
                    </div>
                    <div className="benefit">
                        <FontAwesomeIcon icon={faUser} size="3x" />
                        <h3>Empowerment</h3>
                        <p>Empowers deaf individuals to express themselves more freely.</p>
                    </div>
                </div>
                <div className="chart-container">
                    <h2>Model Statistics</h2>
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={modelStatsData} barSize={600} barGap={5}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="model" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="accuracy" fill="#3F51B5" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
            <div className="dropdown-container">
    <label htmlFor="modelSelect">Models Used:</label>
    <select id="modelSelect" onChange={handleModelChange} className="dropdown-button">
        <option value="">Select a Model</option>
        {modelsData.map((model, index) => (
            <option key={index} value={model.name}>{model.name}</option>
        ))}
    </select>
</div>
{selectedModel && (
    <div className="model-section">
        <div className="model-info">
            <div className="model-description">
                <h2>{selectedModel.name}</h2>
                <p>{selectedModel.description}</p>
            </div>
        </div>
    </div>
)}
<br />
<br></br>
<br />
<br></br>
<br />
<br></br>


            {/* {valueFromIndexHtml === 'noblock' &&
                <div>
                    <button className="button predict-button" onClick={fetchprediction}>Predict</button>
                    {loading && <p className="loading-spinner ">  </p>}
                    {error && <p className="error">Error: {error.message}</p>}
                    {datap && (
                        <div className="data-box">
                            <h2>Data Received:</h2>
                            <pre>{JSON.stringify(datap, null, 2)}</pre>
                        </div>
                    )}
                </div>
            } */}
        </div>
    );
}

export default App;
