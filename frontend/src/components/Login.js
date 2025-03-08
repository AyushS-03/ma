import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Login.css';

function Login({ onLogin }) {
  const navigate = useNavigate();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    try {
      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await axios.post(`${apiUrl}/api/login`, {
        username: username,
        password: password
      });
      
      // Store username in localStorage more securely
      localStorage.removeItem('username'); // Clear any existing value first
      localStorage.setItem('username', username);
      console.log("Login successful - Username stored in localStorage:", username);
      
      // Force a window reload to ensure all components get the updated username
      const token = response.data.token;
      onLogin(token, username); // Pass username to parent component
      
      // Brief delay before navigation to ensure storage is complete
      setTimeout(() => {
        navigate('/mode-selection');
      }, 100);
    } catch (error) {
      setError('Invalid username or password');
    }
  };

  return (
    <div className="login-container">
      <h1>Login</h1>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="username">Username</label>
          <input 
            type="text" 
            id="username" 
            value={username} 
            onChange={(e) => setUsername(e.target.value)} 
            required 
          />
        </div>
        <div className="form-group">
          <label htmlFor="password">Password</label>
          <input 
            type="password" 
            id="password" 
            value={password} 
            onChange={(e) => setPassword(e.target.value)} 
            required 
          />
        </div>
        {error && <div className="error-message">{error}</div>}
        <button type="submit" className="login-button">Login</button>
      </form>
    </div>
  );
}

export default Login;