import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

function Auth({ onLogin }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [success, setSuccess] = useState('');
  const [isFlipping, setIsFlipping] = useState(false);
  const navigate = useNavigate();

  // Clear error when mode changes
  useEffect(() => {
    setError('');
    setSuccess('');
  }, [isLogin]);

  const validateForm = () => {
    if (!username.trim()) {
      setError('Username is required');
      return false;
    }
    
    if (!password.trim()) {
      setError('Password is required');
      return false;
    }

    if (!isLogin && password !== confirmPassword) {
      setError('Passwords do not match');
      return false;
    }

    if (!isLogin && password.length < 6) {
      setError('Password must be at least 6 characters long');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    
    if (!validateForm()) return;
    
    setIsLoading(true);
    
    try {
      const endpoint = isLogin ? '/api/login' : '/api/register';
      console.log(`Sending ${isLogin ? 'login' : 'register'} request to ${endpoint}`);
      
      const response = await axios.post(`http://localhost:5000${endpoint}`, {
        username,
        password
      });
      
      console.log('Response:', response.data);
      
      if (isLogin) {
        // Check if token exists in the response
        if (response.data.token) {
          console.log('Login successful, token received:', response.data.token);
          setSuccess('Login successful! Redirecting...');
          setTimeout(() => {
            onLogin(response.data.token);
          }, 1000);
        } else {
          setError('Server response missing authentication token');
        }
      } else {
        // Show success message and switch to login with animation
        setSuccess('Registration successful! You can now log in.');
        setTimeout(() => {
          toggleMode();
        }, 1500);
      }
    } catch (error) {
      console.error('Auth error:', error);
      
      if (error.response) {
        // The server responded with an error
        setError(error.response.data?.error || `Error (${error.response.status}): ${error.response.statusText}`);
      } else if (error.request) {
        // No response from the server
        setError('No response from server. Please check if the backend is running.');
      } else {
        // Request setup error
        setError(`Request error: ${error.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const toggleMode = () => {
    setIsFlipping(true);
    setTimeout(() => {
      setIsLogin(!isLogin);
      setIsFlipping(false);
    }, 400); // Half of the transition time
  };

  // For quick testing, add a demo login option
  const handleDemoLogin = () => {
    setUsername('test');
    setPassword('test123');
    if (!isLogin) {
      toggleMode();
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container-wrapper">
        <div className={`auth-card ${isFlipping ? 'flipping' : ''}`}>
          <div className="auth-header">
            <h1 className="auth-title">MedAssist AI</h1>
            <div className="auth-tabs">
              <button 
                className={`auth-tab ${isLogin ? 'active' : ''}`}
                onClick={() => !isLogin && toggleMode()}
                disabled={isLoading}
              >
                Login
              </button>
              <button 
                className={`auth-tab ${!isLogin ? 'active' : ''}`}
                onClick={() => isLogin && toggleMode()}
                disabled={isLoading}
              >
                Register
              </button>
            </div>
          </div>
          
          <div className="auth-body">
            <h2 className="form-title">{isLogin ? 'Welcome Back' : 'Create Account'}</h2>
            
            {error && (
              <div className="auth-message error">
                <div className="message-icon">⚠️</div>
                <div className="message-text">{error}</div>
              </div>
            )}
            
            {success && (
              <div className="auth-message success">
                <div className="message-icon">✅</div>
                <div className="message-text">{success}</div>
              </div>
            )}
            
            <form onSubmit={handleSubmit} className="auth-form">
              <div className="form-group">
                <label htmlFor="username">Username</label>
                <div className="input-wrapper">
                  <span className="input-icon">👤</span>
                  <input
                    id="username"
                    type="text"
                    placeholder="Enter your username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    required
                    disabled={isLoading || !!success}
                    autoComplete="username"
                  />
                </div>
              </div>
              
              <div className="form-group">
                <label htmlFor="password">Password</label>
                <div className="input-wrapper">
                  <span className="input-icon">🔒</span>
                  <input
                    id="password"
                    type="password"
                    placeholder={isLogin ? "Enter your password" : "Create a password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    disabled={isLoading || !!success}
                    autoComplete={isLogin ? "current-password" : "new-password"}
                  />
                </div>
              </div>
              
              {!isLogin && (
                <div className="form-group">
                  <label htmlFor="confirmPassword">Confirm Password</label>
                  <div className="input-wrapper">
                    <span className="input-icon">🔒</span>
                    <input
                      id="confirmPassword"
                      type="password"
                      placeholder="Confirm your password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      required
                      disabled={isLoading || !!success}
                      autoComplete="new-password"
                    />
                  </div>
                </div>
              )}
              
              <button 
                type="submit" 
                className="auth-button primary" 
                disabled={isLoading || !!success}
              >
                {isLoading ? (
                  <span className="button-loader"></span>
                ) : isLogin ? 'Login' : 'Create Account'}
              </button>
            </form>
            
            <div className="auth-divider">
              <span>or</span>
            </div>
            
            <div className="auth-actions">
              <button 
                className="auth-button demo" 
                onClick={handleDemoLogin}
                disabled={isLoading || !!success}
              >
                Try Demo Account
              </button>
              
              <button 
                className="auth-link"
                onClick={() => navigate('/')}
                disabled={isLoading}
              >
                Back to Home
              </button>
            </div>
          </div>
        </div>
        
        <div className="auth-footer">
          <p>© {new Date().getFullYear()} MedAssist AI. All rights reserved.</p>
          <p>Your health data is secure and private.</p>
        </div>
      </div>
    </div>
  );
}

export default Auth;
