import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Auth from './components/Auth';
import ModeSelection from './components/ModeSelection';
import GeneralQuery from './components/GeneralQuery';
import MedicalReport from './components/MedicalReport';
import LandingPage from './components/LandingPage';

function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || '');
  const [loading, setLoading] = useState(true);
  
  // Check if user is authenticated
  const isAuthenticated = !!token;
  
  // Simulate authentication check and initial loading
  useEffect(() => {
    // In a real app, you might validate the token with your backend here
    const checkAuth = setTimeout(() => {
      setLoading(false);
    }, 1000);
    
    return () => clearTimeout(checkAuth);
  }, [token]);
  
  // Save token to localStorage when it changes
  const handleLogin = (newToken, username) => {  // Add username parameter
    localStorage.setItem('token', newToken);
    setToken(newToken);
    
    // Explicitly save username again to ensure it's set properly
    if (username) {
      localStorage.setItem('username', username);
      console.log("App - Username set in localStorage:", username);
    }
    
    // Verify username is correctly stored
    const storedUser = localStorage.getItem('username');
    console.log("App - Verifying username in localStorage:", storedUser);
  };
  
  // Remove token on logout
  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    setToken('');
  };

  // Show loading state while checking authentication
  if (loading) {
    return <div className="app-loading">Loading...</div>;
  }

  return (
    <Router>
      <Routes>
        {/* Landing page is the default route */}
        <Route path="/" element={<LandingPage isAuthenticated={isAuthenticated} />} />
        
        {/* Authentication page */}
        <Route path="/login" element={!isAuthenticated ? <Auth onLogin={handleLogin} /> : <Navigate to="/mode-selection" />} />
        
        {/* App routes - protected */}
        <Route path="/mode-selection" element={isAuthenticated ? <ModeSelection onLogout={handleLogout} /> : <Navigate to="/login" />} />
        <Route path="/general-query" element={isAuthenticated ? <GeneralQuery token={token} onLogout={handleLogout} /> : <Navigate to="/login" />} />
        <Route path="/medical-report" element={isAuthenticated ? <MedicalReport token={token} onLogout={handleLogout} /> : <Navigate to="/login" />} />
        
        {/* Fallback route for any unmatched paths */}
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </Router>
  );
}

export default App;
