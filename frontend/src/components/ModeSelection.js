import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

function ModeSelection({ onLogout }) {
  const navigate = useNavigate();
  const [userName, setUserName] = useState('User');
  const [selectedMode, setSelectedMode] = useState(null);
  const [isWelcomeVisible, setIsWelcomeVisible] = useState(false);
  const [areOptionsVisible, setAreOptionsVisible] = useState(false);
  const [hoveredCard, setHoveredCard] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Load username from localStorage on mount - more robust implementation
  useEffect(() => {
    // Force a fresh read from localStorage on each mount
    const storedUser = localStorage.getItem('username');
    console.log("ModeSelection - Loading username from localStorage:", storedUser);
    
    if (storedUser && storedUser !== 'undefined' && storedUser !== 'null') {
      setUserName(storedUser);
    } else {
      console.warn("ModeSelection - Valid username not found in localStorage, using default");
    }
    
    // Animate welcome message and options
    setTimeout(() => setIsWelcomeVisible(true), 300);
    setTimeout(() => setAreOptionsVisible(true), 800);
  }, []);

  const handleModeSelect = (mode) => {
    setSelectedMode(mode);
    setIsLoading(true);
    
    // Navigate to the selected mode after a short delay
    setTimeout(() => {
      navigate(`/${mode}`);
    }, 800);
  };

  const modes = [
    {
      id: 'general-query',
      title: 'General Health Query',
      description: 'Chat with our AI assistant about any health concerns, symptoms, or medical questions you may have.',
      icon: '🩺',
      color: '#4285f4',
      gradient: 'linear-gradient(135deg, #4285f4, #34a853)'
    },
    {
      id: 'medical-report',
      title: 'Medical Report Analysis',
      description: 'Upload your medical reports and get AI-powered analysis and explanation in simple terms.',
      icon: '📄',
      color: '#ea4335',
      gradient: 'linear-gradient(135deg, #ea4335, #fbbc05)'
    }
  ];

  // Fixed UserProfile component with simplified menu
  const UserProfile = ({ userName, onLogout }) => {
    const [isOpen, setIsOpen] = useState(false);
    
    const toggleDropdown = () => {
      setIsOpen(!isOpen);
    };
    
    // Close dropdown when clicking outside
    useEffect(() => {
      const handleClickOutside = (event) => {
        if (isOpen && !event.target.closest('.user-profile-container')) {
          setIsOpen(false);
        }
      };
      
      document.addEventListener('mousedown', handleClickOutside);
      return () => {
        document.removeEventListener('mousedown', handleClickOutside);
      };
    }, [isOpen]);
    
    // Get user initials for avatar
    const userInitials = userName
      .split(' ')
      .map(name => name[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
    
    return (
      <div className="user-profile-container">
        <button 
          className="user-profile-button" 
          onClick={toggleDropdown}
          aria-expanded={isOpen}
          aria-haspopup="true"
        >
          <div className="user-avatar">{userInitials}</div>
          <span className="user-name">{userName}</span>
          <span className="dropdown-arrow">{isOpen ? '▲' : '▼'}</span>
        </button>
        
        {isOpen && (
          <div className="user-dropdown">
            <div className="dropdown-header">
              <div className="user-avatar large">{userInitials}</div>
              <div className="user-info-detail">
                <span className="full-name">{userName}</span>
                <span className="account-type">Basic Account</span>
              </div>
            </div>
            
            <div className="dropdown-menu">
              <button className="menu-item logout" onClick={onLogout}>
                <span className="menu-icon">🚪</span>
                Logout
              </button>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="mode-selection-page">
      <header className="mode-header">
        <div className="mode-logo">
          <span className="logo-icon">🧠</span>
          <h1>MedAssist AI</h1>
        </div>
        <div className="user-profile">
          <UserProfile userName={userName} onLogout={onLogout} />
        </div>
      </header>
      
      <main className="mode-content">
        <div className={`welcome-message ${isWelcomeVisible ? 'visible' : ''}`}>
          <h2>Welcome, {userName}!</h2>
          <p>How would you like MedAssist AI to help you today?</p>
        </div>
        
        <div className={`mode-options ${areOptionsVisible ? 'visible' : ''}`}>
          {modes.map((mode) => (
            <div 
              key={mode.id}
              className={`mode-card ${hoveredCard === mode.id ? 'hovered' : ''} ${selectedMode === mode.id ? 'selected' : ''}`}
              onClick={() => handleModeSelect(mode.id)}
              onMouseEnter={() => setHoveredCard(mode.id)}
              onMouseLeave={() => setHoveredCard(null)}
              style={{
                '--mode-color': mode.color,
                '--mode-gradient': mode.gradient
              }}
            >
              <div className="mode-icon" style={{ backgroundImage: mode.gradient }}>
                {mode.icon}
              </div>
              <div className="mode-info">
                <h3>{mode.title}</h3>
                <p>{mode.description}</p>
              </div>
              <div className="mode-select-icon">
                {selectedMode === mode.id ? (
                  <span className="loading-spinner"></span>
                ) : (
                  <span className="arrow-icon">→</span>
                )}
              </div>
            </div>
          ))}
        </div>
      </main>
      
      <footer className="mode-footer">
        <p>
          Need help getting started? <button className="help-link">View our guide</button>
        </p>
        <p className="disclaimer-text">
          MedAssist AI provides general information and is not a substitute for professional medical advice, diagnosis, or treatment.
        </p>
      </footer>
    </div>
  );
}

export default ModeSelection;
