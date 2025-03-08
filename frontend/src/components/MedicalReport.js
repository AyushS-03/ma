import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

function MedicalReport({ token, onLogout }) {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [fileError, setFileError] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [analysis, setAnalysis] = useState('');
  const [isFallback, setIsFallback] = useState(false);
  const fileInputRef = useRef(null);
  const [userName, setUserName] = useState('User');
  const [isDragging, setIsDragging] = useState(false);
  const [filePreview, setFilePreview] = useState(null);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [expandedSections, setExpandedSections] = useState({});
  const [reportHistory, setReportHistory] = useState([]);
  const dropZoneRef = useRef(null);
  const [activeTerm, setActiveTerm] = useState(null);
  const [analysisStage, setAnalysisStage] = useState(''); // 'uploading', 'extracting', 'analyzing', 'complete'
  
  // Load username and report history when component mounts
  useEffect(() => {
    // Get username specifically from localStorage
    const storedUser = localStorage.getItem('username');
    console.log("MedicalReport - Loading username from localStorage:", storedUser);
    
    if (storedUser && storedUser !== 'undefined' && storedUser !== 'null') {
      setUserName(storedUser);
    } else {
      console.warn("MedicalReport - Valid username not found in localStorage, using default");
    }
    
    // Load report history from local storage
    const savedHistory = localStorage.getItem('reportHistory');
    if (savedHistory) {
      try {
        setReportHistory(JSON.parse(savedHistory));
      } catch (e) {
        console.error('Error loading report history:', e);
      }
    }
  }, []);
  
  // Simulate progress during analysis
  useEffect(() => {
    if (isUploading) {
      setAnalysisProgress(0);
      const interval = setInterval(() => {
        setAnalysisProgress((prevProgress) => {
          const newProgress = prevProgress + Math.random() * 15;
          
          // Update analysis stage based on progress
          if (newProgress < 30) setAnalysisStage('uploading');
          else if (newProgress < 60) setAnalysisStage('extracting');
          else setAnalysisStage('analyzing');
          
          return newProgress >= 95 ? 95 : newProgress;
        });
      }, 500);
      
      return () => clearInterval(interval);
    } else if (analysisProgress > 0) {
      // When complete, jump to 100%
      setAnalysisProgress(100);
      setAnalysisStage('complete');
    }
  }, [isUploading]);

  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    handleFileSelection(selectedFile);
  };
  
  // Common function for handling file selection from both input and drop
  const handleFileSelection = (selectedFile) => {
    setFileError('');
    setFilePreview(null);
    
    if (!selectedFile) {
      setFile(null);
      return;
    }
    
    // Check file type
    const fileType = selectedFile.type;
    const validTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
    
    if (!validTypes.includes(fileType)) {
      setFileError('Please select a PDF, DOC, DOCX, or TXT file.');
      setFile(null);
      return;
    }
    
    // Check file size (max 5MB)
    if (selectedFile.size > 5 * 1024 * 1024) {
      setFileError('File size must be less than 5MB.');
      setFile(null);
      return;
    }
    
    // Set file and create preview
    setFile(selectedFile);
    
    // Create preview based on file type
    if (fileType === 'application/pdf') {
      setFilePreview({
        type: 'pdf',
        name: selectedFile.name,
        size: formatFileSize(selectedFile.size)
      });
    } else if (fileType.includes('document')) {
      setFilePreview({
        type: 'doc',
        name: selectedFile.name,
        size: formatFileSize(selectedFile.size)
      });
    } else {
      setFilePreview({
        type: 'txt',
        name: selectedFile.name,
        size: formatFileSize(selectedFile.size)
      });
    }
  };
  
  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    else return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };
  
  // Handle drag and drop events
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };
  
  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.target === dropZoneRef.current) {
      setIsDragging(false);
    }
  };
  
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length) {
      handleFileSelection(files[0]);
    }
  };
  
  // Extract sections from markdown analysis
  const extractSections = (markdown) => {
    if (!markdown) return [];
    
    // Find section headers (## or ### headings)
    const sectionRegex = /^(#{2,3})\s+(.+)$/gm;
    const sections = [];
    let match;
    let lastIndex = 0;
    
    while ((match = sectionRegex.exec(markdown)) !== null) {
      const level = match[1].length; // 2 for ## or 3 for ###
      const title = match[2];
      const startPos = match.index;
      
      if (sections.length > 0) {
        // Set content of the previous section
        sections[sections.length - 1].content = markdown.slice(
          sections[sections.length - 1].startPos + sections[sections.length - 1].title.length + level + 1, 
          startPos
        ).trim();
      }
      
      sections.push({
        level,
        title,
        startPos,
        content: '',
        id: `section-${sections.length}`
      });
      
      lastIndex = startPos;
    }
    
    // Handle content for the last section
    if (sections.length > 0) {
      const lastSection = sections[sections.length - 1];
      lastSection.content = markdown.slice(
        lastSection.startPos + lastSection.title.length + lastSection.level + 1
      ).trim();
    }
    
    return sections;
  };
  
  // Toggle section expansion
  const toggleSection = (sectionId) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };
  
  // Handle form submission to analyze report
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setFileError('Please select a file to analyze.');
      return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    setIsUploading(true);
    setAnalysis('');
    setExpandedSections({});
    
    try {
      const response = await axios.post('http://localhost:5000/api/analyze-report', formData, {
        headers: {
          'Authorization': token,
          'Content-Type': 'multipart/form-data'
        }
      });
      
      console.log("Response data:", response.data); // Debug logging
      
      if (response.data && response.data.analysis) {
        setAnalysis(response.data.analysis);
        setIsFallback(response.data.is_fallback || false);
        
        // Initialize all sections as expanded
        const sections = extractSections(response.data.analysis);
        const initialExpandedState = {};
        sections.forEach(section => {
          initialExpandedState[section.id] = true;
        });
        setExpandedSections(initialExpandedState);
        
        // Add to report history
        const newHistoryItem = {
          id: Date.now(),
          filename: file.name,
          date: new Date().toISOString(),
          previewType: filePreview?.type || 'file',
          snippet: response.data.analysis.substring(0, 100) + '...'
        };
        
        const updatedHistory = [newHistoryItem, ...reportHistory.slice(0, 4)];
        setReportHistory(updatedHistory);
        
        // Save to localStorage
        localStorage.setItem('reportHistory', JSON.stringify(updatedHistory));
      } else {
        throw new Error('Invalid response format from server');
      }
      
    } catch (error) {
      console.error('Error analyzing report:', error);
      
      if (error.response?.status === 401) {
        alert('Your session has expired. Please login again.');
        onLogout();
      } else {
        const errorMsg = error.response?.data?.error || 'Something went wrong during analysis. Please try again.';
        setFileError(errorMsg);
      }
    } finally {
      setIsUploading(false);
    }
  };
  
  // Handle clearing file selection
  const handleClearFile = () => {
    setFile(null);
    setFileError('');
    setFilePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  // Clear analysis and return to upload screen
  const handleStartNew = () => {
    setAnalysis('');
    setFile(null);
    setFilePreview(null);
    setFileError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };
  
  // Handle selection of a history item
  const handleHistorySelect = (item) => {
    // In a real app, we would fetch the stored analysis
    // For now, just show a placeholder
    alert(`In a complete implementation, this would reload the analysis for: ${item.filename}`);
  };
  
  // Handle mouse enter/leave for medical term tooltips
  const handleTermMouseEnter = (term) => {
    setActiveTerm(term);
  };
  
  const handleTermMouseLeave = () => {
    setActiveTerm(null);
  };
  
  // Modified UserProfile component with simplified menu
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
  
  // Medical terms with definitions for tooltips
  const medicalTerms = {
    'hemoglobin': 'A protein in red blood cells that carries oxygen throughout the body.',
    'hematocrit': 'The percentage of your blood that is made up of red blood cells.',
    'wbc': 'White Blood Cells - cells that help fight infections and other diseases.',
    'rbc': 'Red Blood Cells - cells that carry oxygen from your lungs to the rest of your body.',
    'platelets': 'Cell fragments that help your blood clot and prevent excessive bleeding.',
    'cholesterol': 'A waxy substance found in the blood that the body needs to build healthy cells.',
    'ldl': 'Low-Density Lipoprotein - often called "bad" cholesterol.',
    'hdl': 'High-Density Lipoprotein - often called "good" cholesterol.',
    'triglycerides': 'A type of fat found in your blood that your body uses for energy.',
    'creatinine': 'A waste product in the blood that comes from muscle activity.',
    'glucose': 'A type of sugar and the main source of energy for the body.',
    'hba1c': 'Glycated Hemoglobin - reflects average blood sugar levels over the past 2-3 months.',
  };
  
  // Helper function to highlight medical terms in analysis
  const highlightMedicalTerms = (text) => {
    if (!text) return '';
    
    let highlightedText = text;
    Object.keys(medicalTerms).forEach(term => {
      const regex = new RegExp(`\\b${term}\\b`, 'gi');
      highlightedText = highlightedText.replace(regex, 
        `<span class="medical-term" data-term="${term.toLowerCase()}">${term}</span>`
      );
    });
    
    return highlightedText;
  };

  // Get file icon based on type
  const getFileIcon = (type) => {
    switch (type) {
      case 'pdf': return '📄';
      case 'doc': return '📝';
      case 'txt': return '📃';
      default: return '📁';
    }
  };
  
  // Progress animation styles
  const progressBarStyles = {
    track: {
      width: '100%',
      height: '8px',
      backgroundColor: '#e0e0e0',
      borderRadius: '4px',
      margin: '10px 0'
    },
    indicator: {
      height: '100%',
      backgroundColor: '#4285f4',
      borderRadius: '4px',
      transition: 'width 0.3s ease',
      width: `${analysisProgress}%`
    }
  };

  return (
    <div className="chat-page">
      <header className="chat-header">
        <div className="header-left">
          <button className="back-button" onClick={() => navigate('/mode-selection')}>
            <span className="back-icon">←</span>
          </button>
          <div className="chat-title">
            <h1>Medical Report Analysis</h1>
            <p>AI-powered medical document analysis</p>
          </div>
        </div>
        <div className="header-right">
          <UserProfile userName={userName} onLogout={onLogout} />
        </div>
      </header>

      <main className="report-main-container">
        <div className="report-layout">
          {/* Left Column - Upload or Current Report */}
          <div className="report-primary-column">
            {!analysis ? (
              <div className="report-upload-panel">
                <div className="panel-header">
                  <h2>Upload Medical Report</h2>
                  <p>Upload your medical report document for AI-powered analysis. We support PDF, DOC, DOCX, and TXT files.</p>
                </div>
                
                <div 
                  ref={dropZoneRef}
                  className={`report-dropzone ${isDragging ? 'dragging' : ''}`}
                  onDragEnter={handleDragEnter}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <input
                    type="file"
                    onChange={handleFileChange}
                    ref={fileInputRef}
                    id="file-input"
                    className="file-input-hidden"
                  />
                  
                  {!filePreview ? (
                    <label htmlFor="file-input" className="dropzone-placeholder">
                      <div className="dropzone-icon">📄</div>
                      <div className="dropzone-text">
                        <span className="dropzone-primary">Drag & drop your file here</span>
                        <span className="dropzone-secondary">or click to browse</span>
                      </div>
                      <div className="dropzone-formats">
                        Supports PDF, DOC, DOCX, TXT (Max 5MB)
                      </div>
                    </label>
                  ) : (
                    <div className="file-preview">
                      <div className="file-preview-icon">{getFileIcon(filePreview.type)}</div>
                      <div className="file-preview-details">
                        <div className="file-preview-name">{filePreview.name}</div>
                        <div className="file-preview-size">{filePreview.size}</div>
                      </div>
                      <button 
                        type="button" 
                        className="file-preview-remove"
                        onClick={handleClearFile}
                        aria-label="Remove file"
                      >
                        ×
                      </button>
                    </div>
                  )}
                </div>
                
                {fileError && (
                  <div className="file-error-message">
                    <span className="error-icon">⚠️</span>
                    {fileError}
                  </div>
                )}
                
                <div className="report-action-buttons">
                  <button 
                    type="button" 
                    className="action-button primary"
                    disabled={!file || isUploading}
                    onClick={handleSubmit}
                  >
                    {isUploading ? (
                      <div className="button-content">
                        <div className="button-loader"></div>
                        <span>Analyzing...</span>
                      </div>
                    ) : (
                      <div className="button-content">
                        <span className="button-icon">🔍</span>
                        <span>Analyze Report</span>
                      </div>
                    )}
                  </button>
                  
                  <button 
                    type="button" 
                    className="action-button secondary"
                    onClick={() => navigate('/general-query')}
                  >
                    <div className="button-content">
                      <span className="button-icon">💬</span>
                      <span>Go to Chat Assistant</span>
                    </div>
                  </button>
                </div>
                
                {isUploading && (
                  <div className="analysis-progress-container">
                    <div className="progress-stage">{analysisStage}</div>
                    <div style={progressBarStyles.track}>
                      <div style={progressBarStyles.indicator}></div>
                    </div>
                    <div className="progress-percentage">{Math.round(analysisProgress)}%</div>
                  </div>
                )}
              </div>
            ) : (
              <div className="report-analysis-panel">
                <div className="panel-header with-actions">
                  <div>
                    <h2>Analysis Results</h2>
                    <p>{filePreview?.name || 'Medical Report'}</p>
                  </div>
                  
                  <div className="panel-actions">
                    {isFallback && (
                      <div className="fallback-badge">
                        <span className="fallback-icon">⚠️</span>
                        Using local processing
                      </div>
                    )}
                    <button 
                      className="icon-button"
                      onClick={handleStartNew}
                      title="Analyze another report"
                    >
                      <span className="icon-button-content">🔄</span>
                    </button>
                    <button 
                      className="icon-button"
                      onClick={() => window.print()}
                      title="Print or save as PDF"
                    >
                      <span className="icon-button-content">🖨️</span>
                    </button>
                  </div>
                </div>
                
                <div className="report-content">
                  {analysis && <ReactMarkdown>{analysis}</ReactMarkdown>}
                  
                  {activeTerm && (
                    <div className="medical-term-tooltip">
                      <div className="tooltip-header">
                        {activeTerm.toUpperCase()}
                        <button 
                          className="tooltip-close"
                          onClick={() => setActiveTerm(null)}
                        >
                          ×
                        </button>
                      </div>
                      <div className="tooltip-content">
                        {medicalTerms[activeTerm.toLowerCase()]}
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="panel-footer">
                  <p className="disclaimer-text">
                    This analysis is for informational purposes only and should not replace professional medical advice.
                    Always consult with your healthcare provider to interpret medical reports correctly.
                  </p>
                </div>
              </div>
            )}
          </div>
          
          {/* Right Column - History and Information */}
          <div className="report-secondary-column">
            {/* History Panel */}
            <div className="report-history-panel">
              <h3>Recent Reports</h3>
              
              {reportHistory.length === 0 ? (
                <div className="history-empty">
                  <div className="empty-icon">📂</div>
                  <p>No recent reports</p>
                </div>
              ) : (
                <div className="history-list">
                  {reportHistory.map((item) => (
                    <div 
                      key={item.id} 
                      className="history-item"
                      onClick={() => handleHistorySelect(item)}
                    >
                      <div className="history-icon">
                        {getFileIcon(item.previewType)}
                      </div>
                      <div className="history-details">
                        <div className="history-filename">{item.filename}</div>
                        <div className="history-date">
                          {new Date(item.date).toLocaleDateString()}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            {/* Information Panel */}
            <div className="report-info-panel">
              <h3>About Medical Reports</h3>
              <p>
                Medical reports can contain complex terminology and measurements. Our AI assistant
                helps you understand your report by identifying key metrics and providing context.
              </p>
              
              <div className="info-card">
                <div className="info-icon">💡</div>
                <div className="info-content">
                  <h4>Click on highlighted terms</h4>
                  <p>Medical terms are highlighted in blue. Click on them to see a simple explanation.</p>
                </div>
              </div>
              
              <div className="info-card">
                <div className="info-icon">⚠️</div>
                <div className="info-content">
                  <h4>Important Note</h4>
                  <p>This tool is for educational purposes only. Always consult with healthcare professionals for medical advice.</p>
                </div>
              </div>
              
              <div className="action-links">
                <button 
                  className="text-link"
                  onClick={() => window.open('https://www.who.int/health-topics', '_blank')}
                >
                  <span className="link-icon">🌐</span>
                  WHO Health Resources
                </button>
                
                <button 
                  className="text-link"
                  onClick={() => navigate('/general-query')}
                >
                  <span className="link-icon">💬</span>
                  Ask our AI Assistant
                </button>
              </div>
            </div>
          </div>
        </div>
      </main>
      
      <footer className="chat-footer">
        <div className="footer-content">
          <div className="disclaimer">
            <p>Not a substitute for professional medical advice. Consult a qualified healthcare provider for diagnosis and treatment.</p>
          </div>
          <div className="footer-links">
            <button className="footer-link">Privacy</button>
            <button className="footer-link">Terms</button>
            <button className="footer-link">Help</button>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default MedicalReport;
