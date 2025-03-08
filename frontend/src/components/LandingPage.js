import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

function LandingPage({ isAuthenticated }) {
  const navigate = useNavigate();
  const [isScrolled, setIsScrolled] = useState(false);
  const [activeSection, setActiveSection] = useState('home');
  const sections = useRef({});
  const observer = useRef(null);
  
  // Track scroll position and update nav styling accordingly
  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY;
      if (scrollPosition > 50) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  // Set up intersection observer to track which section is in view
  useEffect(() => {
    sections.current = {
      home: document.getElementById('home'),
      features: document.getElementById('features'),
      howItWorks: document.getElementById('how-it-works'),
      testimonials: document.getElementById('testimonials')
    };
    
    const observerOptions = {
      root: null,
      rootMargin: '-20% 0px -80% 0px', // When section is 20% into view
      threshold: 0
    };
    
    observer.current = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          setActiveSection(entry.target.id);
        }
      });
    }, observerOptions);
    
    // Start observing each section
    Object.values(sections.current).forEach(section => {
      if (section) observer.current.observe(section);
    });
    
    return () => {
      if (observer.current) {
        Object.values(sections.current).forEach(section => {
          if (section) observer.current.unobserve(section);
        });
      }
    };
  }, []);
  
  // Scroll to section with smooth behavior
  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };
  
  // Handle navigation based on authentication status
  const handleGetStarted = () => {
    if (isAuthenticated) {
      navigate('/mode-selection');
    } else {
      navigate('/login');
    }
  };
  
  // Healthcare professionals quotes for testimonials
  const testimonials = [
    {
      quote: "MedAssist AI has transformed how I explain complex medical reports to my patients. It's an invaluable tool for patient education.",
      name: "Dr. Sarah Mitchell",
      title: "Family Physician",
      avatar: "👩‍⚕️"
    },
    {
      quote: "The accuracy and clarity of explanations provided by this platform help bridge the gap between medical terminology and patient understanding.",
      name: "Dr. James Wilson",
      title: "Cardiologist",
      avatar: "👨‍⚕️"
    },
    {
      quote: "I recommend MedAssist AI to patients who want to better understand their health data between appointments.",
      name: "Dr. Lisa Chen",
      title: "Internal Medicine",
      avatar: "👩‍⚕️"
    }
  ];
  
  // Stats for impact section
  const stats = [
    { value: "93%", label: "User satisfaction" },
    { value: "2M+", label: "Questions answered" },
    { value: "24/7", label: "Availability" }
  ];

  return (
    <div className="landing-page">
      {/* Fixed Navigation */}
      <nav className={`landing-nav ${isScrolled ? 'scrolled' : ''}`}>
        <div className="nav-container">
          <div className="nav-logo">
            <span className="logo-icon">🧠</span>
            <span className="logo-text">MedAssist AI</span>
          </div>
          <div className="nav-actions">
            <span 
              className={`nav-link ${activeSection === 'features' ? 'active' : ''}`} 
              onClick={() => scrollToSection('features')}
            >
              Features
            </span>
            <span 
              className={`nav-link ${activeSection === 'how-it-works' ? 'active' : ''}`}
              onClick={() => scrollToSection('how-it-works')}
            >
              How It Works
            </span>
            <span 
              className={`nav-link ${activeSection === 'testimonials' ? 'active' : ''}`}
              onClick={() => scrollToSection('testimonials')}
            >
              Testimonials
            </span>
            <button 
              className="nav-button" 
              onClick={() => navigate(isAuthenticated ? '/mode-selection' : '/login')}
            >
              {isAuthenticated ? 'Open Portal' : 'Sign In'}
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero" id="home">
        <div className="hero-content">
          <h1>Understand Your Health With AI Assistance</h1>
          <p>
            MedAssist AI helps you make sense of your health concerns and medical reports with 
            clear explanations and personalized guidance.
          </p>
          <div className="hero-buttons">
            <button className="cta-button primary" onClick={handleGetStarted}>
              Get Started
            </button>
            <button 
              className="cta-button secondary"
              onClick={() => scrollToSection('how-it-works')}
            >
              Learn More
            </button>
          </div>
          <div className="hero-badges">
            <div className="badge">
              <span className="badge-icon">🔒</span>
              <span>HIPAA Compliant</span>
            </div>
            <div className="badge">
              <span className="badge-icon">🧪</span>
              <span>Evidence-Based</span>
            </div>
            <div className="badge">
              <span className="badge-icon">🌐</span>
              <span>Accessible 24/7</span>
            </div>
          </div>
        </div>
        <div className="hero-image-container">
          <img 
            src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZG9jdG9yJTIwb24lMjBwaG9uZXxlbnwwfHwwfHx8MA%3D%3D"  
            alt="Doctor using digital health technology on smartphone"
            className="hero-image"
          />
        </div>
      </section>

      {/* Impact Stats */}
      <section className="impact-stats">
        <div className="stats-container">
          {stats.map((stat, index) => (
            <div key={index} className="stat-item">
              <div className="stat-value">{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="features">
        <div className="section-header">
          <h2>Features & Benefits</h2>
          <p>Powerful tools to help you understand and manage your health</p>
        </div>
        
        <div className="feature-cards">
          <div className="feature-card">
            <div className="feature-icon">💬</div>
            <h3>Health Query Assistant</h3>
            <p>Get answers to your health questions with personalized, evidence-based explanations in simple language.</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">📄</div>
            <h3>Medical Report Analysis</h3>
            <p>Upload your medical reports and receive clear explanations of what your test results mean.</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">📱</div>
            <h3>Accessible Anywhere</h3>
            <p>Access health information on any device, whenever you need it - no appointments necessary.</p>
          </div>
          
          <div className="feature-card">
            <div className="feature-icon">🔒</div>
            <h3>Privacy First</h3>
            <p>Your health data is encrypted and secured with industry-leading privacy standards.</p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="how-it-works">
        <div className="section-container">
          <div className="section-header">
            <h2>How MedAssist AI Works</h2>
            <p>Simple steps to better health understanding</p>
          </div>
          
          <div className="process-flow">
            <div className="process-step">
              <div className="step-number">1</div>
              <div className="step-icon">🔐</div>
              <h3>Create an Account</h3>
              <p>Sign up with a secure account to keep your health information private.</p>
            </div>
            
            <div className="process-connector"></div>
            
            <div className="process-step">
              <div className="step-number">2</div>
              <div className="step-icon">💬</div>
              <h3>Ask Questions or Upload Reports</h3>
              <p>Describe symptoms, ask health questions, or upload medical reports for analysis.</p>
            </div>
            
            <div className="process-connector"></div>
            
            <div className="process-step">
              <div className="step-number">3</div>
              <div className="step-icon">🧠</div>
              <h3>Get AI-Powered Insights</h3>
              <p>Receive clear explanations, evidence-based information, and suggested next steps.</p>
            </div>
          </div>
          
          <div className="demo-preview">
            <div className="demo-text">
              <h3>See MedAssist in Action</h3>
              <p>Watch how our platform provides personalized health guidance and medical report interpretation.</p>
              <button className="demo-button" onClick={handleGetStarted}>
                Try Demo
              </button>
            </div>
            <div className="demo-image">
              <img 
                src="https://img.freepik.com/premium-photo/medical-technology-ai-innovation-healthcare-digital-surgeon-doctor-analysis-diagnose-checking-brain-testing-result-with-robotic-network-modern-interface_1028938-123619.jpg?w=900" 
                alt="MedAssist AI demo screenshot showing health analysis interface" 
              /> 
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section id="testimonials" className="testimonials">
        <div className="section-header">
          <h2>What Healthcare Professionals Say</h2>
          <p>Trusted by doctors and patients alike</p>
        </div>
        
        <div className="testimonials-container">
          {testimonials.map((testimonial, index) => (
            <div key={index} className="testimonial-card">
              <div className="quote-mark">"</div>
              <p className="testimonial-quote">{testimonial.quote}</p>
              <div className="testimonial-author">
                <div className="author-avatar">{testimonial.avatar}</div>
                <div className="author-info">
                  <h4>{testimonial.name}</h4>
                  <p>{testimonial.title}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="cta-container">
          <h2>Ready to understand your health better?</h2>
          <p>Join thousands of users who have improved their health literacy with MedAssist AI.</p>
          <button className="cta-button primary large" onClick={handleGetStarted}>
            Get Started Now
          </button>
          <div className="disclaimer-note">
            No credit card required. Medical information is provided for educational purposes only.
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-content">
          <div className="footer-section">
            <h3>MedAssist AI</h3>
            <p>Making medical information accessible and understandable through artificial intelligence.</p>
          </div>
          
          <div className="footer-section">
            <h4>Links</h4>
            <ul>
              <li><button onClick={() => navigate('/login')}>Sign In</button></li>
              <li><button onClick={() => navigate('/login')}>Create Account</button></li>
              <li><button onClick={() => scrollToSection('features')}>Features</button></li>
              <li><button onClick={() => scrollToSection('how-it-works')}>How It Works</button></li>
            </ul>
          </div>
          
          <div className="footer-section">
            <h4>Legal</h4>
            <ul>
              <li><button onClick={() => alert('Terms of Service would open here')}>Terms of Service</button></li>
              <li><button onClick={() => alert('Privacy Policy would open here')}>Privacy Policy</button></li>
              <li><button onClick={() => alert('Medical disclaimer would open here')}>Disclaimer</button></li>
            </ul>
          </div>
          
          <div className="footer-section">
            <h4>Contact</h4>
            <ul>
              <li><button onClick={() => alert('Support contact would open here')}>Support</button></li>
              <li><button onClick={() => alert('Feedback form would open here')}>Feedback</button></li>
              <li><button onClick={() => alert('Partnership information would open here')}>Partnerships</button></li>
            </ul>
          </div>
        </div>
        
        <div className="footer-bottom">
          <p>&copy; {new Date().getFullYear()} MedAssist AI. All rights reserved.</p>
          <p className="medical-disclaimer">
            Not a substitute for professional medical advice, diagnosis, or treatment.
            Always seek the advice of your physician or other qualified health provider.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default LandingPage;
