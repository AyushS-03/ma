import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './GeneralQuery.css';

function GeneralQuery({ token, onLogout }) {
  const navigate = useNavigate();
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [showOptions, setShowOptions] = useState(false);
  const [userName, setUserName] = useState('User');
  const [activeTip, setActiveTip] = useState(0);
  const [activeCategory, setActiveCategory] = useState(null);
  const messageListRef = useRef(null);
  const inputRef = useRef(null);
  
  // Sample quick questions
  const quickQuestions = [
    "I have a headache that won't go away",
    "What causes stomach pain after eating?",
    "Is my fever dangerous?",
    "How can I reduce back pain?"
  ];

  // Health categories for suggestions with updated category-specific suggestions
  const healthCategories = [
    { 
      id: "respiratory",
      name: "Respiratory", 
      icon: "🫁",
      color: "#4ECDC4",
      examples: [
        "I've been coughing for 3 days",
        "Why am I short of breath?",
        "What can help with a stuffy nose?",
        "Could my chest pain be from a respiratory issue?"
      ]
    },
    { 
      id: "digestive",
      name: "Digestive", 
      icon: "🫀", // Changed icon for digestive
      color: "#FF6B6B",
      examples: [
        "My stomach hurts after meals",
        "What causes acid reflux?",
        "How to relieve bloating?",
        "Why do I have constant gas pain?"
      ]
    },
    { 
      id: "neurological",
      name: "Neurological", 
      icon: "🧠",
      color: "#3A86FF",
      examples: [
        "I've had migraines for a week",
        "Why do I feel dizzy when standing?",
        "How to manage chronic headaches?",
        "Is tingling in my hands a concern?"
      ]
    },
    { 
      id: "musculoskeletal",
      name: "Musculoskeletal", 
      icon: "🦴",
      color: "#8338EC",
      examples: [
        "My knee pain gets worse at night",
        "What helps with lower back pain?",
        "Why do my joints hurt in cold weather?",
        "How to treat muscle soreness after exercise?"
      ]
    },
    { 
      id: "mental",
      name: "Mental Health", 
      icon: "😌",
      color: "#FB5607",
      examples: [
        "I've been feeling anxious lately",
        "How to improve my sleep?",
        "What are signs of depression?",
        "Tips for managing stress at work?"
      ]
    },
    { 
      id: "skin",
      name: "Skin & Allergies", 
      icon: "👨‍⚕️",
      color: "#06D6A0",
      examples: [
        "I have a rash that won't go away",
        "What causes hives?",
        "How to treat eczema?",
        "Could my itchy skin be an allergic reaction?"
      ]
    }
  ];

  // Health tips carousel
  const healthTips = [
    {
      title: "Stay Hydrated",
      content: "Aim to drink 8-10 glasses of water daily to maintain proper body function and flush out toxins.",
      icon: "💧"
    },
    {
      title: "Sleep Matters",
      content: "Adults need 7-9 hours of quality sleep each night for optimal physical and mental health.",
      icon: "😴"
    },
    {
      title: "Regular Exercise",
      content: "Try to get at least 150 minutes of moderate exercise or 75 minutes of vigorous activity weekly.",
      icon: "🏃‍♀️"
    },
    {
      title: "Balanced Diet",
      content: "Fill half your plate with fruits and vegetables, one quarter with protein, and one quarter with grains.",
      icon: "🥗"
    },
    {
      title: "Mental Wellness",
      content: "Practice mindfulness or meditation for at least 10 minutes daily to reduce stress and anxiety.",
      icon: "🧘‍♂️"
    }
  ];

  // Carousel timer
  useEffect(() => {
    const timer = setInterval(() => {
      setActiveTip((prev) => (prev + 1) % healthTips.length);
    }, 8000);
    
    return () => clearInterval(timer);
  }, []);

  // Update this useEffect to handle username more robustly
  useEffect(() => {
    // Get username if available
    const storedUser = localStorage.getItem('username');
    console.log("GeneralQuery - Loading username from localStorage:", storedUser);
    
    if (storedUser && storedUser !== 'undefined' && storedUser !== 'null') {
      setUserName(storedUser);
    } else {
      console.warn("GeneralQuery - Valid username not found in localStorage, using default");
    }
  }, []); 

  // Verify token and add welcome message - keep this separate
  useEffect(() => {
    if (!token) {
      console.error("No authentication token available");
      onLogout();
      return;
    }
    
    // Add welcome message if chat is empty
    if (chatHistory.length === 0) {
      setChatHistory([
        {
          text: `Hello ${userName}! I'm your medical assistant. How can I help you today? Feel free to describe your symptoms or health concerns, or select a health category from the sidebar to get started.`,
          sender: 'bot',
          timestamp: new Date()
        }
      ]);
    }
  }, [token, onLogout, chatHistory.length, userName]); // Add userName as dependency
  
  // Scroll to bottom whenever chat history changes
  useEffect(() => {
    if (messageListRef.current) {
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
    }
  }, [chatHistory]);

  // Handle form submission
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (message.trim()) {  // Only send if message is not empty
      sendMessage(message);
    }
  };
  
  // Handle quick question selection
  const handleQuickQuestion = (question) => {
    setMessage(question);
    setShowOptions(false);
    // Wait a bit to allow visual feedback before sending
    setTimeout(() => {
      sendMessage(question);
    }, 300);
  };

  // Handle category example selection
  const handleCategoryExample = (example) => {
    setMessage(example);
    setShowOptions(false);
    // Wait a bit to allow visual feedback before sending
    setTimeout(() => {
      // Before sending, log what we'd expect to see as quick replies after getting a response
      console.log("Sending category example:", example);
      console.log("Expected question pattern in response - debug preview:", debugQuestions(example));
      sendMessage(example);
    }, 300);
  };

  // Select a category
  const handleCategorySelect = (categoryId) => {
    setActiveCategory(categoryId === activeCategory ? null : categoryId);
  };
  
  // Common function to send messages - improved error handling and fallbacks
const sendMessage = async (messageText) => {
  if (!messageText.trim()) return;
  
  setError('');

  // Get current time for timestamp
  const now = new Date();
  
  // Add user message to chat
  const userMessage = { 
    text: messageText, 
    sender: 'user',
    timestamp: now
  };
  
  setChatHistory(prevChat => [...prevChat, userMessage]);
  setIsLoading(true);
  setMessage('');
  
  // Focus input after sending
  setTimeout(() => {
    if (inputRef.current) inputRef.current.focus();
  }, 100);

  try {
    const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
    console.log(`Sending message to API at: ${apiUrl}`);
    const response = await axios.post(`${apiUrl}/api/chat`, 
      { message: messageText },
      { 
        headers: { 
          'Authorization': token,
          'Content-Type': 'application/json'
        },
        timeout: 15000 // Keep timeout to prevent hanging requests
      }
    );

    console.log("API response:", response.data);

    // Log details for debugging
    if (response.data && response.data.message) {
      console.log(`Response length: ${response.data.message.length}`);
      console.log(`Response preview: ${response.data.message.substring(0, 100)}...`);
    } else {
      console.error("Response missing expected message field");
    }

    // Enhanced validation of response
    if (response.data && typeof response.data.message === 'string' && response.data.message.trim()) {
      // Check if we got a fallback response or error
      const isSystemFallback = response.data.is_fallback;
      const isError = response.data.error;
      
      // Add bot response to chat
      setChatHistory(prevChat => [
        ...prevChat,
        { 
          text: response.data.message, 
          sender: 'bot',
          isFallback: isSystemFallback || isError,
          timestamp: new Date()
        }
      ]);
      
      // Show notice if using fallback mode but don't disrupt the user experience
      if (isSystemFallback || isError) {
        // More subtle error handling - set a message but don't show a popup
        console.warn("Using fallback response mechanism");
        // Only show error banner if there's a serious error
        if (isError) {
          setError('The AI assistant is experiencing some issues but is still working. You might get more generic responses at the moment.');
        }
      }
    } else {
      throw new Error("Invalid response format");
    }
  } catch (error) {
    console.error('Error sending message:', error);
    
    let errorMsg = 'Something went wrong. Please try again.';
    
    if (error.response?.status === 401) {
      errorMsg = 'Your session has expired. Please login again.';
      setTimeout(() => onLogout(), 2000);
    } else if (error.response?.data?.error) {
      errorMsg = error.response.data.error;
    } else if (error.code === 'ECONNABORTED') {
      errorMsg = 'Request timed out. The server might be busy. Try asking a simpler question.';
    } else if (!navigator.onLine) {
      errorMsg = 'Network connection lost. Please check your internet connection.';
    }
    
    setError(errorMsg);
    
    // Always provide a fallback bot response to maintain conversation flow
    setChatHistory(prevChat => [
      ...prevChat,
      { 
        text: "I understand your question, but I'm having trouble processing it right now. Could you try rephrasing or asking a different health question?", 
        sender: 'bot',
        isFallback: true,
        timestamp: new Date()
      }
    ]);
  } finally {
    setIsLoading(false);
  }
};

  // Format time for messages
  const formatMessageTime = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // New state for interactive question parsing
  const [parsedQuestions, setParsedQuestions] = useState([]);
  const [customAnswer, setCustomAnswer] = useState('');
  const [isQuestionActive, setIsQuestionActive] = useState(false);
  
  // Function to parse structured questions from bot messages - FURTHER IMPROVED
  const parseQuestionsFromMessage = (messageText) => {
    // Skip parsing if the message is the welcome message
    if (messageText.includes("I'm your medical assistant") && 
        messageText.includes("Feel free to describe your symptoms")) {
      return [];
    }

    // Don't parse when the message provides explanations about medical conditions
    if (/understanding|possible causes|causes of|symptoms of|about|overview/i.test(messageText.split('\n')[0])) {
      console.log("Detected educational content - skipping option generation");
      return [];
    }

    // Don't parse if it's a long informational response with markdown headings
    if (messageText.includes("**") && messageText.length > 500) {
      // Check if this appears to be an informational article with headers
      const headerCount = (messageText.match(/\*\*[^*]+\*\*/g) || []).length;
      if (headerCount >= 3) {
        console.log("Detected markdown article format - skipping option generation");
        return [];
      }
    }
    
    // Don't parse if the message contains cause identification or diagnostic content
    if (/diagnosis|diagnosed|likely cause|condition appears to be|suffering from|suggests that you have|indicates|confirmed|conclude|caused by|possible causes|common causes/i.test(messageText)) {
      console.log("Detected diagnosis/cause identification - skipping option generation");
      return [];
    }

    // Look for a clearly defined question section
    let questionSectionFound = false;
    
    // Pattern 1: Look for "Specific Questions" section
    const specificQuestionsHeader = /\*\*Specific Questions[^*]*\*\*\s*-+\s*/i;
    const hasSpecificQuestionsSection = specificQuestionsHeader.test(messageText);
    
    if (hasSpecificQuestionsSection) {
      questionSectionFound = true;
      console.log("Found 'Specific Questions' section");
      
      // Extract the questions section
      const questionSectionRegex = /\*\*Specific Questions[^*]*\*\*\s*-+\s*([^#]+?)(?=\*\*|$)/s;
      const questionSection = messageText.match(questionSectionRegex);
      
      if (questionSection && questionSection[1]) {
        // Extract actual questions with question marks from this section
        const questionsText = questionSection[1];
        const questionMarkRegex = /([^.!?]+\?)/g;
        const matches = [...questionsText.matchAll(questionMarkRegex)];
        
        if (matches.length > 0) {
          return matches
            .filter(match => !match[1].includes("Remember"))  // Filter out reminders
            .map(match => ({
              question: match[1].trim(),
              options: generateOptionsForQuestion(match[1].trim()),
              answered: false,
              answer: ''
            }));
        }
      }
    }
    
    // Pattern 2: Look for questions in bullet points or numbered lists
    // Only do this if we haven't already found a specific questions section
    if (!questionSectionFound) {
      // Try to identify a questions or next steps section
      const nextStepsRegex = /(?:\*\*Next Steps\*\*|\*\*Questions\*\*|\bTo better understand|\bI'd like to ask|\bI would like to know|\bTo provide more specific|could you share:)([^*]+?)(?=\*\*|$)/s;
      const nextStepsSection = messageText.match(nextStepsRegex);
      
      if (nextStepsSection && nextStepsSection[1]) {
        // Look for bullet-pointed or numbered questions
        const bulletPointRegex = /(?:[•*\-]|\d+\.)\s*([^.!?]+\?)/g;
        const bulletMatches = [...nextStepsSection[1].matchAll(bulletPointRegex)];
        
        if (bulletMatches.length > 0) {
          return bulletMatches
            .filter(match => !match[1].toLowerCase().includes("remember") && !match[1].toLowerCase().includes("seek medical"))
            .map(match => ({
              question: match[1].trim(),
              options: generateOptionsForQuestion(match[1].trim()),
              answered: false,
              answer: ''
            }));
        }
      }
    }

    // If we've reached this point and haven't found structured questions, 
    // check for direct questions at the end of the message (common pattern for follow-up questions)
    const lastParagraphs = messageText.split('\n\n').slice(-3).join('\n\n');
    if (lastParagraphs.includes("?") && 
        !lastParagraphs.toLowerCase().includes("feel free") && 
        !lastParagraphs.toLowerCase().includes("please let me know") && 
        !lastParagraphs.toLowerCase().includes("remember")) {
      
      // Extract question marks from these last paragraphs
      const questionMarkRegex = /([^.!?]+\?)/g;
      const matches = [...lastParagraphs.matchAll(questionMarkRegex)];
      
      if (matches.length > 0) {
        // Only get the last 1-2 questions, as these are likely the direct follow-ups
        return matches
          .slice(-2) // Take only the last 2 questions at most
          .filter(match => 
            !match[1].toLowerCase().includes("remember") && 
            !match[1].toLowerCase().includes("seek medical") &&
            !match[1].toLowerCase().includes("any concerns") &&
            !match[1].toLowerCase().includes("any questions") &&
            match[1].length > 15
          )
          .map(match => ({
            question: match[1].trim(),
            options: generateOptionsForQuestion(match[1].trim()),
            answered: false,
            answer: ''
          }));
      }
    }
    
    // If we still haven't found anything, return empty array
    return [];
  };

  // Generate appropriate options based on the question
  const generateOptionsForQuestion = (question) => {
    const questionLower = question.toLowerCase();
    
    // Headache related questions
    if (/headache|migraine|head pain/i.test(questionLower)) {
      if (/how (long|often|frequently)/i.test(questionLower)) {
        return [
          "Just started today", 
          "A few days", 
          "Recurring for weeks",
          "Chronic (months or years)"
        ];
      } else if (/intensity|severe|pain level|rate/i.test(questionLower)) {
        return ["Mild", "Moderate", "Severe", "Varies throughout the day"];
      } else if (/where|location|area|position/i.test(questionLower)) {
        return ["Front of head", "Back of head", "One side only", "All over", "Behind the eyes"];
      } else if (/trigger|cause|worsen/i.test(questionLower)) {
        return ["Stress", "Light or noise", "After eating", "Screen time", "Weather changes", "No clear pattern"];
      } else if (/relief|help|remedy|treat|improve/i.test(questionLower)) {
        return ["Over-the-counter pain relievers", "Rest in a dark room", "Cold/hot compress", "Hydration", "None of these help"];
      } else {
        return ["Yes", "No", "Sometimes", "Not sure", "Other (I'll specify)"];
      }
    }
    
    // Stomach/Digestive related questions
    if (/stomach|digest|nausea|vomit|bowel|diarrhea|constipation|acid reflux|bloat|gas/i.test(questionLower)) {
      if (/how (long|often|frequently)/i.test(questionLower)) {
        return [
          "Just started today", 
          "A few days", 
          "About a week", 
          "Several weeks", 
          "Chronic issue"
        ];
      } else if (/meal|eat|food|diet/i.test(questionLower)) {
        return ["Right after eating", "Hours after meals", "No relation to eating", "With specific foods only"];
      } else if (/trigger|cause|worsen/i.test(questionLower)) {
        return ["Spicy foods", "Dairy products", "Fatty foods", "Alcohol", "Coffee/caffeine", "Stress", "No clear pattern"];
      } else {
        return ["Yes", "No", "Sometimes", "Not sure", "Other (I'll specify)"];
      }
    }
    
    // Respiratory related questions
    if (/breath|cough|wheez|lung|airways|respir|chest congestion|phlegm|mucus/i.test(questionLower)) {
      if (/how (long|often|frequently)/i.test(questionLower)) {
        return ["Just started", "A few days", "About a week", "Several weeks", "Chronic issue"];
      } else if (/cough/i.test(questionLower) && /dry|productive|mucus|phlegm/i.test(questionLower)) {
        return ["Dry cough", "Productive with clear mucus", "Productive with colored mucus", "Varies throughout the day"];
      } else if (/trigger|cause|worsen/i.test(questionLower)) {
        return ["Cold air", "Exercise/exertion", "Allergies", "Lying down", "Time of day", "No clear pattern"];
      } else {
        return ["Yes", "No", "Sometimes", "Not sure", "Other (I'll specify)"];
      }
    }
    
    // Skin related questions
    if (/skin|rash|itch|hive|dermatitis|eczema|acne|bump|spot/i.test(questionLower)) {
      if (/how (long|often|frequently)/i.test(questionLower)) {
        return ["Just appeared", "A few days", "About a week", "Several weeks", "Chronic condition"];
      } else if (/where|location|area|position/i.test(questionLower)) {
        return ["Face", "Arms", "Legs", "Torso", "Widespread", "Specific spot (I'll describe)"];
      } else if (/appearance|look|color|texture/i.test(questionLower)) {
        return ["Red", "Raised/bumpy", "Flat", "Scaly/dry", "Blistering", "Multiple symptoms"];
      } else if (/trigger|cause|worsen/i.test(questionLower)) {
        return ["After using new product", "After eating certain foods", "Contact with substance", "Stress", "Heat/sweating", "No clear pattern"];
      } else {
        return ["Yes", "No", "Sometimes", "Not sure", "Other (I'll specify)"];
      }
    }
    
    // Joint/Muscle related questions
    if (/joint|muscle|bone|sprain|strain|arthritis|back pain|shoulder|knee|hip|ankle/i.test(questionLower)) {
      if (/how (long|often|frequently)/i.test(questionLower)) {
        return ["Just started", "A few days", "About a week", "Several weeks", "Chronic issue"];
      } else if (/intensity|severe|pain level|rate/i.test(questionLower)) {
        return ["Mild", "Moderate", "Severe", "Varies with activity"];
      } else if (/trigger|cause|worsen/i.test(questionLower)) {
        return ["After exercise", "Upon waking", "End of day", "During specific movements", "Weather changes", "No clear pattern"];
      } else {
        return ["Yes", "No", "Sometimes", "Not sure", "Other (I'll specify)"];
      }
    }
    
    // Mental health related questions
    if (/anxiety|depress|stress|mental health|mood|emotion|sleep|insomnia|tired|fatigue/i.test(questionLower)) {
      if (/how (long|often|frequently)/i.test(questionLower)) {
        return ["Just recently", "A few days", "Several weeks", "Months", "Long-term issue"];
      } else if (/intensity|severe|level|rate/i.test(questionLower)) {
        return ["Mild", "Moderate", "Severe", "Fluctuates"];
      } else if (/trigger|cause|worsen/i.test(questionLower)) {
        return ["Work/school stress", "Relationship issues", "Financial concerns", "Health worries", "No clear trigger"];
      } else if (/sleep/i.test(questionLower)) {
        return ["Trouble falling asleep", "Waking up during night", "Waking too early", "Unrefreshing sleep", "Mixed issues"];
      } else {
        return ["Yes", "No", "Sometimes", "Not sure", "Other (I'll specify)"];
      }
    }
    
    // Duration related questions (default)
    if (/how long|when did|duration|start|begin|first notice/i.test(questionLower)) {
      return [
        "Just started today", 
        "A few days", 
        "About a week", 
        "Several weeks", 
        "A month or more"
      ];
    }
    
    // Yes/No pattern questions (default)
    if (/have you|did you|are you|do you|has there/i.test(questionLower)) {
      return ["Yes", "No", "Sometimes", "Not sure"];
    }
    
    // Intensity related questions (default)
    if (/severe|intensity|rate the|pain level|how (bad|much|painful)/i.test(questionLower)) {
      return ["Mild", "Moderate", "Severe", "Varies"];
    }
    
    // Pattern related questions (default)
    if (/constant|comes and goes|intermittent|pattern|frequency/i.test(questionLower)) {
      return ["Constant", "Comes and goes", "Only after specific activities", "At certain times of day", "Random"];
    }
    
    // Food trigger related questions (default)
    if (/food|trigger|diet|eating|meal|drink|consume/i.test(questionLower)) {
      return ["Specific foods (I'll list them)", "After large meals", "When hungry/empty stomach", "No relation to food"];
    }
    
    // Generic options for other questions
    return ["Yes", "No", "Sometimes", "Not sure", "Other (I'll specify)"];
  };

// Add a debugging function to help understand how questions are being parsed
const debugQuestions = (messageText) => {
  console.log("Analyzing message for questions:", messageText);
  const questions = parseQuestionsFromMessage(messageText);
  if (questions.length > 0) {
    console.log(`Found ${questions.length} questions:`, questions);
  } else {
    console.log("No questions found in message");
  }
  return questions;
};

  // Process bot messages for interactive elements - IMPROVED
  useEffect(() => {
    // Check if there are messages and if the last one is from the bot
    if (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].sender === 'bot') {
      const lastMessage = chatHistory[chatHistory.length - 1];
      console.log("Processing bot message for questions:", lastMessage.text.substring(0, 100) + "...");
      
      // IMPORTANT FIX: Skip the welcome message from generating quick replies
      if (lastMessage.text.includes("I'm your medical assistant") && 
          lastMessage.text.includes("Feel free to describe your symptoms")) {
        console.log("Skipping welcome message for quick replies");
        setIsQuestionActive(false);
        setParsedQuestions([]);
        return;
      }
      
      // Skip option generation if the message is clearly an educational article
      if (lastMessage.text.split('\n')[0].includes('Understanding') || 
          lastMessage.text.includes('Possible Causes') ||
          lastMessage.text.length > 1000) { // Long educational responses
        console.log("Detected educational article - skipping option generation");
        setIsQuestionActive(false);
        setParsedQuestions([]);
        return;
      }
      
      const questions = parseQuestionsFromMessage(lastMessage.text);
      console.log(`Found ${questions.length} questions for quick replies`);
      
      // Only set interactive questions if:
      // 1. We found structured questions
      // 2. The message doesn't contain diagnostic conclusions, causes, or treatments
      const hasDiagnosticContent = /diagnosis|diagnosed|likely cause|condition appears to be|suffering from|suggests that you have|indicates|confirmed|conclude|caused by|possible causes|common causes|treatment options|remedy for/i.test(lastMessage.text);
      
      if (questions.length > 0 && !hasDiagnosticContent) {
        setParsedQuestions(questions);
        setIsQuestionActive(true);
      } else {
        setIsQuestionActive(false);
        setParsedQuestions([]);
      }
    }
  }, [chatHistory]);

  // Handle quick-reply button click
  const handleQuickReplyClick = (questionIndex, option) => {
    // Update the parsedQuestions state to reflect the selected option
    setParsedQuestions(prevQuestions => {
      const newQuestions = [...prevQuestions];
      newQuestions[questionIndex] = {
        ...newQuestions[questionIndex],
        answered: true,
        answer: option
      };
      return newQuestions;
    });
  };
  
  // Handle custom answer submission
  const handleCustomAnswer = (questionIndex) => {
    if (!customAnswer.trim()) return;
    
    setParsedQuestions(prevQuestions => {
      const newQuestions = [...prevQuestions];
      newQuestions[questionIndex] = {
        ...newQuestions[questionIndex],
        answered: true,
        answer: customAnswer
      };
      return newQuestions;
    });
    
    setCustomAnswer('');
  };
  
  // Send all answered questions as a single message
  const submitAnswers = () => {
    const answeredQuestions = parsedQuestions.filter(q => q.answered);
    if (answeredQuestions.length === 0) return;
    
    const responseText = answeredQuestions.map(q => 
      `${q.question}: ${q.answer}`
    ).join('\n\n');
    
    sendMessage(responseText);
    setIsQuestionActive(false);
    setParsedQuestions([]);
  };
  
  // Render question parser UI if there are parsed questions
  const renderQuestionParser = () => {
    if (!isQuestionActive || parsedQuestions.length === 0) return null;
    
    return (
      <div className="structured-question question-parser-active">
        <h4>Please answer these questions:</h4>
        {parsedQuestions.map((q, qIndex) => (
          <div key={qIndex} className="question-item">
            <p className="question-text">{q.question}</p>
            {!q.answered ? (
              <>
                <div className="quick-reply-container">
                  {q.options.map((option, oIndex) => (
                    <button
                      key={oIndex}
                      className="quick-reply-button"
                      onClick={() => handleQuickReplyClick(qIndex, option)}
                    >
                      {option}
                    </button>
                  ))}
                </div>
                <div className="quick-reply-input">
                  <input
                    type="text"
                    placeholder="Or type a custom answer..."
                    value={customAnswer}
                    onChange={(e) => setCustomAnswer(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleCustomAnswer(qIndex)}
                  />
                  <button onClick={() => handleCustomAnswer(qIndex)}>Send</button>
                </div>
              </>
            ) : (
              <div className="answer-display">
                <p><strong>Your answer:</strong> {q.answer}</p>
              </div>
            )}
          </div>
        ))}
        
        {parsedQuestions.some(q => q.answered) && (
          <button 
            className="submit-answers-button" 
            onClick={submitAnswers}
          >
            Submit Answers
          </button>
        )}
      </div>
    );
  };

  // Add keyboard handler for sending messages with Enter key
  const handleKeyDown = (e) => {
    // Remove the isQuestionActive condition to allow typing anytime
    if (e.key === 'Enter' && !e.shiftKey && !isLoading && message.trim()) {
      e.preventDefault();
      sendMessage(message);
    }
  };
  
  // Add functionality to retry failed messages
  const handleRetryMessage = (index) => {
    // Get the last user message if no specific index provided
    if (index === undefined) {
      // Find the last user message
      for (let i = chatHistory.length - 1; i >= 0; i--) {
        if (chatHistory[i].sender === 'user') {
          index = i;
          break;
        }
      }
    }
    
    if (index !== undefined && chatHistory[index].sender === 'user') {
      sendMessage(chatHistory[index].text);
    }
  };

  // Add styling for action buttons in the info cards
  const actionButtonStyle = {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    width: '100%',
    padding: '10px',
    margin: '8px 0',
    backgroundColor: '#f0f2f5',
    border: '1px solid #dee1e5',
    borderRadius: '8px',
    color: '#333',
    fontWeight: '500',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  };

  const actionIconStyle = {
    fontSize: '18px',
    marginRight: '8px',
  };

  // Modified user profile component with simplified menu
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

  // Add this inside your chat container, right before the chat input
  const renderRetryOption = () => {
    // Only show retry if there's an error or the last message was an error
    if (error || (chatHistory.length > 0 && chatHistory[chatHistory.length - 1].sender === 'error')) {
      return (
        <div className="retry-container">
          <button className="retry-button" onClick={() => handleRetryMessage()}>
            <span className="retry-icon">🔄</span> Retry last message
          </button>
        </div>
      );
    }
    return null;
  };

  // Add specific health topics that will give better rule-based responses
const healthTopics = [
  {
    name: "Headaches",
    examples: [
      "I've had a headache for 2 days",
      "What causes migraines?",
      "How can I prevent tension headaches?"
    ]
  },
  {
    name: "Stomach Issues",
    examples: [
      "I have stomach pain after eating",
      "What causes acid reflux?",
      "How to relieve bloating?"
    ]
  },
  {
    name: "Sleep Problems",
    examples: [
      "I have trouble falling asleep",
      "What causes insomnia?",
      "How to improve sleep quality?"
    ]
  }
];

// Add a component for suggesting similar questions
const renderSuggestions = () => {
  // Only show suggestions when the chat has just started or after an error
  const showSuggestions = 
    (chatHistory.length <= 2) || // Initial conversation state
    (chatHistory.length > 0 && chatHistory[chatHistory.length-1]?.sender === 'error') || // After error
    (error); // When there's an error state
    
  if (!showSuggestions) return null;
  
  return (
    <div className="chat-suggestions">
      <h4>Try asking about:</h4>
      <div className="suggestion-topics">
        {healthTopics.map((topic, topicIndex) => (
          <div key={topicIndex} className="suggestion-topic">
            <h5>{topic.name}</h5>
            <div className="suggestion-examples">
              {topic.examples.map((example, exIndex) => (
                <button 
                  key={exIndex}
                  className="suggestion-button"
                  onClick={() => handleQuickQuestion(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

// Add improved error recovery mechanism
useEffect(() => {
  // Check if we've had multiple consecutive errors
  const lastMessages = chatHistory.slice(-3);
  const consecutiveErrors = lastMessages.filter(msg => 
    msg.sender === 'bot' && msg.isFallback
  ).length >= 2;
  
  if (consecutiveErrors) {
    // Reset conversation state to recover from errors
    setError('The chat assistant is having some trouble. Starting a new conversation might help.');
  }
}, [chatHistory]);

  // Add this component for rendering markdown properly
  const MarkdownContent = ({ content }) => {
    return (
      <div className="markdown-wrapper">
        <ReactMarkdown>{content}</ReactMarkdown>
      </div>
    );
  };

  // Update the return statement to include the UserProfile component in the header
  return (
    <div className="chat-page">
      <header className="chat-header">
        <div className="header-left">
          <button className="back-button" onClick={() => navigate('/mode-selection')}>
            <span className="back-icon">←</span>
          </button>
          <div className="chat-title">
            <h1>General Health Query</h1>
            <p>AI-powered medical assistant</p>
          </div>
        </div>
        <div className="header-right">
          <UserProfile userName={userName} onLogout={onLogout} />
        </div>
      </header>
      
      <main className="chat-main-container">
        {/* Left Column - Health Categories */}
        <div className="health-categories-section">
          <h3>Health Categories</h3>
          <div className="category-grid">
            {healthCategories.map((category) => (
              <div 
                key={category.id} 
                className={`category-item ${activeCategory === category.id ? 'active' : ''}`}
                onClick={() => handleCategorySelect(category.id)}
                style={{
                  '--category-color': category.color
                }}
              >
                <div className="category-icon-wrapper" style={{ backgroundColor: category.color }}>
                  <span className="category-icon">{category.icon}</span>
                </div>
                <span className="category-name">{category.name}</span>
              </div>
            ))}
          </div>
          
          {activeCategory && (
            <div className="category-examples-panel">
              <h4>
                {healthCategories.find(cat => cat.id === activeCategory)?.name} Topics
              </h4>
              <div className="category-examples-list">
                {healthCategories
                  .find(cat => cat.id === activeCategory)
                  ?.examples.map((example, idx) => (
                    <button 
                      key={idx} 
                      className="category-example-button"
                      onClick={() => handleCategoryExample(example)}
                    >
                      {example}
                    </button>
                  ))}
              </div>
            </div>
          )}
        </div>

        {/* Center Column - Chat Content */}
        <div className="chat-main-content">
          <div className="message-list" ref={messageListRef}>
            {chatHistory.map((chat, index) => (
              <div 
                key={index} 
                className={`message ${
                  chat.sender === 'user' 
                    ? 'user-message' 
                    : chat.sender === 'error' 
                      ? 'error-message' 
                      : 'bot-message'
                } ${chat.isFallback ? 'fallback' : ''}`}
              >
                <div className="message-content">
                  {chat.sender === 'bot' ? (
                    <MarkdownContent content={chat.text} />
                  ) : (
                    chat.text
                  )}
                </div>
                <div className="message-time">
                  {formatMessageTime(chat.timestamp)}
                </div>
                {chat.isFallback && (
                  <div className="fallback-indicator" title="Using local AI processing">
                    <span>⚠️</span>
                  </div>
                )}
              </div>
            ))}
            {isLoading && (
              <div className="message bot-message typing">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            
            {/* Render interactive question parser if active */}
            {!isLoading && renderQuestionParser()}
          </div>

          {/* Display suggestions when appropriate */}
          {renderSuggestions()}

          {/* Display retry option when there's an error */}
          {renderRetryOption()}

          <div className="chat-input-container">
            {error && (
              <div className="error-banner">
                {error}
                <button className="close-error" onClick={() => setError('')}>×</button>
              </div>
            )}
            
            <form onSubmit={handleSendMessage} className="message-input">
              <div className="input-wrapper">
                <input
                  type="text"
                  ref={inputRef}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Type your health concern..."
                  disabled={isLoading} // Remove isQuestionActive condition here
                />
                <button 
                  type="button" 
                  className="options-button"
                  onClick={() => setShowOptions(!showOptions)}
                  disabled={isLoading} // Remove isQuestionActive condition here
                >
                  <span>💡</span>
                </button>
              </div>
              <button 
                type="submit" 
                className="send-button" 
                disabled={isLoading || !message.trim()} // Remove isQuestionActive condition here
              >
                <span className="send-icon">↑</span>
              </button>
            </form>
            
            {showOptions && (
              <div className="quick-options">
                <div className="options-header">
                  <span>Quick questions</span>
                  <button onClick={() => setShowOptions(false)}>×</button>
                </div>
                <div className="options-list">
                  {quickQuestions.map((question, index) => (
                    <button 
                      key={index} 
                      className="quick-option"
                      onClick={() => handleQuickQuestion(question)}
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column - Health Info - FIXED STYLING */}
        <div className="health-info-section">
          {/* Health Tip Card - FIXED STYLING */}
          <div className="health-tip-card">
            <div className="tip-header">
              <h3>Health Tip</h3>
              <div className="tip-indicators">
                {healthTips.map((_, index) => (
                  <span 
                    key={index} 
                    className={`tip-indicator ${activeTip === index ? 'active' : ''}`}
                    onClick={() => setActiveTip(index)}
                  ></span>
                ))}
              </div>
            </div>
            <div className="tip-content">
              <div className="tip-icon">{healthTips[activeTip].icon}</div>
              <h4>{healthTips[activeTip].title}</h4>
              <p>{healthTips[activeTip].content}</p>
            </div>
          </div>
          
          {/* Important Reminder Box - FIXED STYLING */}
          <div className="reminder-box">
            <h4>Remember</h4>
            <p>This assistant provides general information only and is not a substitute for professional medical advice.</p>
          </div>
          
          {/* Action Shortcuts - FIXED STYLING */}
          <div className="action-shortcuts">
            <h4>Quick Actions</h4>
            <button 
              className="action-button"
              onClick={() => navigate('/medical-report')}
            >
              <span className="action-icon">📄</span>
              Analyze Medical Report
            </button>
            <button 
              className="action-button"
              onClick={() => window.open('https://www.who.int/health-topics', '_blank')}
            >
              <span className="action-icon">🌐</span>
              WHO Health Resources
            </button>
            <button 
              className="action-button"
              onClick={() => {
                // Clear chat history but keep welcome message
                const welcomeMessage = chatHistory[0];
                setChatHistory([welcomeMessage]);
                setActiveCategory(null);
              }}
            >
              <span className="action-icon">🔄</span>
              Start New Conversation
            </button>
          </div>
        </div>
      </main>
      
      <footer className="chat-footer">
        <div className="footer-content">
          <div className="disclaimer">
            <p>Not a substitute for professional medical advice. Consult a qualified healthcare provider for diagnosis and treatment.</p>
          </div>
          <div className="footer-links">
            <button className="footer-link" onClick={() => window.open('/privacy.html', '_blank')}>Privacy</button>
            <button className="footer-link" onClick={() => window.open('/terms.html', '_blank')}>Terms</button>
            <button className="footer-link" onClick={() => window.open('/help.html', '_blank')}>Help</button>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default GeneralQuery;
