# AccuMediq: AI-Powered Medical Assistant

A React and Flask-based application that provides AI-powered medical assistance through general health queries and medical report analysis.

## Project Structure

```
/ma
|-- backend/           # Flask backend
|   |-- app.py         # Main Flask application
|   |-- requirements.txt # Python dependencies
|   |-- uploads/       # Directory for uploaded files
|
|-- frontend/          # React frontend
|   |-- public/        # Public assets
|   |   |-- landing.jpg # Landing page image
|   |
|   |-- src/           # React source code
|       |-- components/ # React components
|       |-- App.js     # Main React application
|       |-- index.js   # React entry point
```

## Setup Instructions

### Backend Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Navigate to the backend directory:
   ```bash
   cd backend
   ```

3. Start the Flask server:
   ```bash
   python app.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. **Important**: Make sure the landing.jpg file is in the public folder:
   If the file is not there, copy it manually from the original location to `/frontend/public/landing.jpg`

4. Start the React development server:
   ```bash
   npm start
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

## Login Information

Use these credentials for testing:

- Username: test
- Password: test123

## Features

- **Landing Page**: Information about the application
- **Authentication**: User login and registration
- **Mode Selection**: Choose between General Health Query and Medical Report Analysis
- **General Health Query**: Chat with an AI assistant about health concerns
- **Medical Report Analysis**: Upload medical reports for AI analysis

## Technologies Used

- **Frontend**: React, React Router, Axios
- **Backend**: Flask, Flask-CORS
- **AI Services**: Deepseek R1

## Notes

- This is a development build and not intended for production use
- All AI responses include appropriate medical disclaimers
- The application includes fallback mechanisms when AI services are unavailable
