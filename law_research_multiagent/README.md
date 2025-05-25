# Law Research Multi-Agent System

A comprehensive AI-powered legal research assistant that combines multiple specialized agents to help with legal document analysis, argument extraction, source finding, and research assistance.

##  Features

- **Multi-Agent Architecture**: Specialized agents for different tasks
  - **Greeting Agent**: Handles user interactions and welcomes
  - **Argument Research Agent**: Finds supporting and contradicting academic sources
  - **Paper Analysis Agent**: PDF processing, summarization, and argument extraction
  - **Contextual Agent**: Maintains conversation context and enhances queries

- **Core Capabilities**:
  - PDF document upload and text extraction
  - Academic source discovery (CrossRef, DOAJ integration)
  - Intelligent document summarization
  - Legal argument extraction
  - Context-aware conversational interface
  - Session management and chat history
  - Real-time source categorization (supporting/contradicting)
  - Uses minimum LLM calls (2-3 calls based on query)
  - It takes less time for answering (less than 10sec)
  - Used Gemini llm, Ollama (open source llm models)

##  Project Structure

```
law_research_multiagent/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── app_langgraph.py      # Alternative LangGraph implementation
│   ├── requirements.txt       # Python dependencies
│   ├── law_research.db       # SQLite database (auto-generated)
│   └── venv/                 # Virtual environment
├── frontend/
│   └── index.html            # Web interface (HTML/CSS/JavaScript)
├── run.py                    # Project launcher script
└── README.md                 # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+ installed on your system
- Git (for cloning the repository)
- Internet connection (for API calls and academic source searches)

### Step 1: Clone the Repository

```bash
git clone <link>
cd law_research_multiagent
```

### Step 2: Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys**:
   - Open `app.py` in a text editor
   - Replace the Gemini API key with your own:
   ```python
   GEMINI_API_KEY = "your-actual-api-key-here"
   ```
   - Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

5. **Initialize the database**:
   The SQLite database will be automatically created when you first run the application.

### Step 3: Start the Application

#### Option 1: Using the run script (Recommended)
```bash
# From the project root directory
python run.py
```

#### Option 2: Manual startup

**Terminal 1 - Start Backend**:
```bash
cd backend
python app.py
```
The backend will start on `http://localhost:5000`

**Terminal 2 - Start Frontend**:
```bash
cd frontend
python -m http.server 8080
```
The frontend will be available at `http://localhost:8080`

##  Configuration

### API Configuration

The application uses Google's Gemini API. You'll need to:

1. Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Replace the API key in `backend/app.py`:
   ```python
   GEMINI_API_KEY = "your-actual-gemini-api-key"
   ```

### Database Configuration

The application uses SQLite for data persistence. The database file (`law_research.db`) is automatically created in the backend directory and includes tables for:
- Users and sessions
- Chat history
- Uploaded papers
- Research data

##  Usage Guide

### Getting Started

1. **Access the Application**: Open your browser and go to `http://localhost:8080`

2. **Create a Session**: 
   - Enter your username
   - The system will create a new research session

3. **Upload Documents** (Optional):
   - Click the upload button
   - Select PDF files for analysis
   - The system will extract text and make it available for analysis

### Core Features

#### 1. General Chat
- Ask general legal questions
- Get AI-powered responses with context awareness

#### 2. Document Analysis
- **Summarize**: Get comprehensive or brief summaries of uploaded documents
- **Extract Arguments**: Identify key legal arguments from papers

#### 3. Source Research
- **Find Sources**: Discover academic sources that support or contradict arguments
- **Filter Options**: Choose supporting, contradicting, or both types of sources
- **Source Count**: Specify how many sources to retrieve (1-20)

#### 4. Session Management
- **Multiple Sessions**: Create and switch between different research sessions
- **Chat History**: Access previous conversations and research
- **Persistent Data**: All data is saved and retrievable

### Example Workflow

1. Upload a legal document (PDF)
2. Ask for a summary: "Please summarize this paper"
3. Extract arguments: "What are the main legal arguments?"
4. Find supporting sources: "Find sources supporting the first argument"
5. Continue research with context-aware follow-up questions

##  API Endpoints

### Core Endpoints

- `POST /api/init_session` - Initialize a new research session
- `GET /api/get_sessions` - Retrieve user sessions
- `POST /api/chat` - Main chat interface
- `POST /api/upload_paper` - Upload PDF documents
- `GET /api/chat_history` - Retrieve conversation history

### Chat Types

The `/api/chat` endpoint supports different query types:
- `general` - General conversation
- `find_sources` - Academic source discovery
- `summarize` - Document summarization
- `extract_arguments` - Argument extraction

##  External Integrations

### Academic Databases
- **CrossRef API**: Access to millions of academic papers
- **DOAJ (Directory of Open Access Journals)**: Open access academic sources

### AI Services
- **Google Gemini API**: Advanced language model for text processing
- **Alternative**: Supports Ollama for local/open-source models

##  Development

### Project Dependencies

Key Python packages:
```
Flask==2.3.3
Flask-CORS==4.0.0
google-generativeai==0.3.1
PyPDF2==3.0.1
scikit-learn==1.3.0
aiohttp==3.8.5
requests==2.31.0
```

### Database Schema

The application uses SQLite with the following tables:
- `users` - User management
- `sessions` - Research sessions
- `chat_history` - Conversation logs
- `papers` - Uploaded document content

### Adding New Features

The modular agent architecture makes it easy to add new capabilities:

1. **Create a new agent class** in `app.py`
2. **Add corresponding API endpoints** 
3. **Update the frontend** to use new features
4. **Add new query types** to the chat system

##  Troubleshooting

### Common Issues

**Backend won't start**:
- Check if Python virtual environment is activated
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Ensure API key is properly configured

**Frontend can't connect to backend**:
- Verify backend is running on port 5000
- Check for CORS issues in browser console
- Ensure both frontend and backend are running

**Database errors**:
- Delete `law_research.db` to reset database
- Check file permissions in backend directory

**API rate limits**:
- Monitor Gemini API usage
- Implement request throttling if needed

**PDF processing fails**:
- Ensure uploaded files are valid PDFs
- Check file size limits
- Verify PyPDF2 installation

### Performance Optimization

- **Large documents**: The system processes first 8000 characters for analysis
- **Concurrent searches**: Uses async operations for faster source discovery
- **Caching**: Consider implementing Redis for session caching in production

##  Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -m 'Add feature description'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the API documentation for integration details

##  Future Enhancements

- **Multi-language support** for international legal research
- **Advanced citation formatting** (APA, MLA, Chicago)
- **Collaborative research** features for team work
- **Integration with legal databases** (Westlaw, LexisNexis)
- **Export functionality** for research reports
- **Advanced analytics** and research insights

---


