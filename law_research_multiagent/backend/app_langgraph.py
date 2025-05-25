from flask import Flask, request, jsonify, session
from flask_cors import CORS
import google.generativeai as genai
import requests
import json
import os
import re
from datetime import datetime
import sqlite3
from werkzeug.utils import secure_filename
import PyPDF2
from io import BytesIO
import asyncio
import aiohttp
from typing import List, Dict, Any, TypedDict
import urllib.parse
import time
from functools import wraps

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
CORS(app)

GEMINI_API_KEY = "AIzaSyBiV-1p26IFcWeLZHVXXEHRsoae7cTskDs"
genai.configure(api_key=GEMINI_API_KEY)

# Performance timing decorator
def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"⏱️  {func.__name__} executed in {execution_time:.3f} seconds")
        return result, execution_time
    return wrapper

def async_time_it(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"⏱️  {func.__name__} executed in {execution_time:.3f} seconds")
        return result, execution_time
    return wrapper

# LangGraph State Definition
class AgentState(TypedDict):
    messages: List[str]
    user_query: str
    query_type: str
    session_id: str
    paper_content: str
    sources: Dict[str, List[Dict]]
    summary: str
    arguments: List[str]
    final_response: Dict[str, Any]
    chat_history: List[tuple]
    execution_times: Dict[str, float]
    total_start_time: float

class DatabaseManager:
    def __init__(self):
        self.init_db()
   
    def init_db(self):
        conn = sqlite3.connect('law_research.db')
        cursor = conn.cursor()
       
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
       
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_id TEXT UNIQUE NOT NULL,
                session_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
       
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_message TEXT,
                bot_response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                execution_time REAL,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
       
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                filename TEXT,
                content TEXT,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
       
        conn.commit()
        conn.close()
        print("✓ Database initialized successfully")
   
    def create_user(self, username):
        conn = sqlite3.connect('law_research.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username) VALUES (?)', (username,))
            user_id = cursor.lastrowid
            conn.commit()
            print(f"✓ New user created: {username} (ID: {user_id})")
            return user_id
        except sqlite3.IntegrityError:
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            user_id = cursor.fetchone()[0]
            print(f"✓ Existing user found: {username} (ID: {user_id})")
            return user_id
        finally:
            conn.close()
   
    def create_session(self, user_id, session_id, session_name=None):
        conn = sqlite3.connect('law_research.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO sessions (user_id, session_id, session_name) VALUES (?, ?, ?)', (user_id, session_id, session_name))
        conn.commit()
        conn.close()
        print(f"✓ New session created: {session_id} with name: {session_name}")
   
    def get_user_sessions(self, user_id):
        conn = sqlite3.connect('law_research.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT session_id, session_name, created_at FROM sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        sessions = cursor.fetchall()
        conn.close()
        return sessions
   
    def save_chat(self, session_id, user_message, bot_response, execution_time=None):
        conn = sqlite3.connect('law_research.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chat_history (session_id, user_message, bot_response, execution_time)
            VALUES (?, ?, ?, ?)
        ''', (session_id, user_message, bot_response, execution_time))
        conn.commit()
        conn.close()
        print(f"✓ Chat saved for session: {session_id} (Execution time: {execution_time:.3f}s)")
   
    def get_recent_chats(self, session_id, limit=10):
        conn = sqlite3.connect('law_research.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_message, bot_response FROM chat_history
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, limit))
        chats = cursor.fetchall()
        conn.close()
        print(f"✓ Retrieved {len(chats)} recent chats for session: {session_id}")
        return list(reversed(chats))
   
    def save_paper(self, session_id, filename, content):
        conn = sqlite3.connect('law_research.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO papers (session_id, filename, content)
            VALUES (?, ?, ?)
        ''', (session_id, filename, content))
        conn.commit()
        conn.close()
        print(f"✓ Paper saved: {filename} for session: {session_id}")
   
    def get_paper_content(self, session_id):
        conn = sqlite3.connect('law_research.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT content FROM papers
            WHERE session_id = ?
            ORDER BY upload_time DESC
            LIMIT 1
        ''', (session_id,))
        result = cursor.fetchone()
        conn.close()
        if result:
            print(f"✓ Paper content retrieved for session: {session_id}")
        else:
            print(f"✗ No paper found for session: {session_id}")
        return result[0] if result else None

# Initialize database
db = DatabaseManager()

# Initialize LangChain LLM for LangGraph
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

class OptimizedGreetingAgent:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]
        # Cache common greeting responses
        self.cached_responses = {}
   
    def is_greeting(self, message: str):
        message_lower = message.lower().strip()
        is_greeting = any(greeting in message_lower for greeting in self.greetings)
        print(f"✓ Greeting check for '{message}': {is_greeting}")
        return is_greeting
   
    @time_it
    def generate_greeting_response(self):
        # Use cached response if available
        if "default" in self.cached_responses:
            return self.cached_responses["default"]
           
        prompt = """
        Generate a professional, friendly greeting response for a law research assistant.
        Be welcoming and mention the key features: finding sources, summarizing papers, extracting arguments.
        Keep it concise and professional.
        """
        response = self.model.generate_content(prompt)
        self.cached_responses["default"] = response.text
        print("✓ Greeting response generated and cached")
        return response.text

class OptimizedArgumentResearchAgent:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.query_cache = {}
   
    @async_time_it
    async def find_sources(self, argument: str, support_type: str = "both", source_count: int = 5):
        # Check cache first
        cache_key = f"{argument[:100]}_{support_type}_{source_count}"
        if cache_key in self.query_cache:
            print("✓ Using cached sources")
            return self.query_cache[cache_key]
           
        print(f"✓ Finding sources for argument: '{argument[:50]}...' | Type: {support_type} | Count: {source_count}")
       
        # Optimize by running searches concurrently
        search_queries = await self._generate_search_queries(argument)
       
        # Use asyncio.gather for concurrent API calls
        tasks = [self._search_academic_sources(query) for query in search_queries[:3]]  # Limit to 3 for speed
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
       
        sources = []
        for result in search_results:
            if not isinstance(result, Exception):
                sources.extend(result)
       
        filtered_sources = self._filter_valid_sources(sources)
        categorized_sources = self._categorize_sources(filtered_sources, argument, support_type)
        final_sources = self._apply_source_count_limit(categorized_sources, support_type, source_count)
       
        # Cache the result
        self.query_cache[cache_key] = final_sources
       
        print(f"✓ Found {len(final_sources.get('supporting', []))} supporting and {len(final_sources.get('contradicting', []))} contradicting sources")
        return final_sources
   
    def _apply_source_count_limit(self, sources, support_type, count):
        if support_type == "supporting":
            return {'supporting': sources.get('supporting', [])[:count]}
        elif support_type == "contradicting":
            return {'contradicting': sources.get('contradicting', [])[:count]}
        else:
            supporting_count = count // 2
            contradicting_count = count - supporting_count
            return {
                'supporting': sources.get('supporting', [])[:supporting_count],
                'contradicting': sources.get('contradicting', [])[:contradicting_count]
            }
   
    async def _generate_search_queries(self, argument: str):
        prompt = f"""
        Generate 3 specific search queries for finding academic legal sources related to this argument:
        "{argument}"
       
        Return only the search queries, one per line, focused on legal databases and academic sources.
        """
       
        response = self.model.generate_content(prompt)
        queries = [q.strip() for q in response.text.split('\n') if q.strip()]
        print(f"✓ Generated {len(queries[:3])} search queries")
        return queries[:3]  # Reduced from 5 to 3 for speed
   
    async def _search_academic_sources(self, query: str):
        sources = []
       
        # Use timeout for faster responses
        timeout = aiohttp.ClientTimeout(total=5)
       
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Run both searches concurrently
                crossref_task = self._search_crossref(query, session)
                doaj_task = self._search_doaj(query, session)
               
                results = await asyncio.gather(crossref_task, doaj_task, return_exceptions=True)
               
                for result in results:
                    if not isinstance(result, Exception):
                        sources.extend(result)
                       
        except Exception as e:
            print(f"✗ Academic search failed: {e}")
       
        return sources
   
    async def _search_crossref(self, query: str, session):
        url = f"https://api.crossref.org/works?query={urllib.parse.quote(query)}&filter=type:journal-article&rows=5"
       
        try:
            async with session.get(url) as response:
                data = await response.json()
                sources = []
               
                for item in data.get('message', {}).get('items', []):
                    if 'ISSN' in item:
                        source = {
                            'title': item.get('title', [''])[0],
                            'authors': [author.get('given', '') + ' ' + author.get('family', '')
                                      for author in item.get('author', [])],
                            'journal': item.get('container-title', [''])[0],
                            'year': item.get('published-print', {}).get('date-parts', [[None]])[0][0],
                            'doi': item.get('DOI', ''),
                            'issn': item.get('ISSN', []),
                            'url': item.get('URL', ''),
                            'abstract': item.get('abstract', ''),
                            'type': 'academic'
                        }
                        sources.append(source)
               
                print(f"✓ Found {len(sources)} sources from CrossRef")
                return sources
        except Exception as e:
            print(f"✗ CrossRef search failed: {e}")
            return []
   
    async def _search_doaj(self, query: str, session):
        url = f"https://doaj.org/api/v2/search/articles/{urllib.parse.quote(query)}?pageSize=5"
       
        try:
            async with session.get(url) as response:
                data = await response.json()
                sources = []
               
                for item in data.get('results', []):
                    bibjson = item.get('bibjson', {})
                    source = {
                        'title': bibjson.get('title', ''),
                        'authors': [author.get('name', '') for author in bibjson.get('author', [])],
                        'journal': bibjson.get('journal', {}).get('title', ''),
                        'year': bibjson.get('year'),
                        'issn': [bibjson.get('journal', {}).get('issn', [''])[0]],
                        'url': bibjson.get('link', [{}])[0].get('url', ''),
                        'abstract': bibjson.get('abstract', ''),
                        'type': 'academic'
                    }
                    sources.append(source)
               
                print(f"✓ Found {len(sources)} sources from DOAJ")
                return sources
        except Exception as e:
            print(f"✗ DOAJ search failed: {e}")
            return []
   
    def _filter_valid_sources(self, sources):
        valid_sources = []
        for source in sources:
            if source.get('issn') and any(issn for issn in source['issn'] if self._validate_issn(issn)):
                valid_sources.append(source)
        print(f"✓ Filtered to {len(valid_sources)} valid sources")
        return valid_sources
   
    def _validate_issn(self, issn: str):
        if not issn:
            return False
       
        issn = re.sub(r'[^\dX]', '', issn.upper())
       
        if len(issn) != 8:
            return False
       
        check_sum = 0
        for i, digit in enumerate(issn[:-1]):
            if digit == 'X':
                return False
            check_sum += int(digit) * (8 - i)
       
        check_digit = issn[-1]
        expected_check = (11 - (check_sum % 11)) % 11
       
        if expected_check == 10:
            return check_digit == 'X'
        else:
            return check_digit == str(expected_check)
   
    def _categorize_sources(self, sources, argument: str, support_type: str):
        if not sources:
            return {'supporting': [], 'contradicting': []}
           
        prompt = f"""
        Categorize these sources as "supporting" or "contradicting" the argument: "{argument}"
       
        Sources to categorize:
        {json.dumps([{'title': s['title'], 'abstract': s['abstract'][:200]} for s in sources[:10]], indent=2)}
       
        Return a JSON object with two arrays: "supporting" and "contradicting", containing the indices of sources.
        """
       
        try:
            response = self.model.generate_content(prompt)
            categorization = json.loads(response.text)
           
            result = {
                'supporting': [sources[i] for i in categorization.get('supporting', []) if i < len(sources)],
                'contradicting': [sources[i] for i in categorization.get('contradicting', []) if i < len(sources)]
            }
           
            print(f"✓ Categorized {len(result['supporting'])} supporting and {len(result['contradicting'])} contradicting sources")
            return result
               
        except Exception as e:
            print(f"✗ Categorization failed, using fallback: {e}")
            return {'supporting': sources[:len(sources)//2], 'contradicting': sources[len(sources)//2:]}

class OptimizedPaperAnalysisAgent:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.summary_cache = {}
   
    def extract_text_from_pdf(self, file_content):
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            # Limit to first 50 pages for speed
            for i, page in enumerate(pdf_reader.pages[:50]):
                text += page.extract_text() + "\n"
            print(f"✓ Extracted text from PDF: {len(text)} characters")
            return text
        except Exception as e:
            print(f"✗ PDF extraction failed: {e}")
            return None
   
    @time_it
    def summarize_paper(self, content: str, summary_type: str = "comprehensive"):
        # Check cache first
        cache_key = f"{content[:100]}_{summary_type}"
        if cache_key in self.summary_cache:
            print("✓ Using cached summary")
            return self.summary_cache[cache_key]
           
        print(f"✓ Summarizing paper: {summary_type} summary")
       
        # Limit content length for faster processing
        max_content = 6000 if summary_type == "brief" else 12000
        content = content[:max_content]
       
        if summary_type == "brief":
            prompt = f"""
            Provide a brief 2-3 sentence summary of this legal paper:
           
            {content}
            """
        else:
            prompt = f"""
            Provide a comprehensive summary of this legal paper including:
            1. Main arguments
            2. Key legal principles
            3. Conclusions
            4. Methodology (if applicable)
           
            Paper content:
            {content}
            """
       
        response = self.model.generate_content(prompt)
        self.summary_cache[cache_key] = response.text
        print("✓ Paper summary generated and cached")
        return response.text
   
    @time_it
    def extract_arguments(self, content: str):
        print("✓ Extracting arguments from paper")
       
        # Limit content for faster processing
        content = content[:8000]
       
        prompt = f"""
        Extract the main legal arguments from this paper. List them clearly:
       
        {content}
       
        Return a JSON array of arguments.
        """
       
        try:
            response = self.model.generate_content(prompt)
            arguments = json.loads(response.text)
            print(f"✓ Extracted {len(arguments)} arguments")
            return arguments
        except Exception as e:
            print(f"✗ JSON parsing failed, using fallback: {e}")
            response = self.model.generate_content(f"List the main arguments from this legal paper: {content}")
            arguments = [arg.strip() for arg in response.text.split('\n') if arg.strip()]
            print(f"✓ Extracted {len(arguments)} arguments (fallback)")
            return arguments

class OptimizedContextualAgent:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
   
    @time_it
    def restructure_question(self, current_question: str, chat_history: List[tuple]):
        # Skip restructuring for very short history to save time
        if len(chat_history) < 2:
            return current_question
           
        print(f"✓ Restructuring question based on chat history")
        context = self._build_context(chat_history)
       
        prompt = f"""
        Based on the conversation history, restructure this question to be more specific and contextual:
       
        Current question: "{current_question}"
       
        Conversation context:
        {context}
       
        Return a more specific, contextual version of the question that incorporates relevant information from the chat history.
        """
       
        response = self.model.generate_content(prompt)
        restructured = response.text.strip()
        print(f"✓ Question restructured: '{current_question}' -> '{restructured}'")
        return restructured
   
    def _build_context(self, chat_history: List[tuple]):
        context_parts = []
        # Limit to last 5 exchanges for speed
        for user_msg, bot_response in chat_history[-5:]:
            context_parts.append(f"User: {user_msg}")
            context_parts.append(f"Assistant: {bot_response[:150]}...")
       
        return "\n".join(context_parts)

# Initialize optimized agents
greeting_agent = OptimizedGreetingAgent()
research_agent = OptimizedArgumentResearchAgent()
paper_agent = OptimizedPaperAnalysisAgent()
context_agent = OptimizedContextualAgent()

# LangGraph Node Functions
def greeting_node(state: AgentState) -> AgentState:
    start_time = time.time()
   
    if greeting_agent.is_greeting(state["user_query"]):
        response, exec_time = greeting_agent.generate_greeting_response()
        state["final_response"] = {
            'type': 'general',
            'message': response
        }
   
    state["execution_times"]["greeting"] = time.time() - start_time
    return state

def context_node(state: AgentState) -> AgentState:
    start_time = time.time()
   
    if state["chat_history"] and state["query_type"] == 'general':
        restructured, exec_time = context_agent.restructure_question(state["user_query"], state["chat_history"])
        state["user_query"] = restructured
   
    state["execution_times"]["context"] = time.time() - start_time
    return state

def source_research_node(state: AgentState) -> AgentState:
    start_time = time.time()
   
    if state["query_type"] == 'find_sources':
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
       
        # Extract parameters from the query (this would be better handled through proper state management)
        support_type = "both"  # Default, should be passed through state
        source_count = 5       # Default, should be passed through state
       
        sources, exec_time = loop.run_until_complete(
            research_agent.find_sources(state["user_query"], support_type, source_count)
        )
       
        state["sources"] = sources
        supporting_count = len(sources.get('supporting', []))
        contradicting_count = len(sources.get('contradicting', []))
       
        state["final_response"] = {
            'type': 'sources',
            'sources': sources,
            'message': f"Found {supporting_count} supporting and {contradicting_count} contradicting sources"
        }
   
    state["execution_times"]["source_research"] = time.time() - start_time
    return state

def paper_analysis_node(state: AgentState) -> AgentState:
    start_time = time.time()
   
    if state["query_type"] in ['summarize', 'extract_arguments']:
        if not state["paper_content"]:
            state["final_response"] = {'error': 'No paper uploaded for this session'}
        else:
            if state["query_type"] == 'summarize':
                summary_type = "comprehensive"  # Should be passed through state
                summary, exec_time = paper_agent.summarize_paper(state["paper_content"], summary_type)
                state["summary"] = summary
                state["final_response"] = {
                    'type': 'summary',
                    'summary': summary
                }
           
            elif state["query_type"] == 'extract_arguments':
                arguments, exec_time = paper_agent.extract_arguments(state["paper_content"])
                state["arguments"] = arguments
                state["final_response"] = {
                    'type': 'arguments',
                    'arguments': arguments
                }
   
    state["execution_times"]["paper_analysis"] = time.time() - start_time
    return state

def general_response_node(state: AgentState) -> AgentState:
    start_time = time.time()
   
    if state["query_type"] == 'general' and "final_response" not in state:
        model = genai.GenerativeModel('gemini-1.5-flash')
        ai_response = model.generate_content(state["user_query"])
        state["final_response"] = {
            'type': 'general',
            'message': ai_response.text
        }
   
    state["execution_times"]["general_response"] = time.time() - start_time
    return state

def route_query(state: AgentState) -> str:
    """Router function to determine which path to take"""
    if greeting_agent.is_greeting(state["user_query"]):
        return "greeting"
    elif state["query_type"] == 'find_sources':
        return "source_research"
    elif state["query_type"] in ['summarize', 'extract_arguments']:
        return "paper_analysis"
    else:
        return "context"

# Build LangGraph workflow
def create_workflow():
    workflow = StateGraph(AgentState)
   
    # Add nodes
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("context", context_node)
    workflow.add_node("source_research", source_research_node)
    workflow.add_node("paper_analysis", paper_analysis_node)
    workflow.add_node("general_response", general_response_node)
   
    # Add conditional routing
    workflow.add_conditional_edges(
        "greeting",
        lambda x: "end" if "final_response" in x else "context",
        {
            "end": END,
            "context": "context"
        }
    )
   
    workflow.add_conditional_edges(
        "context",
        route_query,
        {
            "greeting": "greeting",
            "source_research": "source_research",
            "paper_analysis": "paper_analysis",
            "context": "general_response"
        }
    )
   
    workflow.add_edge("source_research", END)
    workflow.add_edge("paper_analysis", END)
    workflow.add_edge("general_response", END)
   
    # Set entry point
    workflow.set_entry_point("greeting")
   
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
   
    return app

# Initialize workflow
workflow_app = create_workflow()

# Flask Routes
@app.route('/api/init_session', methods=['POST'])
def init_session():
    data = request.json
    username = data.get('username', 'anonymous')
    session_id = data.get('session_id')
    session_name = data.get('session_name', f'Session {datetime.now().strftime("%Y-%m-%d %H:%M")}')
   
    print(f"✓ Initializing session: {session_id} for user: {username}")
    user_id = db.create_user(username)
    db.create_session(user_id, session_id, session_name)
   
    return jsonify({'status': 'success', 'session_id': session_id, 'session_name': session_name})

@app.route('/api/get_sessions', methods=['GET'])
def get_sessions():
    username = request.args.get('username', 'anonymous')
    user_id = db.create_user(username)
    sessions = db.get_user_sessions(user_id)
   
    session_list = []
    for session_id, session_name, created_at in sessions:
        session_list.append({
            'session_id': session_id,
            'session_name': session_name or f'Session {created_at}',
            'created_at': created_at
        })
   
    print(f"✓ Retrieved {len(session_list)} sessions for user: {username}")
    return jsonify({'sessions': session_list})

@app.route('/api/upload_paper', methods=['POST'])
def upload_paper():
    print("✓ File upload request received")
    if 'file' not in request.files:
        print("✗ No file in request")
        return jsonify({'error': 'No file uploaded'}), 400
   
    file = request.files['file']
    session_id = request.form.get('session_id')
   
    if file.filename == '':
        print("✗ No file selected")
        return jsonify({'error': 'No file selected'}), 400
   
    filename = secure_filename(file.filename)
    file_content = file.read()
    print(f"✓ Processing file: {filename} ({len(file_content)} bytes)")
   
    if filename.endswith('.pdf'):
        text_content = paper_agent.extract_text_from_pdf(file_content)
        if not text_content:
            return jsonify({'error': 'Could not extract text from PDF'}), 400
    else:
        text_content = file_content.decode('utf-8', errors='ignore')
   
    db.save_paper(session_id, filename, text_content)
   
    return jsonify({
        'status': 'success',
        'filename': filename,
        'content_preview': text_content[:500] + '...' if len(text_content) > 500 else text_content
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    total_start_time = time.time()
   
    data = request.json
    user_message = data.get('message')
    session_id = data.get('session_id')
    query_type = data.get('type', 'general')
    support_type = data.get('support_type', 'both')
    source_count = data.get('source_count', 5)
    summary_type = data.get('summary_type', 'comprehensive')
   
    print(f"🚀 Chat request started: '{user_message}' | Type: {query_type} | Session: {session_id}")
   
    # Prepare state for LangGraph
    chat_history = db.get_recent_chats(session_id)
    paper_content = db.get_paper_content(session_id) if query_type in ['summarize', 'extract_arguments'] else None
   
    initial_state = AgentState(
        messages=[],
        user_query=user_message,
        query_type=query_type,
        session_id=session_id,
        paper_content=paper_content or "",
        sources={},
        summary="",
        arguments=[],
        final_response={},
        chat_history=chat_history,
        execution_times={},
        total_start_time=total_start_time
    )
   
    # Execute workflow
    try:
        config = {"configurable": {"thread_id": session_id}}
        result = workflow_app.invoke(initial_state, config)
       
        # Calculate total execution time
        total_execution_time = time.time() - total_start_time
       
        # Add execution time info to response
        response = result["final_response"]
        response["execution_time"] = total_execution_time
        response["execution_breakdown"] = result["execution_times"]
       
        print(f"⏱️  Total execution time: {total_execution_time:.3f} seconds")
        print(f"📊 Execution breakdown: {result['execution_times']}")
       
        # Save to database with execution time
        response_text = json.dumps(response) if isinstance(response, dict) else str(response)
        db.save_chat(session_id, user_message, response_text, total_execution_time)
       
        return jsonify(response)
       
    except Exception as e:
        error_time = time.time() - total_start_time
        print(f"✗ Error in workflow execution ({error_time:.3f}s): {e}")
       
        # Fallback to direct agent calls
        return fallback_chat_handler(data, session_id, user_message, query_type, total_start_time)

def fallback_chat_handler(data, session_id, user_message, query_type, start_time):
    """Fallback handler if LangGraph fails"""
    print("🔄 Using fallback chat handler")
   
    try:
        if greeting_agent.is_greeting(user_message):
            response, exec_time = greeting_agent.generate_greeting_response()
            result = {
                'type': 'general',
                'message': response,
                'execution_time': time.time() - start_time
            }
       
        elif query_type == 'find_sources':
            support_type = data.get('support_type', 'both')
            source_count = data.get('source_count', 5)
           
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            sources, exec_time = loop.run_until_complete(
                research_agent.find_sources(user_message, support_type, source_count)
            )
           
            supporting_count = len(sources.get('supporting', []))
            contradicting_count = len(sources.get('contradicting', []))
           
            result = {
                'type': 'sources',
                'sources': sources,
                'message': f"Found {supporting_count} supporting and {contradicting_count} contradicting sources",
                'execution_time': time.time() - start_time
            }
       
        elif query_type == 'summarize':
            paper_content = db.get_paper_content(session_id)
            if not paper_content:
                result = {'error': 'No paper uploaded for this session'}
            else:
                summary_type = data.get('summary_type', 'comprehensive')
                summary, exec_time = paper_agent.summarize_paper(paper_content, summary_type)
                result = {
                    'type': 'summary',
                    'summary': summary,
                    'execution_time': time.time() - start_time
                }
       
        elif query_type == 'extract_arguments':
            paper_content = db.get_paper_content(session_id)
            if not paper_content:
                result = {'error': 'No paper uploaded for this session'}
            else:
                arguments, exec_time = paper_agent.extract_arguments(paper_content)
                result = {
                    'type': 'arguments',
                    'arguments': arguments,
                    'execution_time': time.time() - start_time
                }
       
        else:
            chat_history = db.get_recent_chats(session_id)
            if chat_history and query_type == 'general':
                restructured, exec_time = context_agent.restructure_question(user_message, chat_history)
                user_message = restructured
           
            model = genai.GenerativeModel('gemini-1.5-flash')
            ai_response = model.generate_content(user_message)
            result = {
                'type': 'general',
                'message': ai_response.text,
                'execution_time': time.time() - start_time
            }
       
        # Save to database
        response_text = json.dumps(result) if isinstance(result, dict) else str(result)
        db.save_chat(session_id, data.get('message'), response_text, result.get('execution_time', 0))
       
        print(f"⏱️  Fallback execution time: {result.get('execution_time', 0):.3f} seconds")
        return jsonify(result)
       
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"✗ Fallback also failed ({execution_time:.3f}s): {e}")
        return jsonify({
            'error': 'Processing failed',
            'execution_time': execution_time
        }), 500

@app.route('/api/chat_history', methods=['GET'])
def get_chat_history():
    session_id = request.args.get('session_id')
    chat_history = db.get_recent_chats(session_id, 50)
   
    formatted_history = []
    for user_msg, bot_response in chat_history:
        formatted_history.append({
            'type': 'user',
            'message': user_msg
        })
        try:
            bot_data = json.loads(bot_response)
            formatted_history.append({
                'type': 'bot',
                'data': bot_data
            })
        except:
            formatted_history.append({
                'type': 'bot',
                'message': bot_response
            })
   
    print(f"✓ Retrieved chat history: {len(formatted_history)} messages")
    return jsonify({'history': formatted_history})

@app.route('/api/performance_stats', methods=['GET'])
def get_performance_stats():
    """New endpoint to get performance statistics"""
    session_id = request.args.get('session_id')
   
    conn = sqlite3.connect('law_research.db')
    cursor = conn.cursor()
   
    # Get average execution times
    cursor.execute('''
        SELECT AVG(execution_time), COUNT(*), MIN(execution_time), MAX(execution_time)
        FROM chat_history
        WHERE session_id = ? AND execution_time IS NOT NULL
    ''', (session_id,))
   
    stats = cursor.fetchone()
    conn.close()
   
    if stats and stats[0]:
        return jsonify({
            'average_response_time': round(stats[0], 3),
            'total_queries': stats[1],
            'fastest_response': round(stats[2], 3),
            'slowest_response': round(stats[3], 3)
        })
    else:
        return jsonify({
            'average_response_time': 0,
            'total_queries': 0,
            'fastest_response': 0,
            'slowest_response': 0
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with performance info"""
    start_time = time.time()
   
    # Test database connection
    try:
        conn = sqlite3.connect('law_research.db')
        conn.close()
        db_status = "healthy"
    except Exception as e:
        db_status = f"error: {e}"
   
    # Test Gemini API
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        model.generate_content("Hello")
        api_status = "healthy"
    except Exception as e:
        api_status = f"error: {e}"
   
    response_time = time.time() - start_time
   
    return jsonify({
        'status': 'healthy',
        'database': db_status,
        'gemini_api': api_status,
        'response_time': round(response_time, 3),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Optimized Law Research Assistant with LangGraph...")
    print("📊 Performance monitoring enabled")
    print("🔧 Multi-agent workflow optimized for speed")
    app.run(debug=True, port=5000)

