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
from typing import List, Dict, Any
import urllib.parse

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
CORS(app)

GEMINI_API_KEY = "AIzaSyBiV-1p26IFcWeLZHVXXEHRsoae7cTskDs"
genai.configure(api_key=GEMINI_API_KEY)


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
        cursor.execute('INSERT INTO sessions (user_id, session_id, session_name) VALUES (?, ?, ?)',
                       (user_id, session_id, session_name))
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

    def save_chat(self, session_id, user_message, bot_response):
        conn = sqlite3.connect('law_research.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chat_history (session_id, user_message, bot_response)
            VALUES (?, ?, ?)
        ''', (session_id, user_message, bot_response))
        conn.commit()
        conn.close()
        print(f"✓ Chat saved for session: {session_id}")

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


db = DatabaseManager()









import requests
import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
import logging

class OllamaClient:
    """
    Ollama client for interacting with local Ollama models
    Supports both synchronous and asynchronous operations
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            model: Model name (default: llama3.2:3b for lighter weight)
                   Other options: llama3.2:1b, llama3.1:8b, mistral:7b, codellama:7b
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        
        # Test connection on initialization
        self._test_connection()
    
    def _test_connection(self):
        """Test if Ollama server is accessible"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✓ Ollama server connected at {self.base_url}")
                available_models = [model['name'] for model in response.json().get('models', [])]
                if self.model in available_models:
                    print(f"✓ Model '{self.model}' is available")
                else:
                    print(f"⚠ Model '{self.model}' not found. Available models: {available_models}")
                    if available_models:
                        self.model = available_models[0]
                        print(f"✓ Switched to available model: {self.model}")
            else:
                print(f"✗ Ollama server responded with status {response.status_code}")
        except Exception as e:
            print(f"✗ Cannot connect to Ollama server: {e}")
            print("Make sure Ollama is running: 'ollama serve'")
    
    def generate_content(self, prompt: str, system_prompt: str = None, **kwargs) -> 'OllamaResponse':
        """
        Generate content using Ollama model (synchronous)
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            OllamaResponse object with .text property
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', 0.7),
                    "num_predict": kwargs.get('max_tokens', 1000),
                    "top_p": kwargs.get('top_p', 0.9),
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=kwargs.get('timeout', 60)
            )
            
            if response.status_code == 200:
                result = response.json()
                return OllamaResponse(result.get('response', ''))
            else:
                print(f"✗ Ollama API error: {response.status_code} - {response.text}")
                return OllamaResponse(f"Error: Failed to generate response (Status: {response.status_code})")
                
        except Exception as e:
            print(f"✗ Ollama generation error: {e}")
            return OllamaResponse(f"Error: {str(e)}")
    
    async def generate_content_async(self, prompt: str, system_prompt: str = None, **kwargs) -> 'OllamaResponse':
        """
        Generate content using Ollama model (asynchronous)
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', 0.7),
                    "num_predict": kwargs.get('max_tokens', 1000),
                    "top_p": kwargs.get('top_p', 0.9),
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=kwargs.get('timeout', 60))
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return OllamaResponse(result.get('response', ''))
                    else:
                        error_text = await response.text()
                        print(f"✗ Ollama API error: {response.status} - {error_text}")
                        return OllamaResponse(f"Error: Failed to generate response (Status: {response.status})")
                        
        except Exception as e:
            print(f"✗ Ollama async generation error: {e}")
            return OllamaResponse(f"Error: {str(e)}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> 'OllamaResponse':
        """
        Chat completion using Ollama model
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     Example: [{"role": "user", "content": "Hello"}]
        """
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', 0.7),
                    "num_predict": kwargs.get('max_tokens', 1000),
                    "top_p": kwargs.get('top_p', 0.9),
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=kwargs.get('timeout', 60)
            )
            
            if response.status_code == 200:
                result = response.json()
                message_content = result.get('message', {}).get('content', '')
                return OllamaResponse(message_content)
            else:
                print(f"✗ Ollama chat error: {response.status_code} - {response.text}")
                return OllamaResponse(f"Error: Failed to generate chat response (Status: {response.status_code})")
                
        except Exception as e:
            print(f"✗ Ollama chat error: {e}")
            return OllamaResponse(f"Error: {str(e)}")
    
    def pull_model(self, model_name: str = None) -> bool:
        """
        Pull/download a model from Ollama repository
        
        Args:
            model_name: Model to pull (if None, pulls the current model)
        
        Returns:
            bool: True if successful, False otherwise
        """
        model_to_pull = model_name or self.model
        
        try:
            print(f"🔄 Pulling model: {model_to_pull}")
            payload = {"name": model_to_pull}
            
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=300  # 5 minutes timeout for model download
            )
            
            if response.status_code == 200:
                print(f"✓ Model {model_to_pull} pulled successfully")
                return True
            else:
                print(f"✗ Failed to pull model: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"✗ Model pull error: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            print(f"✗ Error listing models: {e}")
            return []
    
    def switch_model(self, model_name: str):
        """Switch to a different model"""
        available_models = self.list_models()
        if model_name in available_models:
            self.model = model_name
            print(f"✓ Switched to model: {model_name}")
        else:
            print(f"✗ Model {model_name} not available. Available models: {available_models}")
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """Get information about a model"""
        model_to_check = model_name or self.model
        
        try:
            payload = {"name": model_to_check}
            response = self.session.post(f"{self.base_url}/api/show", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Model {model_to_check} not found"}
                
        except Exception as e:
            return {"error": str(e)}


# class OllamaResponse:
#     """Response wrapper to mimic Gemini API response structure"""
    
#     def __init__(self, text: str):
#         self.text = text
    
#     def __str__(self):
#         return self.text


# Utility class for model recommendations
class OllamaModelManager:
    """Helper class for managing Ollama models based on use case"""
    
    LIGHTWEIGHT_MODELS = [
        "llama3.2:1b",    # ~1.3GB - Very fast, good for simple tasks
        "llama3.2:3b",    # ~2.0GB - Good balance of speed and capability
        "phi3:3.8b",      # ~2.3GB - Microsoft's efficient model
    ]
    
    BALANCED_MODELS = [
        "llama3.1:8b",    # ~4.7GB - Good performance for most tasks
        "mistral:7b",     # ~4.1GB - Good for reasoning tasks
        "gemma2:9b",      # ~5.4GB - Google's efficient model
    ]
    
    POWERFUL_MODELS = [
        "llama3.1:70b",   # ~40GB - Very capable but resource intensive
        "codellama:13b",  # ~7.3GB - Good for code-related tasks
        "mixtral:8x7b",   # ~26GB - Mixture of experts model
    ]
    
    @classmethod
    def recommend_model(cls, use_case: str = "general", resource_level: str = "light") -> str:
        """
        Recommend a model based on use case and available resources
        
        Args:
            use_case: "general", "legal", "code", "reasoning"
            resource_level: "light", "balanced", "powerful"
        
        Returns:
            str: Recommended model name
        """
        if resource_level == "light":
            models = cls.LIGHTWEIGHT_MODELS
        elif resource_level == "balanced":
            models = cls.BALANCED_MODELS
        else:
            models = cls.POWERFUL_MODELS
        
        # Use case specific recommendations
        if use_case == "legal":
            return models[0] if resource_level == "light" else "llama3.1:8b"
        elif use_case == "code":
            return "codellama:7b" if resource_level != "light" else models[0]
        elif use_case == "reasoning":
            return "mistral:7b" if resource_level != "light" else models[1] if len(models) > 1 else models[0]
        
        return models[0]  # Default recommendation



class GreetingAgent:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        # self.model = OllamaClient(model="llama3.2:3b")
        # self.greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]

    def is_greeting(self, message: str):
        """Check if message contains greeting words (used for logging/general detection)"""
        message_lower = message.lower().strip()
        is_greeting = any(greeting in message_lower for greeting in self.greetings)
        print(f"✓ Greeting check for '{message}': {is_greeting}")
        return is_greeting

    def is_pure_greeting(self, message: str):
        """Check if message is ONLY a greeting (more restrictive)"""
        message_lower = message.lower().strip()

        # Remove common punctuation
        cleaned_message = message_lower.replace('!', '').replace('.', '').replace(',', '').strip()

        # Check if the entire message is just a greeting (possibly with extra words like "there")
        greeting_patterns = [
            "hi", "hello", "hey", "hi there", "hello there", "hey there",
            "good morning", "good afternoon", "good evening", "greetings",
            "howdy", "what's up", "whats up", "sup"
        ]

        is_pure_greeting = cleaned_message in greeting_patterns or len(cleaned_message.split()) <= 2 and any(
            greeting in cleaned_message for greeting in ["hi", "hello", "hey"])

        print(f"✓ Pure greeting check for '{message}': {is_pure_greeting}")
        return is_pure_greeting

    def generate_greeting_response(self):
        prompt = """
        Generate a professional, friendly greeting response for a law research assistant.
        Be welcoming and mention the key features: finding sources, summarizing papers, extracting arguments.
        Keep it concise and professional.
        """
        response = self.model.generate_content(prompt)
        print("✓ Greeting response generated")
        return response.text


import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class ArgumentResearchAgent:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    async def find_sources(self, argument: str, support_type: str = "both", source_count: int = 5,
                           paper_content: str = None):
        print(f"✓ Finding sources for argument: '{argument[:50]}...' | Type: {support_type} | Count: {source_count}")

        # Enhanced query generation with paper content context
        search_queries = await self._generate_enhanced_search_queries(argument, paper_content)
        sources = []

        # Search with more sources initially for better reranking
        for query in search_queries:
            academic_sources = await self._search_academic_sources(query, rows=20)
            sources.extend(academic_sources)

        # Remove duplicates
        sources = self._remove_duplicates(sources)

        # Filter valid sources
        filtered_sources = self._filter_valid_sources(sources)

        # Rerank sources based on relevance to question
        reranked_sources = self._rerank_sources_by_relevance(filtered_sources, argument)

        # Categorize top sources
        categorized_sources = self._categorize_sources(reranked_sources[:20], argument, support_type)

        # Apply final count limit with relevance scoring
        final_sources = self._apply_source_count_limit_with_scoring(categorized_sources, support_type, source_count,
                                                                    argument)

        print(
            f"✓ Found {len(final_sources.get('supporting', []))} supporting and {len(final_sources.get('contradicting', []))} contradicting sources")
        return final_sources

    def _remove_duplicates(self, sources):
        seen_titles = set()
        unique_sources = []
        for source in sources:
            title_normalized = source.get('title', '').lower().strip()
            if title_normalized and title_normalized not in seen_titles:
                seen_titles.add(title_normalized)
                unique_sources.append(source)
        print(f"✓ Removed duplicates: {len(sources)} -> {len(unique_sources)} sources")
        return unique_sources

    def _rerank_sources_by_relevance(self, sources, query):
        if not sources:
            return sources

        print("✓ Reranking sources by relevance")

        # Prepare texts for vectorization
        query_text = query.lower()
        source_texts = []

        for source in sources:
            # Combine title, abstract, and journal for better matching
            text_parts = [
                source.get('title', ''),
                source.get('abstract', ''),
                source.get('journal', ''),
                ' '.join(source.get('authors', []))
            ]
            combined_text = ' '.join(filter(None, text_parts)).lower()
            source_texts.append(combined_text)

        if not source_texts:
            return sources

        try:
            # Add query to texts for vectorization
            all_texts = [query_text] + source_texts

            # Vectorize
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)

            # Calculate cosine similarity between query and each source
            query_vector = tfidf_matrix[0:1]
            source_vectors = tfidf_matrix[1:]

            similarities = cosine_similarity(query_vector, source_vectors)[0]

            # Add title relevance bonus
            title_bonuses = []
            for source in sources:
                title = source.get('title', '').lower()
                title_bonus = self._calculate_title_relevance(title, query_text)
                title_bonuses.append(title_bonus)

            # Combine similarity scores with title bonuses
            final_scores = similarities + np.array(title_bonuses) * 0.3

            # Sort sources by combined score
            scored_sources = list(zip(sources, final_scores))
            scored_sources.sort(key=lambda x: x[1], reverse=True)

            reranked_sources = [source for source, score in scored_sources]
            print(f"✓ Reranked {len(reranked_sources)} sources by relevance")
            return reranked_sources

        except Exception as e:
            print(f"✗ Reranking failed, using original order: {e}")
            return sources

    def _calculate_title_relevance(self, title, query):
        if not title or not query:
            return 0

        # Extract key terms from query
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        title_terms = set(re.findall(r'\b\w+\b', title.lower()))

        # Calculate overlap
        common_terms = query_terms.intersection(title_terms)
        if not query_terms:
            return 0

        overlap_ratio = len(common_terms) / len(query_terms)

        # Bonus for exact phrase matches
        phrase_bonus = 0
        query_phrases = re.findall(r'\b\w+(?:\s+\w+){1,2}\b', query.lower())
        for phrase in query_phrases:
            if phrase in title.lower():
                phrase_bonus += 0.2

        return overlap_ratio + phrase_bonus

    async def _generate_enhanced_search_queries(self, argument: str, paper_content: str = None):
        context_info = ""
        if paper_content:
            # Extract key terms from paper for context
            context_info = f"\n\nContext from uploaded document: {paper_content[:1000]}"

        prompt = f"""
        Generate 3 highly specific search queries for finding academic sources related to this argument:
        "{argument}"
        {context_info}

        Focus on:
        1. Key terms and concepts from the argument
        2. Related legal/academic terminology
        3. Specific domain context if provided

        Return only the search queries, one per line, without numbering.
        """

        response = self.model.generate_content(prompt)
        queries = [q.strip().replace('1. ', '').replace('2. ', '').replace('3. ', '')
                   for q in response.text.split('\n') if q.strip()]
        print(f"✓ Generated {len(queries[:3])} enhanced search queries")
        return queries[:3]

    import asyncio

    # Replace the existing _search_academic_sources method in ArgumentResearchAgent class
    async def _search_academic_sources(self, query: str, rows: int = 10):
        print(f"✓ Starting parallel search for query: {query}")

        # Create tasks for parallel execution
        tasks = [
            self._search_crossref(query, rows),
            self._search_doaj(query, rows)
        ]

        # Execute both searches concurrently
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            sources = []

            # Process CrossRef results
            if isinstance(results[0], list):
                crossref_sources = results[0]
                sources.extend(crossref_sources)
                print(f"✓ Found {len(crossref_sources)} sources from CrossRef")
            else:
                print(f"✗ CrossRef search failed: {results[0]}")

            # Process DOAJ results
            if isinstance(results[1], list):
                doaj_sources = results[1]
                sources.extend(doaj_sources)
                print(f"✓ Found {len(doaj_sources)} sources from DOAJ")
            else:
                print(f"✗ DOAJ search failed: {results[1]}")

            return sources

        except Exception as e:
            print(f"✗ Parallel search failed: {e}")
            # Fallback to sequential if parallel fails
            sources = []
            try:
                crossref_sources = await self._search_crossref(query, rows)
                sources.extend(crossref_sources)
                print(f"✓ Fallback: Found {len(crossref_sources)} sources from CrossRef")
            except Exception as e:
                print(f"✗ Fallback CrossRef search failed: {e}")

            try:
                doaj_sources = await self._search_doaj(query, rows)
                sources.extend(doaj_sources)
                print(f"✓ Fallback: Found {len(doaj_sources)} sources from DOAJ")
            except Exception as e:
                print(f"✗ Fallback DOAJ search failed: {e}")

            return sources

    # Also optimize the find_sources method to make multiple query searches parallel
    async def find_sources(self, argument: str, support_type: str = "both", source_count: int = 5,
                           paper_content: str = None):
        print(f"✓ Finding sources for argument: '{argument[:50]}...' | Type: {support_type} | Count: {source_count}")

        # Enhanced query generation with paper content context
        search_queries = await self._generate_enhanced_search_queries(argument, paper_content)

        # Create tasks for parallel query execution
        search_tasks = [
            self._search_academic_sources(query, rows=20)
            for query in search_queries
        ]

        # Execute all searches in parallel
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        sources = []
        for i, result in enumerate(search_results):
            if isinstance(result, list):
                sources.extend(result)
                print(f"✓ Query {i + 1} returned {len(result)} sources")
            else:
                print(f"✗ Query {i + 1} failed: {result}")

        # Remove duplicates
        sources = self._remove_duplicates(sources)

        # Filter valid sources
        filtered_sources = self._filter_valid_sources(sources)

        # Rerank sources based on relevance to question
        reranked_sources = self._rerank_sources_by_relevance(filtered_sources, argument)

        # Categorize top sources
        categorized_sources = self._categorize_sources(reranked_sources[:20], argument, support_type)

        # Apply final count limit with relevance scoring
        final_sources = self._apply_source_count_limit_with_scoring(categorized_sources, support_type, source_count,
                                                                    argument)

        print(
            f"✓ Found {len(final_sources.get('supporting', []))} supporting and {len(final_sources.get('contradicting', []))} contradicting sources")
        return final_sources

    # async def _search_academic_sources(self, query: str, rows: int = 10):
    #     sources = []

    #     try:

    #         crossref_sources = await self._search_crossref(query, rows)
    #         sources.extend(crossref_sources)
    #         print(f"✓ Found {len(crossref_sources)} sources from CrossRef")
    #     except Exception as e:
    #         print(f"✗ CrossRef search failed: {e}")
    #     try:
    #         doaj_sources = await self._search_doaj(query, rows)
    #         sources.extend(doaj_sources)
    #         print(f"✓ Found {len(doaj_sources)} sources from DOAJ")
    #     except Exception as e:
    #         print(f"✗ DOAJ search failed: {e}")

    #     return sources

    import asyncio



    async def _search_crossref(self, query: str, rows: int = 10):
        url = f"https://api.crossref.org/works?query={urllib.parse.quote(query)}&filter=type:journal-article&rows={rows}&sort=relevance"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()
                sources = []

                for item in data.get('message', {}).get('items', []):
                    source = {
                        'title': item.get('title', [''])[0] if item.get('title') else '',
                        'authors': [author.get('given', '') + ' ' + author.get('family', '')
                                    for author in item.get('author', [])],
                        'journal': item.get('container-title', [''])[0] if item.get('container-title') else '',
                        'year': item.get('published-print', {}).get('date-parts', [[None]])[0][0] if item.get(
                            'published-print') else None,
                        'doi': item.get('DOI', ''),
                        'issn': item.get('ISSN', []),
                        'url': item.get('URL', ''),
                        'abstract': item.get('abstract', ''),
                        'type': 'academic',
                        'source_db': 'crossref'
                    }
                    sources.append(source)

                return sources

    async def _search_doaj(self, query: str, rows: int = 10):
        url = f"https://doaj.org/api/v2/search/articles/{urllib.parse.quote(query)}?pageSize={rows}"

        async with aiohttp.ClientSession() as session:
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
                        'url': bibjson.get('link', [{}])[0].get('url', '') if bibjson.get('link') else '',
                        'abstract': bibjson.get('abstract', ''),
                        'type': 'academic',
                        'source_db': 'doaj'
                    }
                    sources.append(source)

                return sources

    def _apply_source_count_limit_with_scoring(self, sources, support_type, count, query):
        # Further refine by relevance within each category
        if support_type == "supporting":
            supporting = self._final_relevance_filter(sources.get('supporting', []), query, count)
            return {'supporting': supporting}
        elif support_type == "contradicting":
            contradicting = self._final_relevance_filter(sources.get('contradicting', []), query, count)
            return {'contradicting': contradicting}
        else:
            supporting_count = count // 2
            contradicting_count = count - supporting_count
            supporting = self._final_relevance_filter(sources.get('supporting', []), query, supporting_count)
            contradicting = self._final_relevance_filter(sources.get('contradicting', []), query, contradicting_count)
            return {
                'supporting': supporting,
                'contradicting': contradicting
            }

    def _final_relevance_filter(self, sources, query, limit):
        if not sources:
            return []

        # Score each source for final selection
        scored_sources = []
        for source in sources:
            title_score = self._calculate_title_relevance(source.get('title', ''), query)
            abstract_score = self._calculate_abstract_relevance(source.get('abstract', ''), query)
            final_score = title_score * 0.7 + abstract_score * 0.3
            scored_sources.append((source, final_score))

        # Sort by score and return top sources
        scored_sources.sort(key=lambda x: x[1], reverse=True)
        return [source for source, score in scored_sources[:limit]]

    def _calculate_abstract_relevance(self, abstract, query):
        if not abstract or not query:
            return 0

        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        abstract_terms = set(re.findall(r'\b\w+\b', abstract.lower()))

        if not query_terms:
            return 0

        common_terms = query_terms.intersection(abstract_terms)
        return len(common_terms) / len(query_terms)

    def _filter_valid_sources(self, sources):
        valid_sources = []
        for source in sources:
            # Check if source has essential fields
            if (source.get('title') and
                    source.get('title').strip() and
                    len(source.get('title', '').strip()) > 10):
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
        {json.dumps([{'title': s['title'], 'abstract': s.get('abstract', '')[:200]} for s in sources[:10]], indent=2)}

        Return a JSON object with two arrays: "supporting" and "contradicting", containing the indices of sources.
        Example: {{"supporting": [0, 2, 4], "contradicting": [1, 3, 5]}}
        """

        try:
            response = self.model.generate_content(prompt)
            # Clean the response to extract JSON
            response_text = response.text.strip()
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]

            categorization = json.loads(response_text)

            result = {
                'supporting': [sources[i] for i in categorization.get('supporting', []) if i < len(sources)],
                'contradicting': [sources[i] for i in categorization.get('contradicting', []) if i < len(sources)]
            }

            print(
                f"✓ Categorized {len(result['supporting'])} supporting and {len(result['contradicting'])} contradicting sources")
            return result

        except Exception as e:
            print(f"✗ Categorization failed, using fallback: {e}")
            # Fallback: split sources evenly
            mid_point = len(sources) // 2
            return {
                'supporting': sources[:mid_point],
                'contradicting': sources[mid_point:]
            }


# Updated chat route function
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    session_id = data.get('session_id')
    query_type = data.get('type', 'general')

    print(f"✓ Chat request: '{user_message}' | Type: {query_type} | Session: {session_id}")

    # Check for greeting ONLY if query_type is 'general' and message is purely a greeting
    if query_type == 'general' and greeting_agent.is_pure_greeting(user_message):
        print("✓ Greeting detected, generating greeting response")
        greeting_response = greeting_agent.generate_greeting_response()
        response = {
            'type': 'general',
            'message': greeting_response
        }
    else:
        chat_history = db.get_recent_chats(session_id)
        paper_content = db.get_paper_content(session_id)  # Get paper content for context

        # Enhanced context for argument support queries
        if query_type == 'find_sources' and len(chat_history) > 0:
            # Check if this is related to extracted arguments
            context_message = context_agent.build_enhanced_context(user_message, chat_history, paper_content)
            user_message = context_message

        # Only restructure question for general queries with chat history
        if len(chat_history) > 0 and query_type == 'general':
            restructured_question = context_agent.restructure_question(user_message, chat_history)
            user_message = restructured_question

        if query_type == 'find_sources':
            support_type = data.get('support_type', 'both')
            source_count = data.get('source_count', 5)
            print(f"✓ Finding sources: {support_type} type, {source_count} count")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            sources = loop.run_until_complete(
                research_agent.find_sources(user_message, support_type, source_count, paper_content)
            )

            supporting_count = len(sources.get('supporting', []))
            contradicting_count = len(sources.get('contradicting', []))

            response = {
                'type': 'sources',
                'sources': sources,
                'message': f"Found {supporting_count} supporting and {contradicting_count} contradicting sources"
            }

        elif query_type == 'summarize':
            if not paper_content:
                response = {'error': 'No paper uploaded for this session'}
            else:
                summary_type = data.get('summary_type', 'comprehensive')
                summary = paper_agent.summarize_paper(paper_content, summary_type)
                response = {
                    'type': 'summary',
                    'summary': summary
                }

        elif query_type == 'extract_arguments':
            if not paper_content:
                response = {'error': 'No paper uploaded for this session'}
            else:
                arguments = paper_agent.extract_arguments(paper_content)
                response = {
                    'type': 'arguments',
                    'arguments': arguments
                }

        else:  # query_type == 'general'
            print("✓ Generating general response")
            model = genai.GenerativeModel('gemini-1.5-flash')
            ai_response = model.generate_content(user_message)
            response = {
                'type': 'general',
                'message': ai_response.text
            }

    response_text = json.dumps(response) if isinstance(response, dict) else str(response)
    db.save_chat(session_id, data.get('message'), response_text)

    print("✓ Chat response generated and saved")
    return jsonify(response)


# Enhanced ContextualAgent
class ContextualAgent:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def build_enhanced_context(self, current_question: str, chat_history: List[tuple], paper_content: str = None):
        print(f"✓ Building enhanced context for argument support query")
        context = self._build_context(chat_history)

        paper_context = ""
        if paper_content:
            paper_context = f"\n\nDocument context: {paper_content[:1000]}"

        prompt = f"""
        Based on the conversation history and document context, enhance this query for finding supporting sources:

        Current query: "{current_question}"

        Conversation context:
        {context}
        {paper_context}

        Return an enhanced query that incorporates relevant context for finding the most relevant academic sources.
        """

        response = self.model.generate_content(prompt)
        enhanced_query = response.text.strip()
        print(f"✓ Enhanced query: '{current_question}' -> '{enhanced_query}'")
        return enhanced_query

    def restructure_question(self, current_question: str, chat_history: List[tuple]):
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
        for user_msg, bot_response in chat_history[-5:]:  # Reduced to last 5 for efficiency
            context_parts.append(f"User: {user_msg}")
            context_parts.append(f"Assistant: {bot_response[:150]}...")  # Reduced length

        return "\n".join(context_parts)


# Enhanced ContextualAgent
class ContextualAgent:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def build_enhanced_context(self, current_question: str, chat_history: List[tuple], paper_content: str = None):
        print(f"✓ Building enhanced context for argument support query")
        context = self._build_context(chat_history)

        paper_context = ""
        if paper_content:
            paper_context = f"\n\nDocument context: {paper_content[:1000]}"

        prompt = f"""
        Based on the conversation history and document context, enhance this query for finding supporting sources:

        Current query: "{current_question}"

        Conversation context:
        {context}
        {paper_context}

        Return an enhanced query that incorporates relevant context for finding the most relevant academic sources.
        """

        response = self.model.generate_content(prompt)
        enhanced_query = response.text.strip()
        print(f"✓ Enhanced query: '{current_question}' -> '{enhanced_query}'")
        return enhanced_query

    def restructure_question(self, current_question: str, chat_history: List[tuple]):
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
        for user_msg, bot_response in chat_history[-5:]:  # Reduced to last 5 for efficiency
            context_parts.append(f"User: {user_msg}")
            context_parts.append(f"Assistant: {bot_response[:150]}...")  # Reduced length

        return "\n".join(context_parts)


class PaperAnalysisAgent:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def extract_text_from_pdf(self, file_content):
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            print(f"✓ Extracted text from PDF: {len(text)} characters")
            return text
        except Exception as e:
            print(f"✗ PDF extraction failed: {e}")
            return None

    def summarize_paper(self, content: str, summary_type: str = "comprehensive"):
        print(f"✓ Summarizing paper: {summary_type} summary")
        if summary_type == "brief":
            prompt = f"""
            Provide a brief 2-3 sentence summary of this legal paper:

            {content[:4000]}
            """
        else:
            prompt = f"""
            Provide a comprehensive summary of this legal paper including:
            1. Main arguments
            2. Key legal principles
            3. Conclusions
            4. Methodology (if applicable)

            Paper content:
            {content[:8000]}
            """

        response = self.model.generate_content(prompt)
        print("✓ Paper summary generated")
        return response.text

    def extract_arguments(self, content: str):
        print("✓ Extracting arguments from paper")
        prompt = f"""
        Extract the main legal arguments from this paper. List them clearly:

        {content[:6000]}

        Return a JSON array of arguments.
        """

        try:
            response = self.model.generate_content(prompt)
            arguments = json.loads(response.text)
            print(f"✓ Extracted {len(arguments)} arguments")
            return arguments
        except Exception as e:
            print(f"✗ JSON parsing failed, using fallback: {e}")
            response = self.model.generate_content(f"List the main arguments from this legal paper: {content[:4000]}")
            arguments = [arg.strip() for arg in response.text.split('\n') if arg.strip()]
            print(f"✓ Extracted {len(arguments)} arguments (fallback)")
            return arguments


greeting_agent = GreetingAgent()
research_agent = ArgumentResearchAgent()
paper_agent = PaperAnalysisAgent()
context_agent = ContextualAgent()


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


if __name__ == '__main__':
    print("🚀 Starting Law Research Assistant Backend...")
    app.run(debug=True, port=5000)


