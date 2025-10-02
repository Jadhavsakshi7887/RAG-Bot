#RAG Chatbot – Document-Aware AI Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions directly from uploaded documents, combining semantic search with large language models for context-aware responses.

Features

✅ Answers queries from PDFs, text documents, and other sources

✅ Semantic search using Hugging Face embedding models

✅ Generative responses powered by OpenAI GPT-5 API

✅ Complete retrieval + generation pipeline

✅ Web interface built with Django for smooth user interaction

✅ Handles large documents efficiently and provides relevant answers

Challenge & Solution

Initially experimented with Ollama LLaMA 3.2, but it didn’t meet requirements

Switched to OpenAI GPT-5 API for reliable, context-aware generation

Built a full pipeline integrating document embeddings, retrieval, and LLM inference

Tech Stack

Backend: Python, Django

Embedding Model: Hugging Face

LLM: OpenAI GPT-5 API

Data Handling: PyPDFLoader, RecursiveCharacterTextSplitter

Vector Store: Chroma or any embedding database

Frontend: Django templates
