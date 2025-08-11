"""
Streamlit app for PDF chat with RAG system and performance metrics.
"""
# Initialize LangTrace FIRST - before any LLM imports
from src.utils.config import LANGTRACE_API_KEY
from langtrace_python_sdk import langtrace
langtrace.init(api_key=LANGTRACE_API_KEY)

import streamlit as st
import time
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
from typing import List, Dict, Any
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

# Import your existing RAG components
from src.models.database import RAGDatabase
from src.models.document_processor import process_documents
from src.models.query_engine import QueryEngine
from src.utils.config import validate_config
from src.utils.logging_config import setup_logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.document import Document

# Page config
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
setup_logging("INFO")

class StreamlitRAGApp:
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'database' not in st.session_state:
            st.session_state.database = None
        if 'query_engine' not in st.session_state:
            st.session_state.query_engine = None
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = []
    
    def setup_rag_system(self):
        """Initialize RAG system components."""
        try:
            validate_config()
            if st.session_state.database is None:
                st.session_state.database = RAGDatabase("openai")
                st.session_state.query_engine = QueryEngine()
            return True
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def process_uploaded_files(self, uploaded_files: List, clear_existing: bool = False) -> bool:
        """Process uploaded PDF files and add to vector database."""
        if not uploaded_files:
            return False
            
        try:
            all_documents = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Load the PDF using PyPDFLoader
                    loader = PyPDFLoader(tmp_file_path)
                    documents = loader.load()
                    
                    # Add filename to metadata
                    for doc in documents:
                        doc.metadata['source'] = uploaded_file.name
                    
                    all_documents.extend(documents)
                    
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                finally:
                    # Clean up temp file
                    os.unlink(tmp_file_path)
            
            if all_documents:
                # Process documents into chunks
                status_text.text("Processing documents into chunks...")
                chunks = process_documents(all_documents)
                
                # Handle database reset if requested
                if clear_existing:
                    status_text.text("Clearing existing documents...")
                    try:
                        # Use safer collection clearing method
                        st.session_state.database.clear_collection()
                        existing_count = 0
                        # Clear uploaded files list
                        st.session_state.uploaded_files = []
                    except Exception as e:
                        st.error(f"Error clearing documents: {e}")
                        # Fallback to full reset if collection clearing fails
                        try:
                            st.session_state.database.reset()
                            existing_count = 0
                            st.session_state.uploaded_files = []
                        except Exception as e2:
                            st.error(f"Error with fallback reset: {e2}")
                            st.info("💡 Try restarting the app if files are locked.")
                            return False
                else:
                    # Get existing document count for unique IDs
                    try:
                        existing_data = st.session_state.database.db.get()
                        existing_count = len(existing_data.get('ids', []))
                    except:
                        existing_count = 0
                
                # Add documents to database
                status_text.text("Adding documents to vector database...")
                texts = [chunk.page_content for chunk in chunks]
                metadatas = [chunk.metadata for chunk in chunks]
                ids = [f"chunk_{existing_count + i}" for i in range(len(chunks))]
                
                try:
                    st.session_state.database.db.add_texts(
                        texts=texts, 
                        metadatas=metadatas, 
                        ids=ids
                    )
                except Exception as e:
                    st.error(f"Error adding documents to database: {e}")
                    st.info("💡 Try using the 'Clear existing documents' option if you're replacing documents.")
                    return False
                
                # Store uploaded file info
                new_files = [f.name for f in uploaded_files]
                if clear_existing:
                    st.session_state.uploaded_files = new_files
                else:
                    # Append to existing files (avoid duplicates)
                    existing_files = set(st.session_state.uploaded_files)
                    for file_name in new_files:
                        if file_name not in existing_files:
                            st.session_state.uploaded_files.append(file_name)
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Successfully processed {len(chunks)} document chunks from {len(all_documents)} pages!")
                time.sleep(1)
                status_text.empty()
                progress_bar.empty()
                
                return True
            else:
                st.error("No documents were extracted from the uploaded files.")
                return False
                
        except Exception as e:
            st.error(f"Error processing files: {e}")
            return False
    
    def calculate_performance_metrics(self, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for the query result."""
        metrics = {
            'response_time': result.get('response_time', 0),
            'num_sources': len(result.get('sources', [])),
            'answer_length': len(result.get('answer', '')),
            'confidence_score': self.estimate_confidence(result),
            'source_diversity': self.calculate_source_diversity(result.get('sources', [])),
            'query_complexity': len(query.split()),
            'avg_similarity': result.get('avg_similarity', 0),
            'max_similarity': result.get('max_similarity', 0),
            'retrieved_chunks': len(result.get('retrieved_docs_info', []))
        }
        return metrics
    
    def estimate_confidence(self, result: Dict[str, Any]) -> float:
        """Estimate confidence score based on various factors."""
        answer = result.get('answer', '')
        sources = result.get('sources', [])
        avg_similarity = result.get('avg_similarity', 0)
        max_similarity = result.get('max_similarity', 0)
        
        confidence = 0.3  # Base confidence
        
        # Similarity score contribution (40% of confidence)
        similarity_score = min(avg_similarity / 2.0, 0.4)  # Normalize and cap
        confidence += similarity_score
        
        # Source diversity contribution (20% of confidence)
        if len(sources) >= 3:
            confidence += 0.2
        elif len(sources) >= 2:
            confidence += 0.1
        
        # Answer quality indicators (20% of confidence)
        if len(answer) > 150:  # Detailed answer
            confidence += 0.1
        if len(answer) > 300:  # Very detailed answer
            confidence += 0.1
        
        # Check for uncertainty phrases (penalty)
        uncertainty_phrases = [
            'not sure', 'unclear', 'might be', 'possibly', 'perhaps',
            'i don\'t know', 'not mentioned', 'not specified', 'unclear from'
        ]
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in answer.lower())
        confidence -= uncertainty_count * 0.1
        
        # Check for definitive phrases (bonus)
        definitive_phrases = [
            'according to', 'the document states', 'specifically mentions',
            'clearly indicates', 'explicitly says'
        ]
        definitive_count = sum(1 for phrase in definitive_phrases if phrase in answer.lower())
        confidence += definitive_count * 0.05
        
        return max(0.0, min(1.0, confidence))
    
    def calculate_source_diversity(self, sources: List[str]) -> float:
        """Calculate diversity of sources used."""
        if not sources:
            return 0.0
        
        unique_sources = set(sources)
        return len(unique_sources) / len(sources)
    
    def display_performance_metrics(self, metrics: Dict[str, Any]):
        """Display performance metrics in the sidebar."""
        st.sidebar.subheader("📊 Query Performance")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric("Response Time", f"{metrics['response_time']:.2f}s")
            st.metric("Sources Used", metrics['num_sources'])
            st.metric("Answer Length", f"{metrics['answer_length']} chars")
        
        with col2:
            st.metric("Confidence", f"{metrics['confidence_score']:.1%}")
            st.metric("Max Similarity", f"{metrics.get('max_similarity', 0):.3f}")
            st.metric("Query Complexity", f"{metrics['query_complexity']} words")
    
    def display_chat_interface(self):
        """Display the chat interface."""
        st.subheader("💬 Chat with your PDFs")
        
        # Display chat history (handle both old and new format)
        for i, chat_item in enumerate(st.session_state.chat_history):
            # Handle backward compatibility
            if len(chat_item) == 3:
                query, response, metrics = chat_item
                result_data = None
            else:
                query, response, metrics, result_data = chat_item
            with st.container():
                st.write(f"**You:** {query}")
                st.write(f"**Assistant:** {response}")
                
                # Show metrics and visualization in expander
                with st.expander(f"📊 Query Analysis (Query {i+1})"):
                    # Performance metrics
                    st.subheader("Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"⏱️ Response Time: {metrics['response_time']:.2f}s")
                        st.write(f"📄 Sources: {metrics['num_sources']}")
                    with col2:
                        st.write(f"🎯 Confidence: {metrics['confidence_score']:.1%}")
                        st.write(f"📏 Length: {metrics['answer_length']} chars")
                    with col3:
                        st.write(f"🔄 Source Diversity: {metrics['source_diversity']:.1%}")
                        st.write(f"🧠 Complexity: {metrics['query_complexity']} words")
                    
                    # Retrieved chunks visualization
                    if result_data and result_data.get('retrieved_docs_info'):
                        st.subheader("🎯 Retrieved Chunks Visualization")
                        
                        # Get embeddings and create visualization
                        embeddings, texts, metadatas = self.get_vector_embeddings()
                        
                        if embeddings is not None:
                            # Find indices of retrieved chunks
                            retrieved_indices = self.find_retrieved_chunk_indices(
                                result_data['retrieved_docs_info'], 
                                texts
                            )
                            
                            if retrieved_indices:
                                # Create visualization with highlighted retrieved chunks
                                method = st.selectbox(
                                    "Visualization Method", 
                                    options=['umap', 'tsne', 'pca'],
                                    key=f"viz_method_{i}"
                                )
                                
                                fig = self.create_2d_visualization(
                                    embeddings, 
                                    texts, 
                                    metadatas, 
                                    method=method,
                                    highlighted_indices=retrieved_indices,
                                    query_text=query
                                )
                                
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show retrieved chunks details
                                    st.subheader("📄 Retrieved Chunks Details")
                                    for j, doc_info in enumerate(result_data['retrieved_docs_info']):
                                        with st.expander(f"Chunk {j+1} (Similarity: {doc_info['similarity_score']:.3f})"):
                                            st.write(f"**Source:** {doc_info['metadata'].get('source', 'Unknown')}")
                                            st.write(f"**Content:** {doc_info['content']}")
                            else:
                                st.warning("Could not find retrieved chunks in the current database.")
                        else:
                            st.info("No embeddings available for visualization.")
                
                st.divider()
        
        # Query input
        query = st.text_input("Ask a question about your uploaded PDFs:", key="query_input")
        
        # Retrieval strategy option
        col1, col2 = st.columns([3, 1])
        with col2:
            diverse_search = st.checkbox("Diverse sources", value=True, help="Ensure results from multiple documents")
        
        if st.button("Send", type="primary") and query:
            if st.session_state.database is None:
                st.error("Please upload and process PDFs first!")
                return
            
            with st.spinner("Thinking..."):
                try:
                    # Get response from RAG system
                    if diverse_search:
                        # Use diverse search to get chunks from multiple sources
                        results = st.session_state.database.search_diverse(query, k=8, min_sources=2)
                        result = st.session_state.query_engine._process_results(results, query)
                    else:
                        # Use standard similarity search
                        result = st.session_state.query_engine.query(
                            st.session_state.database, 
                            query, 
                            k=8
                        )
                    
                    # Calculate performance metrics
                    metrics = self.calculate_performance_metrics(query, result)
                    
                    # Add to chat history with full result data
                    st.session_state.chat_history.append((query, result['answer'], metrics, result))
                    st.session_state.performance_metrics.append(metrics)
                    
                    # Display current metrics in sidebar
                    self.display_performance_metrics(metrics)
                    
                    # Rerun to update chat display
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")
    
    def get_vector_embeddings(self):
        """Get vector embeddings from the database for visualization."""
        try:
            if st.session_state.database is None:
                return None, None, None
            
            # Get all documents from the database with embeddings explicitly included
            data = st.session_state.database.db.get(include=['embeddings', 'documents', 'metadatas'])
            
            if not data.get('documents') or len(data['documents']) == 0:
                return None, None, None
            
            # Check if embeddings are available
            embeddings = data.get('embeddings')
            if embeddings is None or len(embeddings) == 0:
                st.warning("Embeddings not found in database.")
                return None, None, None
            
            # Convert to numpy array
            embeddings = np.array(embeddings)
            texts = data['documents']
            metadatas = data['metadatas']
            
            return embeddings, texts, metadatas
            
        except Exception as e:
            st.error(f"Error getting embeddings: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return None, None, None
    
    def create_2d_visualization(self, embeddings, texts, metadatas, method='umap', highlighted_indices=None, query_text=None):
        """Create 2D visualization of document embeddings with optional highlighting."""
        try:
            if method == 'umap':
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings)-1))
            elif method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            else:  # pca
                reducer = PCA(n_components=2, random_state=42)
            
            # Reduce dimensions
            embeddings_2d = reducer.fit_transform(embeddings)
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'text': [text[:100] + "..." if len(text) > 100 else text for text in texts],
                'source': [meta.get('source', 'Unknown') for meta in metadatas],
                'chunk_id': range(len(texts)),
                'is_retrieved': [i in highlighted_indices if highlighted_indices else False for i in range(len(texts))]
            })
            
            # Create title
            title = f"Document Embeddings Visualization ({method.upper()})"
            if query_text:
                title += f" - Query: '{query_text[:50]}...'" if len(query_text) > 50 else f" - Query: '{query_text}'"
            
            # Create interactive plot with highlighting
            if highlighted_indices:
                # Create separate traces for retrieved and non-retrieved documents
                df_normal = df[~df['is_retrieved']]
                df_retrieved = df[df['is_retrieved']]
                
                fig = go.Figure()
                
                # Add normal documents
                if not df_normal.empty:
                    fig.add_trace(go.Scatter(
                        x=df_normal['x'],
                        y=df_normal['y'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            opacity=0.4,
                            color='lightgray',
                            line=dict(width=0.5, color='gray')
                        ),
                        text=df_normal['text'],
                        customdata=df_normal[['source', 'chunk_id']],
                        hovertemplate='<b>Chunk %{customdata[1]}</b><br>' +
                                    'Source: %{customdata[0]}<br>' +
                                    'Text: %{text}<br>' +
                                    '<extra></extra>',
                        name='Other Documents',
                        showlegend=True
                    ))
                
                # Add retrieved documents with highlighting
                if not df_retrieved.empty:
                    fig.add_trace(go.Scatter(
                        x=df_retrieved['x'],
                        y=df_retrieved['y'],
                        mode='markers',
                        marker=dict(
                            size=12,
                            opacity=0.9,
                            color='red',
                            line=dict(width=2, color='darkred'),
                            symbol='star'
                        ),
                        text=df_retrieved['text'],
                        customdata=df_retrieved[['source', 'chunk_id']],
                        hovertemplate='<b>🎯 Retrieved Chunk %{customdata[1]}</b><br>' +
                                    'Source: %{customdata[0]}<br>' +
                                    'Text: %{text}<br>' +
                                    '<extra></extra>',
                        name='Retrieved for Query',
                        showlegend=True
                    ))
                
                fig.update_layout(
                    title=title,
                    xaxis_title=f'{method.upper()} 1',
                    yaxis_title=f'{method.upper()} 2',
                    height=600,
                    showlegend=True
                )
                
            else:
                # Standard visualization without highlighting
                fig = px.scatter(
                    df, 
                    x='x', 
                    y='y', 
                    color='source',
                    hover_data=['text', 'chunk_id'],
                    title=title,
                    labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2'}
                )
                
                fig.update_traces(marker=dict(size=8, opacity=0.7))
                fig.update_layout(height=600)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating visualization: {e}")
            return None
    
    def find_retrieved_chunk_indices(self, retrieved_docs_info, all_texts):
        """Find the indices of retrieved chunks in the full document list."""
        retrieved_indices = []
        
        for retrieved_doc in retrieved_docs_info:
            retrieved_content = retrieved_doc['content']
            
            # Find matching chunk by content
            for i, text in enumerate(all_texts):
                if text == retrieved_content:
                    retrieved_indices.append(i)
                    break
        
        return retrieved_indices
    
    def display_document_overview(self):
        """Display overview of documents in the database."""
        if st.session_state.database is None:
            st.info("No documents loaded yet. Upload some PDFs to see the overview!")
            return
        
        try:
            data = st.session_state.database.db.get()
            
            if not data['documents']:
                st.info("No documents in database yet.")
                return
            
            st.subheader("📚 Document Database Overview")
            
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Chunks", len(data['documents']))
            
            with col2:
                sources = [meta.get('source', 'Unknown') for meta in data['metadatas']]
                unique_sources = len(set(sources))
                st.metric("Unique Documents", unique_sources)
            
            with col3:
                total_chars = sum(len(doc) for doc in data['documents'])
                st.metric("Total Characters", f"{total_chars:,}")
            
            with col4:
                avg_chunk_size = total_chars / len(data['documents']) if data['documents'] else 0
                st.metric("Avg Chunk Size", f"{avg_chunk_size:.0f} chars")
            
            # Document breakdown
            st.subheader("📄 Documents in Database")
            
            # Count chunks per source
            source_counts = {}
            for meta in data['metadatas']:
                source = meta.get('source', 'Unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
            
            # Create DataFrame for display
            df_sources = pd.DataFrame([
                {'Document': source, 'Chunks': count, 'Percentage': f"{count/len(data['documents'])*100:.1f}%"}
                for source, count in source_counts.items()
            ])
            
            st.dataframe(df_sources, use_container_width=True)
            
            # Pie chart of document distribution
            fig_pie = px.pie(
                values=list(source_counts.values()),
                names=list(source_counts.keys()),
                title="Document Distribution by Source"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying document overview: {e}")
    
    def display_vector_visualization(self):
        """Display vector space visualization."""
        st.subheader("🎯 Vector Space Visualization")
        
        # Debug information
        if st.session_state.database is not None:
            try:
                debug_data = st.session_state.database.db.get(include=['embeddings', 'documents', 'metadatas'])
                st.write(f"**Debug Info:** {len(debug_data.get('documents', []))} documents in database")
                
                embeddings = debug_data.get('embeddings')
                if embeddings is not None:
                    st.write(f"**Embeddings found:** {len(embeddings)} embeddings of dimension {np.array(embeddings[0]).shape[0] if len(embeddings) > 0 else 'unknown'}")
                else:
                    st.write("**Embeddings:** None returned")
                    
            except Exception as e:
                st.error(f"Debug error: {e}")
        
        embeddings, texts, metadatas = self.get_vector_embeddings()
        
        if embeddings is None:
            st.info("No embeddings available. Upload and process some documents first!")
            return
        
        if len(embeddings) < 2:
            st.warning("Need at least 2 document chunks for visualization.")
            return
        
        # Method selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            method = st.selectbox(
                "Visualization Method",
                options=['umap', 'tsne', 'pca'],
                help="Choose dimensionality reduction method"
            )
        
        with col2:
            st.write(f"**{method.upper()}** - Showing {len(embeddings)} document chunks in 2D space")
        
        # Create and display visualization
        with st.spinner(f"Creating {method.upper()} visualization..."):
            fig = self.create_2d_visualization(embeddings, texts, metadatas, method)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation
                st.info(f"""
                **How to interpret this visualization:**
                - Each point represents a chunk of text from your documents
                - Points that are close together have similar semantic meaning
                - Different colors represent different source documents
                - Hover over points to see the text content
                - Clusters indicate topics or themes in your documents
                """)
            else:
                st.error("Failed to create visualization.")
    
    def display_analytics_dashboard(self):
        """Display enhanced analytics dashboard."""
        st.subheader("📈 Performance Analytics")
        
        # Create tabs for different analytics views
        tab1, tab2, tab3 = st.tabs(["📊 Query Performance", "📚 Document Overview", "🎯 Vector Space"])
        
        with tab1:
            if not st.session_state.performance_metrics:
                st.info("No queries yet. Start chatting to see analytics!")
                return
            
            metrics_data = st.session_state.performance_metrics
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_response_time = sum(m['response_time'] for m in metrics_data) / len(metrics_data)
                st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
            
            with col2:
                avg_confidence = sum(m['confidence_score'] for m in metrics_data) / len(metrics_data)
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with col3:
                total_queries = len(metrics_data)
                st.metric("Total Queries", total_queries)
            
            with col4:
                avg_sources = sum(m['num_sources'] for m in metrics_data) / len(metrics_data)
                st.metric("Avg Sources Used", f"{avg_sources:.1f}")
            
            # Performance trends with Plotly
            if len(metrics_data) > 1:
                st.subheader("📊 Performance Trends")
                
                # Create DataFrame for plotting
                df_metrics = pd.DataFrame([
                    {
                        'Query': i+1,
                        'Response Time': m['response_time'],
                        'Confidence': m['confidence_score'],
                        'Sources Used': m['num_sources'],
                        'Similarity': m.get('avg_similarity', 0)
                    }
                    for i, m in enumerate(metrics_data)
                ])
                
                # Response time trend
                fig_time = px.line(df_metrics, x='Query', y='Response Time', 
                                 title='Response Time Over Queries',
                                 markers=True)
                st.plotly_chart(fig_time, use_container_width=True)
                
                # Multi-metric comparison
                fig_multi = go.Figure()
                fig_multi.add_trace(go.Scatter(x=df_metrics['Query'], y=df_metrics['Confidence'], 
                                             mode='lines+markers', name='Confidence Score'))
                fig_multi.add_trace(go.Scatter(x=df_metrics['Query'], y=df_metrics['Similarity'], 
                                             mode='lines+markers', name='Avg Similarity', yaxis='y2'))
                
                fig_multi.update_layout(
                    title='Confidence vs Similarity Over Queries',
                    xaxis_title='Query Number',
                    yaxis=dict(title='Confidence Score', side='left'),
                    yaxis2=dict(title='Average Similarity', side='right', overlaying='y'),
                    height=400
                )
                st.plotly_chart(fig_multi, use_container_width=True)
        
        with tab2:
            self.display_document_overview()
        
        with tab3:
            self.display_vector_visualization()
    
    def run(self):
        """Main app runner."""
        st.title("📚 PDF Chat Assistant with Performance Analytics")
        st.markdown("Upload your PDFs and chat with them using AI. See real-time performance metrics!")
        
        # Initialize RAG system
        if not self.setup_rag_system():
            st.stop()
        
        # Sidebar for file upload and metrics
        with st.sidebar:
            st.header("📁 Upload PDFs")
            
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Upload one or more PDF files to chat with"
            )
            
            # Option to clear existing data
            clear_existing = st.checkbox(
                "🗑️ Clear existing documents first", 
                value=False,
                help="Check this to replace all existing documents. Uncheck to add to existing documents."
            )
            
            if uploaded_files and st.button("Process PDFs", type="primary"):
                if self.process_uploaded_files(uploaded_files, clear_existing):
                    st.success(f"✅ Processed {len(uploaded_files)} files!")
                    st.balloons()
            
            # Show uploaded files
            if st.session_state.uploaded_files:
                st.subheader("📄 Uploaded Files")
                for filename in st.session_state.uploaded_files:
                    st.write(f"• {filename}")
            
            # Reset button
            if st.button("🗑️ Clear All Data"):
                with st.spinner("Clearing all data..."):
                    try:
                        # Clear database using safer method
                        if st.session_state.database is not None:
                            try:
                                st.session_state.database.clear_collection()
                            except:
                                # Fallback to full reset
                                st.session_state.database.reset()
                        
                        # Clear session state
                        st.session_state.uploaded_files = []
                        st.session_state.chat_history = []
                        st.session_state.performance_metrics = []
                        
                        st.success("All data cleared!")
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error clearing data: {e}")
                        st.info("💡 You may need to restart the app if files are locked.")
        
        # Main content area
        tab1, tab2 = st.tabs(["💬 Chat", "📈 Analytics"])
        
        with tab1:
            self.display_chat_interface()
        
        with tab2:
            self.display_analytics_dashboard()

# Run the app
if __name__ == "__main__":
    app = StreamlitRAGApp()
    app.run()