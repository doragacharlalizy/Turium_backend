"""
Views for RAG Inbox - All ViewSets in one file
"""
import time
import logging
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from django.db.models import Count
from sklearn.metrics.pairwise import cosine_similarity

from rag_inbox.models import Content, Chunk, QueryLog
from rag_inbox.serializers import (
    ContentListSerializer, ContentDetailSerializer, ContentCreateSerializer,
    ChunkSerializer, QueryRequestSerializer, QueryResponseSerializer,
    QueryLogSerializer, ContentStatsSerializer
)
from rag_inbox.utils.chunking import chunk_text_sliding_window
from rag_inbox.services.llm_client import get_llm_client

logger = logging.getLogger(__name__)




def get_llm_client_instance():
    """Get LLM client singleton."""
    try:
        return get_llm_client()
    except ValueError as e:
        logger.error(f"LLM client error: {str(e)}")
        raise


def retrieve_similar_chunks(query_embedding, top_k=5, content_id=None):
    """Find similar chunks using cosine similarity."""
    chunks = Chunk.objects.select_related('content')
    
    if content_id:
        chunks = chunks.filter(content_id=content_id)
    
    results = []
    
    for chunk in chunks:
        if chunk.embedding_json:
            similarity = cosine_similarity(
                [query_embedding],
                [chunk.embedding_json]
            )[0][0]
            results.append((chunk, similarity))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in results[:top_k]]


def format_context_for_llm(chunks):
    """Format chunks into LLM context string."""
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        source_info = f"[Source {i}: {chunk.content.title or 'Untitled'}"
        if chunk.content.source_url:
            source_info += f" - {chunk.content.source_url}"
        source_info += "]"
        
        context_parts.append(f"{source_info}\n{chunk.text}")
    
    return "\n\n---\n\n".join(context_parts)


def ingest_text_chunks(content_id, text):
    """
    Ingest text by chunking and generating embeddings.
    Returns number of chunks created or raises exception.
    """
    try:
        content = Content.objects.get(id=content_id)
        content.status = 'processing'
        content.save(update_fields=['status'])
        
        llm_client = get_llm_client_instance()
        
        # Step 1: Chunk text
        chunks = chunk_text_sliding_window(text)
        if not chunks:
            raise ValueError("No chunks generated from text")
        
        logger.info(f"Chunking: {len(chunks)} chunks created")
        
        # Step 2: Generate embeddings
        embeddings = llm_client.embed_batch(chunks)
        
        # Step 3: Save chunks
        chunk_objects = []
        for order, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk = Chunk(
                content=content,
                text=chunk_text,
                embedding_json=embedding,
                order=order,
            )
            chunk_objects.append(chunk)
        
        Chunk.objects.bulk_create(chunk_objects)
        
        # Step 4: Update content status
        content.status = 'completed'
        content.save(update_fields=['status', 'updated_at'])
        
        logger.info(f"Ingestion completed: {len(chunks)} chunks")
        return len(chunks)
    
    except Exception as e:
        content = Content.objects.get(id=content_id)
        content.status = 'failed'
        content.error_message = str(e)
        content.save(update_fields=['status', 'error_message'])
        logger.error(f"Ingestion failed: {str(e)}")
        raise


# ============================================================================
# ViewSets
# ============================================================================

class ContentViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Content (notes and URLs).
    
    Endpoints:
    - GET /api/content/ - List all content
    - POST /api/content/ - Create new content
    - GET /api/content/{id}/ - Get content detail
    - PUT /api/content/{id}/ - Update content
    - DELETE /api/content/{id}/ - Delete content
    - GET /api/content/stats/overview/ - Get statistics
    """
    
    queryset = Content.objects.all()
    parser_classes = (JSONParser,)
    
    def get_serializer_class(self):
        """Use different serializers for different actions."""
        if self.action == 'list':
            return ContentListSerializer
        elif self.action in ['create', 'update', 'partial_update']:
            return ContentCreateSerializer
        else:
            return ContentDetailSerializer
    
    def get_queryset(self):
        """Filter queryset by status and type if provided."""
        queryset = Content.objects.all()
        
        # Filter by type
        content_type = self.request.query_params.get('type')
        if content_type:
            queryset = queryset.filter(content_type=content_type)
        
        # Filter by status
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        return queryset
    
    def create(self, request, *args, **kwargs):
        """
        Create content.
        If content_type is 'note', chunks are generated synchronously.
        If content_type is 'url', only content is created (URL fetching can be async).
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Create content with pending status
        content = Content.objects.create(
            content_type=serializer.validated_data['content_type'],
            raw_text=serializer.validated_data.get('raw_text'),
            source_url=serializer.validated_data.get('source_url'),
            title=serializer.validated_data.get('title', 'Untitled'),
            status='pending'
        )
        
        logger.info(f"Content created: {content.id}")
        
        # For notes, ingest synchronously
        # For notes, ingest synchronously
        if content.content_type == 'note':
            try:
                ingest_text_chunks(str(content.id), content.raw_text)
            except Exception as e:
                logger.error(f"Failed to ingest: {str(e)}")

        # For URLs, fetch and ingest content
        elif content.content_type == 'url':
            try:
                # Fetch URL content
                import requests
                from bs4 import BeautifulSoup
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(content.source_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Extract text from HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks_text = '\n'.join(line for line in lines if line)
                
                if not chunks_text.strip():
                    raise ValueError("No text content found in URL")
                
                # Save extracted text
                content.raw_text = chunks_text
                content.save(update_fields=['raw_text'])
                
                # Process into chunks
                ingest_text_chunks(str(content.id), chunks_text)
                
            except Exception as e:
                content.status = 'failed'
                content.error_message = f"Failed to fetch URL: {str(e)}"
                content.save(update_fields=['status', 'error_message'])
                logger.error(f"URL fetch failed: {str(e)}")        
        # For URLs, would be handled by async task in production
        # For now, just mark as pending
        
        response_serializer = ContentDetailSerializer(content)
        return Response(response_serializer.data, status=status.HTTP_201_CREATED)
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """
        Get content statistics.
        Endpoint: GET /api/content/stats/
        """
        all_content = Content.objects.all()
        total_chunks = Chunk.objects.count()
        
        # Get status counts
        status_counts = (
            all_content
            .values('status')
            .annotate(count=Count('id'))
            .values_list('status', 'count')
        )
        status_dict = {status: count for status, count in status_counts}
        
        # Get type counts
        type_counts = (
            all_content
            .values('content_type')
            .annotate(count=Count('id'))
            .values_list('content_type', 'count')
        )
        type_dict = {ctype: count for ctype, count in type_counts}
        
        stats_data = {
            'total_content': all_content.count(),
            'total_chunks': total_chunks,
            'by_status': status_dict,
            'by_type': type_dict,
        }
        
        serializer = ContentStatsSerializer(stats_data)
        return Response(serializer.data)


class ChunkViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ReadOnly ViewSet for Chunks.
    
    Endpoints:
    - GET /api/chunks/ - List all chunks
    - GET /api/chunks/{id}/ - Get chunk detail
    """
    
    queryset = Chunk.objects.select_related('content')
    serializer_class = ChunkSerializer
    
    def get_queryset(self):
        """Filter chunks by content if provided."""
        queryset = Chunk.objects.select_related('content')
        
        content_id = self.request.query_params.get('content_id')
        if content_id:
            queryset = queryset.filter(content_id=content_id)
        
        return queryset


class QueryViewSet(viewsets.ViewSet):
    """
    Custom ViewSet for RAG queries and query logs.
    
    Endpoints:
    - POST /api/query/ask/ - Ask a question
    - GET /api/query/logs/ - Get query logs
    - GET /api/query/logs/{id}/ - Get query log detail
    """
    
    @action(detail=False, methods=['post'])
    def ask(self, request):
        """
        Ask a question over ingested content.
        Endpoint: POST /api/query/ask/
        
        Request body:
        {
            "question": "What is...?",
            "content_id": "uuid (optional)",
            "top_k": 5
        }
        """
        start_time = time.time()
        
        # Validate request
        serializer = QueryRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        question = serializer.validated_data['question']
        content_id = serializer.validated_data.get('content_id')
        top_k = serializer.validated_data.get('top_k', 5)
        
        # Check if content exists
        if content_id:
            if not Content.objects.filter(id=content_id).exists():
                return Response(
                    {'error': 'Content not found'},
                    status=status.HTTP_404_NOT_FOUND
                )
        
        # Check if any completed content exists
        if not Content.objects.filter(status='completed').exists():
            return Response(
                {'error': 'No processed content available'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            llm_client = get_llm_client_instance()
            
            # Step 1: Embed question
            logger.info(f"Processing query: {question[:100]}")
            query_embedding = llm_client.embed(question)
            
            # Step 2: Retrieve similar chunks
            chunks = retrieve_similar_chunks(query_embedding, top_k, content_id)
            
            if not chunks:
                return Response({
                    'answer': 'No relevant information found in the knowledge base.',
                    'sources': [],
                    'processing_time_ms': int((time.time() - start_time) * 1000)
                })
            
            # Step 3: Build context
            context = format_context_for_llm(chunks)
            
            # Step 4: Call LLM
            system_prompt = (
                "You are a helpful assistant. Answer questions based ONLY on the provided context. "
                "If the context doesn't contain enough information, say so. "
                "Always cite your sources by referencing [Source N]."
            )
            user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
            
            answer = llm_client.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=1024,
            )
            
            # Step 5: Format sources
            sources = [
                {
                    'id': str(chunk.id),
                    'title': chunk.content.title or 'Untitled',
                    'snippet': chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text,
                    'url': chunk.content.source_url,
                }
                for chunk in chunks
            ]
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            # Step 6: Log query
            QueryLog.objects.create(
                query_text=question,
                retrieved_chunk_ids=[str(c.id) for c in chunks],
                answer=answer,
                sources=sources,
                processing_time_ms=elapsed_ms,
            )
            
            logger.info(f"Query processed in {elapsed_ms}ms, {len(chunks)} sources")
            
            # Return response
            response_data = {
                'answer': answer,
                'sources': sources,
                'processing_time_ms': elapsed_ms,
            }
            
            serializer = QueryResponseSerializer(response_data)
            return Response(serializer.data)
        
        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            return Response(
                {'error': 'Failed to process query', 'details': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def logs(self, request):
        """
        Get query logs.
        Endpoint: GET /api/query/logs/
        """
        logs = QueryLog.objects.all()
        
        # Pagination
        limit = request.query_params.get('limit', 20)
        offset = request.query_params.get('offset', 0)
        
        try:
            limit = int(limit)
            offset = int(offset)
        except (ValueError, TypeError):
            limit = 20
            offset = 0
        
        total = logs.count()
        logs = logs[offset:offset + limit]
        
        serializer = QueryLogSerializer(logs, many=True)
        return Response({
            'total': total,
            'limit': limit,
            'offset': offset,
            'results': serializer.data
        })
    
    @action(detail=True, methods=['get'], url_path='logs/(?P<log_id>[^/.]+)')
    def log_detail(self, request, log_id=None):
        """
        Get single query log.
        Endpoint: GET /api/query/logs/{id}/
        """
        try:
            log = QueryLog.objects.get(id=log_id)
            serializer = QueryLogSerializer(log)
            return Response(serializer.data)
        except QueryLog.DoesNotExist:
            return Response(
                {'error': 'Query log not found'},
                status=status.HTTP_404_NOT_FOUND
            )