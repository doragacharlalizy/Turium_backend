"""
Serializers for RAG Inbox - All serializers in one file
"""
from rest_framework import serializers
from rag_inbox.models import Content, Chunk, QueryLog


class ChunkSerializer(serializers.ModelSerializer):
    """Serializer for Chunk model."""
    content_title = serializers.CharField(source='content.title', read_only=True)
    content_type = serializers.CharField(source='content.get_content_type_display', read_only=True)
    
    class Meta:
        model = Chunk
        fields = ['id', 'content', 'content_title', 'content_type', 'text', 'order', 'embedding_json', 'metadata', 'created_at']
        read_only_fields = ['id', 'created_at', 'content_title', 'content_type', 'embedding_json']


class ContentListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for listing content."""
    chunk_count = serializers.SerializerMethodField()
    preview = serializers.SerializerMethodField()
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    type_display = serializers.CharField(source='get_content_type_display', read_only=True)
    
    class Meta:
        model = Content
        fields = ['id', 'content_type', 'type_display', 'title', 'status', 'status_display', 'chunk_count', 'preview', 'source_url', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at', 'status', 'chunk_count', 'preview', 'type_display', 'status_display']
    
    def get_chunk_count(self, obj):
        """Get chunk count for content."""
        return obj.chunks.count()
    
    def get_preview(self, obj):
        """Get preview text."""
        text = obj.raw_text or "URL content"
        return text[:100] + "..." if len(text) > 100 else text


class ContentDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for single content with chunks."""
    chunks = ChunkSerializer(many=True, read_only=True)
    chunk_count = serializers.SerializerMethodField()
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    type_display = serializers.CharField(source='get_content_type_display', read_only=True)
    
    class Meta:
        model = Content
        fields = ['id', 'content_type', 'type_display', 'raw_text', 'source_url', 'title', 'metadata', 'status', 'status_display', 'error_message', 'chunk_count', 'chunks', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at', 'status', 'error_message', 'chunks', 'chunk_count', 'type_display', 'status_display']

    def get_chunk_count(self, obj):
        """Get chunk count for content."""
        return obj.chunks.count()


class ContentCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating content (notes or URLs)."""
    
    class Meta:
        model = Content
        fields = ['id', 'content_type', 'raw_text', 'source_url', 'title']
        read_only_fields = ['id']
    
    def validate(self, data):
        content_type = data.get('content_type')
        
        if content_type == 'note':
            if not data.get('raw_text') or not data['raw_text'].strip():
                raise serializers.ValidationError("Text cannot be empty for notes")
        
        elif content_type == 'url':
            if not data.get('source_url'):
                raise serializers.ValidationError("URL cannot be empty")
            url = data['source_url']
            if not (url.startswith('http://') or url.startswith('https://')):
                raise serializers.ValidationError("URL must start with http:// or https://")
        
        else:
            raise serializers.ValidationError("Content type must be 'note' or 'url'")
        
        return data


class QueryRequestSerializer(serializers.Serializer):
    """Serializer for query requests."""
    question = serializers.CharField(min_length=3, max_length=2000)
    content_id = serializers.UUIDField(required=False, allow_null=True)
    top_k = serializers.IntegerField(default=5, min_value=1, max_value=20)


class SourceSerializer(serializers.Serializer):
    """Serializer for answer sources."""
    id = serializers.UUIDField()
    title = serializers.CharField()
    snippet = serializers.CharField()
    url = serializers.URLField(required=False, allow_null=True)


class QueryResponseSerializer(serializers.Serializer):
    """Serializer for query responses."""
    answer = serializers.CharField()
    sources = SourceSerializer(many=True)
    processing_time_ms = serializers.IntegerField()


class QueryLogSerializer(serializers.ModelSerializer):
    """Serializer for query logs."""
    sources = serializers.SerializerMethodField()
    
    class Meta:
        model = QueryLog
        fields = ['id', 'query_text', 'answer', 'sources', 'processing_time_ms', 'created_at']
        read_only_fields = ['id', 'created_at']
    
    def get_sources(self, obj):
        return obj.sources if obj.sources else []


class ContentStatsSerializer(serializers.Serializer):
    """Serializer for content statistics."""
    total_content = serializers.IntegerField()
    total_chunks = serializers.IntegerField()
    by_status = serializers.DictField()
    by_type = serializers.DictField()