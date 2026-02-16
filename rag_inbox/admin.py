from django.contrib import admin
from rag_inbox.models import Content, Chunk, QueryLog

@admin.register(Content)
class ContentAdmin(admin.ModelAdmin):
    list_display = ['title', 'content_type', 'status', 'created_at']
    list_filter = ['status', 'content_type', 'created_at']
    search_fields = ['title', 'raw_text']

@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ['content', 'order', 'created_at']
    list_filter = ['created_at']

@admin.register(QueryLog)
class QueryLogAdmin(admin.ModelAdmin):
    list_display = ['query_text', 'processing_time_ms', 'created_at']
    list_filter = ['created_at']
    search_fields = ['query_text']