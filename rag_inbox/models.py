"""
Models for RAG Inbox - All models in one file
"""
from django.db import models
import uuid


class Content(models.Model):
    """Stores ingested content (notes or URLs)."""
    CONTENT_TYPE_CHOICES = [
        ('note', 'Text Note'),
        ('url', 'URL'),
    ]
    STATUS_CHOICES = [
        ('pending', 'Pending Processing'),
        ('processing', 'Currently Processing'),
        ('completed', 'Ready for Query'),
        ('failed', 'Processing Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    content_type = models.CharField(max_length=10, choices=CONTENT_TYPE_CHOICES)
    raw_text = models.TextField(blank=True, null=True)
    source_url = models.URLField(blank=True, null=True)
    
    title = models.CharField(max_length=255, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['content_type']),
            models.Index(fields=['status']),
            models.Index(fields=['-created_at']),
        ]
        verbose_name = "Content Item"
        verbose_name_plural = "Content Items"
        db_table = 'rag_inbox_content'
    def __str__(self):
        return f"{self.get_content_type_display()} - {self.title or 'Untitled'}"


class Chunk(models.Model):
    """Stores text chunks with embeddings."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    content = models.ForeignKey(
        Content,
        on_delete=models.CASCADE,
        related_name='chunks'
    )
    
    text = models.TextField()
    order = models.IntegerField()
    
    embedding_json = models.JSONField(default=list)
    
    metadata = models.JSONField(default=dict, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['content', 'order']
        indexes = [
            models.Index(fields=['content']),
            models.Index(fields=['order']),
        ]
        verbose_name = "Text Chunk"
        verbose_name_plural = "Text Chunks"
        db_table = 'rag_inbox_chunk'
    def __str__(self):
        return f"Chunk {self.order} - {self.text[:50]}..."


class QueryLog(models.Model):
    """Logs queries for debugging and analytics."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    query_text = models.TextField()
    retrieved_chunk_ids = models.JSONField(default=list, blank=True)
    answer = models.TextField()
    sources = models.JSONField(default=list, blank=True)
    processing_time_ms = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
        ]
        verbose_name = "Query Log"
        verbose_name_plural = "Query Logs"
        db_table = 'rag_inbox_querylog'
    def __str__(self):
        return f"Query - {self.query_text[:50]}..."