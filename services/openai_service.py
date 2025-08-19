import os
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional, Generator
import logging
from models.processing_models import ChatMessage
    
class AzureOpenAIService:
    """Service class for Azure OpenAI operations"""
    
    def __init__(self):
        """
        Initialize Azure OpenAI service
        
        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            api_version: API version to use
            default_model: Default model deployment name
        """
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"]
        api_key=os.environ["AZURE_OPENAI_API_KEY"]
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        self.default_model = os.getenv("AZURE_OPENAI_DEFAULT_MODEL", "gpt-4o")
        
        if not endpoint or not api_key:
            raise ValueError("Azure OpenAI endpoint and API key must be set in environment variables.")
        
        # Initialize the client
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
        
        logging.info(f"Azure OpenAI service initialized with endpoint: {endpoint}")
    
    def simple_chat(
        self, 
        user_message: str, 
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Simple chat interface for single user message
        
        Args:
            user_message: The user's message
            system_message: Optional system message
            model: Model deployment name
            temperature: Sampling temperature
            
        Returns:
            The assistant's response as a string
        """
        messages = []
        
        if system_message:
            messages.append(ChatMessage(role="system", content=system_message))
        
        messages.append(ChatMessage(role="user", content=user_message))
        
        response = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature
        )
        
        return response["content"]
    
    def chat_completion(
        self, 
        messages: List[ChatMessage], 
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate chat completion
        
        Args:
            messages: List of ChatMessage objects
            model: Model deployment name (uses default if None)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            stream: Whether to stream the response
            
        Returns:
            Dict containing the response
        """
        try:
            model_name = model or self.default_model
            
            # Convert ChatMessage objects to dict format
            message_dicts = [
                {"role": msg.role, "content": msg.content} 
                for msg in messages
            ]
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=message_dicts,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=stream
            )
            
            if stream:
                return {"stream": response}
            else:
                return {
                    "content": response.choices[0].message.content,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "model": response.model,
                    "created": response.created
                }
                
        except Exception as e:
            logging.error(f"Error in chat completion: {str(e)}")
            raise
    

    # def chat_completion_stream(
    #     self, 
    #     messages: List[ChatMessage], 
    #     model: Optional[str] = None,
    #     temperature: float = 0.7,
    #     max_tokens: Optional[int] = None
    # ) -> Generator[str, None, None]:
    #     """
    #     Stream chat completion responses
        
    #     Args:
    #         messages: List of ChatMessage objects
    #         model: Model deployment name
    #         temperature: Sampling temperature
    #         max_tokens: Maximum tokens in response
            
    #     Yields:
    #         String chunks of the response
    #     """
    #     try:
    #         response = self.chat_completion(
    #             messages=messages,
    #             model=model,
    #             temperature=temperature,
    #             max_tokens=max_tokens,
    #             stream=True
    #         )
            
    #         for chunk in response["stream"]:
    #             if chunk.choices[0].delta.content is not None:
    #                 yield chunk.choices[0].delta.content
                    
    #     except Exception as e:
    #         logging.error(f"Error in streaming chat completion: {str(e)}")
    #         raise
    
    # def analyze_excel_data(
    #     self, 
    #     excel_records: List[Dict[str, Any]], 
    #     analysis_prompt: str,
    #     model: Optional[str] = None
    # ) -> str:
    #     """
    #     Analyze Excel data using OpenAI
        
    #     Args:
    #         excel_records: List of Excel records from your AzureTableService
    #         analysis_prompt: What kind of analysis to perform
    #         model: Model deployment name
            
    #     Returns:
    #         Analysis results as a string
    #     """
    #     # Prepare the data for analysis
    #     data_preview = json.dumps(excel_records[:5], indent=2, default=str)  # First 5 records
    #     total_records = len(excel_records)
        
    #     system_message = """You are a data analyst expert. Analyze the provided Excel data and provide insights based on the user's request. Be thorough and provide actionable insights."""
        
    #     user_message = f"""
    #     I have {total_records} records from an Excel file. Here's a preview of the first 5 records:
        
    #     {data_preview}
        
    #     Analysis request: {analysis_prompt}
        
    #     Please provide a detailed analysis based on this data.
    #     """
        
    #     return self.simple_chat(
    #         user_message=user_message,
    #         system_message=system_message,
    #         model=model,
    #         temperature=0.3  # Lower temperature for more focused analysis
    #     )
    
    # def generate_embeddings(
    #     self, 
    #     texts: List[str], 
    #     model: str = "text-embedding-ada-002"
    # ) -> List[List[float]]:
    #     """
    #     Generate embeddings for text data
        
    #     Args:
    #         texts: List of texts to embed
    #         model: Embedding model name
            
    #     Returns:
    #         List of embedding vectors
    #     """
    #     try:
    #         response = self.client.embeddings.create(
    #             model=model,
    #             input=texts
    #         )
            
    #         return [data.embedding for data in response.data]
            
    #     except Exception as e:
    #         logging.error(f"Error generating embeddings: {str(e)}")
    #         raise
    
    # def summarize_excel_file(
    #     self, 
    #     excel_records: List[Dict[str, Any]], 
    #     source_file: str,
    #     model: Optional[str] = None
    # ) -> Dict[str, Any]:
    #     """
    #     Generate a summary of an Excel file
        
    #     Args:
    #         excel_records: List of Excel records
    #         source_file: Name of the source file
    #         model: Model deployment name
            
    #     Returns:
    #         Dictionary with summary information
    #     """
    #     if not excel_records:
    #         return {"summary": "No data found in the file", "record_count": 0}
        
    #     # Get column names and data types
    #     sample_record = excel_records[0]
    #     columns = list(sample_record.keys())
        
    #     # Create data sample for analysis
    #     data_sample = json.dumps(excel_records[:3], indent=2, default=str)
        
    #     prompt = f"""
    #     Analyze this Excel file data and provide a structured summary:
        
    #     File: {source_file}
    #     Total Records: {len(excel_records)}
    #     Columns: {', '.join(columns)}
        
    #     Sample data:
    #     {data_sample}
        
    #     Please provide:
    #     1. Brief description of the data
    #     2. Key columns and their apparent purpose
    #     3. Data quality observations
    #     4. Potential use cases for this data
        
    #     Format your response as a clear, structured summary.
    #     """
        
    #     summary_text = self.simple_chat(
    #         user_message=prompt,
    #         system_message="You are a data analyst. Provide clear, concise summaries of datasets.",
    #         model=model,
    #         temperature=0.3
    #     )
        
    #     return {
    #         "summary": summary_text,
    #         "record_count": len(excel_records),
    #         "columns": columns,
    #         "source_file": source_file,
    #         "analyzed_at": datetime.utcnow().isoformat()
    #     }
    
    # def get_model_info(self) -> Dict[str, Any]:
    #     """
    #     Get information about the service configuration
        
    #     Returns:
    #         Dictionary with service information
    #     """
    #     return {
    #         "endpoint": self.endpoint,
    #         "api_version": self.api_version,
    #         "default_model": self.default_model,
    #         "service_type": "Azure OpenAI"
    #     }