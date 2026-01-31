import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import json

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# Load environment variables
load_dotenv()


class SupabaseLogger:
    """Logger for storing query history and analytics in Supabase"""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        # Use service role key for full database access
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        self.client: Optional[Client] = None
        self.enabled = False
        
        # In-memory storage for testing when database is not available
        self.api_keys_storage = {}
        
        if SUPABASE_AVAILABLE and self.supabase_url and self.supabase_key:
            try:
                self.client = create_client(self.supabase_url, self.supabase_key)
                self.enabled = True
                print("✅ Supabase logger initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize Supabase client: {str(e)}")
                self.enabled = False
        else:
            missing_items = []
            if not SUPABASE_AVAILABLE:
                missing_items.append("supabase package")
            if not self.supabase_url:
                missing_items.append("SUPABASE_URL")
            if not self.supabase_key:
                missing_items.append("SUPABASE_ANON_KEY")
            
            print(f"⚠️ Supabase logging disabled. Missing: {', '.join(missing_items)}")
    
    def log_query(self, 
                  query: str, 
                  generated_code: str, 
                  result: Any = None, 
                  error: str = None, 
                  success: bool = True,
                  execution_time: float = None,
                  dataset_info: Dict = None,
                  user_id: str = None) -> Optional[Dict]:
        """
        Log a query and its results to Supabase
        
        Args:
            query: The natural language query
            generated_code: The pandas code generated
            result: The result (will be stringified if not None)
            error: Error message if any
            success: Whether the query was successful
            execution_time: Time taken to execute in seconds
            dataset_info: Information about the dataset used
            user_id: User ID for the query (required for user-specific logging)
            
        Returns:
            Dict: The inserted record if successful, None otherwise
        """
        if not self.enabled:
            return None
        
        try:
            # Extract file_name from dataset_info
            file_name = None
            if dataset_info and isinstance(dataset_info, dict):
                file_name = dataset_info.get('file_name')
            
            # Prepare the data
            log_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "generated_code": generated_code,
                "success": success,
                "error_message": error,
                "execution_time": execution_time,
                "dataset_info": json.dumps(dataset_info) if dataset_info else None,
                "user_id": user_id,  # Add user_id to the log data
                "file_name": file_name,  # Add file_name from dataset_info
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Convert result to string if it exists
            if result is not None:
                try:
                    if hasattr(result, 'to_string'):
                        # DataFrame or Series
                        log_data["result"] = result.to_string()
                    elif hasattr(result, '__len__') and len(str(result)) > 10000:
                        # Large result - truncate
                        log_data["result"] = str(result)[:10000] + "... [truncated]"
                    else:
                        log_data["result"] = str(result)
                except Exception:
                    log_data["result"] = "[Result too complex to serialize]"
            
            # Insert into Supabase
            response = self.client.table("query_history").insert(log_data).execute()
            
            if response.data:
                return response.data[0] if response.data else None
            else:
                print(f"Failed to log query: {response}")
                return None
                
        except Exception as e:
            print(f"Error logging to Supabase: {str(e)}")
            return None
    
    def get_query_history(self, limit: int = 100, user_id: str = None) -> List[Dict]:
        """
        Retrieve query history from Supabase
        
        Args:
            limit: Maximum number of records to retrieve
            user_id: User ID to filter queries (if provided)
            
        Returns:
            List of query history records
        """
        if not self.enabled:
            return []
        
        try:
            query = self.client.table("query_history").select("*")
            
            # Filter by user_id if provided
            if user_id:
                query = query.eq("user_id", user_id)
            
            response = query.order("timestamp", desc=True).limit(limit).execute()
            return response.data if response.data else []
        except Exception as e:
            print(f"Error retrieving query history: {str(e)}")
            return []
    
    def get_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about query usage
        
        Returns:
            Dictionary with analytics data
        """
        if not self.enabled:
            return {}
        
        try:
            # Get total queries
            total_response = self.client.table("query_history").select("id", count="exact").execute()
            total_queries = len(total_response.data) if total_response.data else 0
            
            # Get success rate
            success_response = self.client.table("query_history").select("success").execute()
            if success_response.data:
                successful = sum(1 for record in success_response.data if record.get("success", False))
                success_rate = (successful / len(success_response.data)) * 100 if success_response.data else 0
            else:
                success_rate = 0
            
            # Get common errors
            error_response = self.client.table("query_history").select("error_message").neq("error_message", None).execute()
            common_errors = {}
            if error_response.data:
                for record in error_response.data:
                    error = record.get("error_message", "Unknown error")
                    # Get first line of error for grouping
                    error_key = error.split('\n')[0][:100] if error else "Unknown"
                    common_errors[error_key] = common_errors.get(error_key, 0) + 1
            
            return {
                "total_queries": total_queries,
                "success_rate": round(success_rate, 2),
                "common_errors": dict(sorted(common_errors.items(), key=lambda x: x[1], reverse=True)[:5])
            }
            
        except Exception as e:
            print(f"Error getting analytics: {str(e)}")
            return {}
    
    def create_table_if_not_exists(self) -> bool:
        """
        Create the query_history table if it doesn't exist
        Note: This requires appropriate permissions in Supabase
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        # Note: Table creation via API requires admin privileges
        # It's better to create the table directly in Supabase dashboard
        print("Please create the 'query_history' table in your Supabase dashboard with the following structure:")
        print("""
        CREATE TABLE query_history (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            query TEXT NOT NULL,
            generated_code TEXT,
            result TEXT,
            success BOOLEAN DEFAULT FALSE,
            error_message TEXT,
            execution_time FLOAT,
            dataset_info JSONB
        );
        """)
        return True

    def save_user_api_key(self, user_id: str, encrypted_api_key: str, provider: str = "groq"):
        """Save or update user's encrypted API key"""
        try:
            # Try database first
            if self.enabled and self.client:
                try:
                    data = {
                        'user_id': user_id,
                        'api_key_encrypted': encrypted_api_key,
                        'api_provider': provider,
                        'updated_at': datetime.now(timezone.utc).isoformat(),
                        'is_active': True
                    }
                    
                    # Use upsert to insert or update
                    result = self.client.table('user_api_keys').upsert(
                        data,
                        on_conflict='user_id,api_provider'
                    ).execute()
                    
                    if len(result.data) > 0:
                        print(f"✅ API key saved to database for user {user_id}")
                        return True
                except Exception as db_error:
                    print(f"Database save failed: {db_error}")
                    # Fall through to in-memory storage
            
            # Use in-memory storage as fallback
            self.api_keys_storage[f"{user_id}_{provider}"] = {
                "api_key_encrypted": encrypted_api_key,
                "api_provider": provider,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "is_active": True
            }
            print(f"✅ API key saved to memory for user {user_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving API key: {e}")
            return False
    
    def get_user_api_key(self, user_id: str, provider: str = "groq") -> Optional[str]:
        """Retrieve user's encrypted API key"""
        try:
            # Try database first
            if self.enabled and self.client:
                try:
                    result = self.client.table('user_api_keys').select('api_key_encrypted').eq(
                        'user_id', user_id
                    ).eq('api_provider', provider).eq('is_active', True).execute()
                    
                    if result.data:
                        return result.data[0]['api_key_encrypted']
                except Exception as db_error:
                    print(f"Database get failed: {db_error}")
                    # Fall through to in-memory storage
            
            # Check in-memory storage
            key = f"{user_id}_{provider}"
            if key in self.api_keys_storage:
                return self.api_keys_storage[key]['api_key_encrypted']
            
            return None
            
        except Exception as e:
            print(f"Error retrieving API key: {e}")
            return None
    
    def get_user_api_key_info(self, user_id: str, provider: str = "groq") -> Optional[Dict]:
        """Get user's API key information (without the key itself)"""
        try:
            # Try database first
            if self.enabled and self.client:
                try:
                    result = self.client.table('user_api_keys').select(
                        'api_provider,created_at,updated_at,is_active'
                    ).eq('user_id', user_id).eq('api_provider', provider).eq('is_active', True).execute()
                    
                    if result.data:
                        return result.data[0]
                except Exception as db_error:
                    print(f"Database get info failed: {db_error}")
                    # Fall through to in-memory storage
            
            # Check in-memory storage
            key = f"{user_id}_{provider}"
            if key in self.api_keys_storage:
                data = self.api_keys_storage[key].copy()
                data.pop('api_key_encrypted', None)  # Remove the actual key
                return data
            
            return None
            
        except Exception as e:
            print(f"Error retrieving API key info: {e}")
            return None
    
    def delete_user_api_key(self, user_id: str, provider: str = "groq") -> bool:
        """Delete user's API key"""
        try:
            # Try database first
            if self.enabled and self.client:
                try:
                    result = self.client.table('user_api_keys').update({
                        'is_active': False,
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    }).eq('user_id', user_id).eq('api_provider', provider).execute()
                    
                    if len(result.data) > 0:
                        print(f"✅ API key deleted from database for user {user_id}")
                        return True
                except Exception as db_error:
                    print(f"Database delete failed: {db_error}")
                    # Fall through to in-memory storage
            
            # Delete from in-memory storage
            key = f"{user_id}_{provider}"
            if key in self.api_keys_storage:
                del self.api_keys_storage[key]
                print(f"✅ API key deleted from memory for user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error deleting API key: {e}")
            return False
