from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import os
import tempfile
import json
import jwt
import requests
from datetime import datetime
import traceback
import uvicorn
from cryptography.fernet import Fernet
import base64

# Import your existing modules
# When running from project root (using run_backend.py)
try:
    from backend.utils.query_processor import QueryProcessor
    from backend.utils.groq_client import GroqClient
    from backend.utils.supabase_logger import SupabaseLogger
# When running directly from backend directory
except ModuleNotFoundError:
    import sys
    import os

    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.query_processor import QueryProcessor
    from utils.groq_client import GroqClient
    from utils.supabase_logger import SupabaseLogger


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    file_id: Optional[str] = None
    user_id: str


class QueryResponse(BaseModel):
    success: bool
    result: Any
    generated_code: Optional[str] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    query_id: Optional[str] = None
    rephrased_query: Optional[str] = None


class APIKeyRequest(BaseModel):
    api_key: str
    provider: str = "groq"


class APIKeyResponse(BaseModel):
    success: bool
    message: str
    has_api_key: bool = False


class UserAPIKeyInfo(BaseModel):
    has_api_key: bool
    provider: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class FileUploadResponse(BaseModel):
    success: bool
    file_id: str
    file_name: str
    file_size: int
    columns: List[str]
    shape: List[int]
    preview: Dict[str, Any]
    error: Optional[str] = None


class UserFile(BaseModel):
    id: str
    name: str
    size: int
    upload_date: str
    columns: List[str]
    shape: List[int]


class FeedbackRequest(BaseModel):
    user_id: str
    accuracy_rating: Optional[int] = None
    speed_rating: Optional[int] = None
    overall_rating: Optional[int] = None
    text_feedback: Optional[str] = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    feedback_id: Optional[int] = None


# Initialize FastAPI app
app = FastAPI(title="HorizonAI API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://h-frontend-one.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize your existing components
query_processor = QueryProcessor()
groq_client = None
supabase_logger = SupabaseLogger()

# Initialize encryption for API keys
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key())
if isinstance(ENCRYPTION_KEY, str):
    ENCRYPTION_KEY = ENCRYPTION_KEY.encode()
cipher_suite = Fernet(ENCRYPTION_KEY)

# In-memory storage for user files (in production, use proper database/storage)
user_files: Dict[str, Dict] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global groq_client
    try:
        groq_client = GroqClient()
        print("✅ Groq client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Groq client: {e}")
        groq_client = None


# Security (basic implementation - enhance for production)
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Extract user info from Supabase JWT token"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        token = credentials.credentials

        # First try to decode without verification to get user info
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub")
            email = payload.get("email")

            if user_id:
                print(f"Extracted user from token: {user_id} ({email})")
                return {"sub": user_id, "user_id": user_id, "email": email}
        except Exception as decode_error:
            print(f"Failed to decode token payload: {decode_error}")

        # Get Supabase JWT secret from environment for verification
        supabase_jwt_secret = os.getenv("SUPABASE_JWT_SECRET")

        if not supabase_jwt_secret or supabase_jwt_secret == "your-jwt-secret-here":
            print("Warning: SUPABASE_JWT_SECRET not properly configured")
            raise HTTPException(status_code=401, detail="JWT configuration error")
        else:
            # Verify JWT token with secret
            payload = jwt.decode(
                token,
                supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
            )

            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token payload")

            return {"sub": user_id, "user_id": user_id, "email": payload.get("email")}

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        print(f"JWT decode error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        print(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


# Helper functions for API key management
async def get_user_api_key(user_id: str) -> Optional[str]:
    """Retrieve and decrypt user's API key"""
    try:
        encrypted_key = supabase_logger.get_user_api_key(user_id)
        if encrypted_key and encrypted_key != "No API key found":
            # Try to decrypt user's saved API key
            try:
                return cipher_suite.decrypt(encrypted_key.encode()).decode()
            except Exception as decrypt_error:
                print(f"Decryption failed for user {user_id}: {decrypt_error}")
                return None

        # No API key found for user
        print(f"No API key found for user {user_id}")
        return None

    except Exception as e:
        print(f"Error retrieving API key: {e}")
        return None


# Test endpoint
@app.get("/test")
async def test_endpoint():
    return {"message": "Backend is working"}


# API Key Management Endpoints
@app.post("/api/user/api-key", response_model=APIKeyResponse)
async def save_user_api_key(
    request: APIKeyRequest,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    """Save or update user's API key"""
    print(f"🔥 API key save endpoint called")
    print(f"🔥 Request data: {request}")
    print(f"🔥 Credentials received: {credentials is not None}")
    if credentials:
        print(f"🔥 Token preview: {credentials.credentials[:50]}...")
    try:
        # Require authentication
        if not credentials:
            print("❌ No credentials provided")
            raise HTTPException(status_code=401, detail="Authentication required")

        print("🔥 Attempting to get current user...")
        # Get authenticated user
        user = await get_current_user(credentials)
        print(f"🔥 Got user: {user}")
        user_id = user.get("sub", user.get("user_id"))

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user authentication")

        print(f"Saving API key for user: {user_id}")

        # Encrypt the API key
        encrypted_key = cipher_suite.encrypt(request.api_key.encode()).decode()
        print(f"Encrypted key length: {len(encrypted_key)}")

        # Save to database
        result = supabase_logger.save_user_api_key(
            user_id=user_id, encrypted_api_key=encrypted_key, provider=request.provider
        )

        print(f"Database save result: {result}")

        if result:
            return APIKeyResponse(
                success=True, message="API key saved successfully", has_api_key=True
            )
        else:
            return APIKeyResponse(
                success=False,
                message="Failed to save API key to database",
                has_api_key=False,
            )

    except Exception as e:
        print(f"Error saving API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/api-key/info", response_model=UserAPIKeyInfo)
async def get_user_api_key_info(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    """Get user's API key information (without exposing the key)"""
    try:
        # Require authentication
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        # Get authenticated user
        user = await get_current_user(credentials)
        user_id = user.get("sub", user.get("user_id"))

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user authentication")

        print(f"Getting API key info for user: {user_id}")

        key_info = supabase_logger.get_user_api_key_info(user_id)

        return UserAPIKeyInfo(
            has_api_key=key_info is not None,
            provider=key_info.get("api_provider", "groq") if key_info else "groq",
            created_at=key_info.get("created_at") if key_info else None,
            updated_at=key_info.get("updated_at") if key_info else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting API key info: {e}")
        return UserAPIKeyInfo(has_api_key=False, provider="groq")


@app.delete("/api/user/api-key", response_model=APIKeyResponse)
async def delete_user_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    """Delete user's API key"""
    try:
        # Require authentication
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        # Get authenticated user
        user = await get_current_user(credentials)
        user_id = user.get("sub", user.get("user_id"))

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user authentication")

        print(f"Deleting API key for user: {user_id}")

        result = supabase_logger.delete_user_api_key(user_id)

        if result:
            return APIKeyResponse(
                success=True, message="API key deleted successfully", has_api_key=False
            )
        else:
            return APIKeyResponse(
                success=False, message="Failed to delete API key", has_api_key=True
            )

    except Exception as e:
        print(f"Error deleting API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "HorizonAI API is running",
        "groq_available": groq_client is not None,
        "supabase_available": supabase_logger.enabled,
    }


@app.post("/api/files/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    """Upload and process a CSV/Excel file"""
    try:
        # Validate file type
        if not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
            raise HTTPException(
                status_code=400, detail="Only CSV and Excel files are supported"
            )

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(file.filename)[1]
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Load the file using pandas
            if file.filename.lower().endswith(".csv"):
                df = pd.read_csv(tmp_file_path)
            else:
                df = pd.read_excel(tmp_file_path)

            # Generate file ID
            file_id = (
                f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            )

            # Store file info and DataFrame
            file_info = {
                "id": file_id,
                "name": file.filename,
                "size": len(content),
                "upload_date": datetime.now().isoformat(),
                "columns": df.columns.tolist(),
                "shape": list(df.shape),
                "dataframe": df,  # Store the actual DataFrame
                "user_id": user_id,
            }

            user_files[file_id] = file_info

            # Create preview data
            preview = {
                "columns": df.columns.tolist(),
                "rows": df.head(5).values.tolist(),
                "totalRows": len(df),
                "totalColumns": len(df.columns),
            }

            # Log to Supabase (optional - file upload logging)
            if supabase_logger.enabled:
                try:
                    # You can log file uploads to Supabase here if needed
                    pass
                except Exception as e:
                    print(f"Failed to log file upload: {e}")

            return FileUploadResponse(
                success=True,
                file_id=file_id,
                file_name=file.filename,
                file_size=len(content),
                columns=df.columns.tolist(),
                shape=list(df.shape),
                preview=preview,
            )

        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    except Exception as e:
        print(f"File upload error: {e}")
        return FileUploadResponse(
            success=False,
            file_id="",
            file_name=file.filename or "",
            file_size=0,
            columns=[],
            shape=[0, 0],
            preview={},
            error=str(e),
        )


@app.post("/api/query/process", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    """Process a natural language query using user's API key"""
    start_time = datetime.now()

    try:
        # Validate inputs
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Require authentication
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        # Get authenticated user
        user = await get_current_user(credentials)
        user_id = user.get("sub", user.get("user_id"))

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user authentication")

        print(f"Processing query for user: {user_id}")

        # Get user ID from token
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user")

        # Get user's API key
        user_api_key = await get_user_api_key(user_id)
        if not user_api_key:
            raise HTTPException(
                status_code=400,
                detail="No API key found. Please add your Groq API key in settings.",
            )

        # Get user's file
        if not request.file_id or request.file_id not in user_files:
            raise HTTPException(
                status_code=404, detail="File not found. Please upload a file first."
            )

        file_info = user_files[request.file_id]
        df = file_info["dataframe"]

        # Validate user access to file
        if file_info["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied to this file")

        # Initialize Groq client with user's API key
        user_groq_client = GroqClient(api_key=user_api_key)

        # Process query using user's Groq client
        try:
            # Use the enhanced fuzzy matching if available
            result = query_processor.process_query_with_fuzzy_matching(
                df, request.query, user_groq_client
            )
        except Exception as fuzzy_error:
            print(
                f"Fuzzy matching failed, falling back to basic processing: {fuzzy_error}"
            )
            # Fallback to basic processing
            result = query_processor.process_query(df, request.query, user_groq_client)

        # Check if the result contains a rate limit error
        if not result["success"] and result.get("error"):
            error_msg = result["error"]
            if (
                "RATE_LIMIT_EXCEEDED" in error_msg
                or "rate limit" in error_msg.lower()
                or "too many requests" in error_msg.lower()
            ):
                # Handle rate limit error specially
                execution_time = (datetime.now() - start_time).total_seconds()

                # Log rate limit error to Supabase
                if supabase_logger.enabled:
                    try:
                        supabase_logger.log_query(
                            query=request.query,
                            generated_code="",
                            result=None,
                            error="Rate limit exceeded",
                            success=False,
                            execution_time=execution_time,
                            dataset_info={"error": "rate_limit_exceeded"},
                            user_id=request.user_id,
                        )
                    except Exception as log_error:
                        print(
                            f"Failed to log rate limit error to Supabase: {log_error}"
                        )

                return QueryResponse(
                    success=False,
                    result=None,
                    generated_code="",
                    error="RATE_LIMIT_EXCEEDED: Too many requests. Please wait a few seconds and try again.",
                    execution_time=execution_time,
                )

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        # Prepare result for JSON serialization
        processed_result = None
        if result["success"] and result["result"] is not None:
            if isinstance(result["result"], pd.DataFrame):
                # Check if DataFrame is empty
                if result["result"].empty:
                    processed_result = {
                        "type": "message",
                        "data": "No matching data found for your query. The filter conditions returned an empty result.",
                    }
                else:
                    # Convert DataFrame to JSON-serializable format
                    # Handle NaN values by replacing them with None for JSON serialization
                    df_result = result["result"].head(100).fillna("N/A")
                    processed_result = {
                        "type": "dataframe",
                        "columns": result["result"].columns.tolist(),
                        "data": df_result.values.tolist(),
                        "shape": list(result["result"].shape),
                    }
            elif isinstance(result["result"], pd.Series):
                # Check if Series is empty
                if result["result"].empty:
                    processed_result = {
                        "type": "message",
                        "data": "No matching data found for your query. The filter conditions returned an empty result.",
                    }
                else:
                    # Handle NaN values and convert to JSON-serializable format
                    series_result = result["result"].head(100)
                    # Convert values to native Python types, handling NaN
                    data_values = []
                    for val in series_result.values:
                        if pd.isna(val):
                            data_values.append(None)
                        elif isinstance(val, (np.floating, np.integer)):
                            data_values.append(float(val) if not np.isnan(val) else None)
                        else:
                            data_values.append(val)
                    
                    # Convert index to native Python types
                    index_values = []
                    for idx in series_result.index:
                        if pd.isna(idx):
                            index_values.append(str(idx))
                        else:
                            index_values.append(str(idx) if not isinstance(idx, (int, float)) else idx)
                    
                    processed_result = {
                        "type": "series",
                        "data": data_values,
                        "index": index_values,
                        "name": result["result"].name,
                    }
            elif isinstance(result["result"], pd.Index):
                # Handle pandas Index objects (like column names)
                if len(result["result"]) == 0:
                    processed_result = {
                        "type": "message",
                        "data": "No matching data found for your query.",
                    }
                else:
                    processed_result = {
                        "type": "series",
                        "data": list(range(len(result["result"]))),
                        "index": result["result"].tolist(),
                        "name": "Column Names",
                    }
            else:
                processed_result = {"type": "scalar", "data": str(result["result"])}

        # Prepare dataset info for logging
        dataset_info = {
            "file_id": request.file_id,
            "file_name": file_info["name"],
            "shape": file_info["shape"],
            "columns": file_info["columns"],
        }

        # Log to Supabase with user_id
        query_id = None
        if supabase_logger.enabled:
            try:
                log_result = supabase_logger.log_query(
                    query=request.query,
                    generated_code=result.get("code", ""),
                    result=processed_result,
                    error=result.get("error"),
                    success=result.get("success", False),
                    execution_time=execution_time,
                    dataset_info=dataset_info,
                    user_id=request.user_id,  # Add user_id for logging
                )
                if log_result and "id" in log_result:
                    query_id = str(log_result["id"])
            except Exception as e:
                print(f"Failed to log query to Supabase: {e}")

        return QueryResponse(
            success=result["success"],
            result=processed_result,
            generated_code=result.get("code", ""),
            error=result.get("error"),
            execution_time=execution_time,
            query_id=query_id,
            rephrased_query=result.get("rephrased_query"),
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        execution_time = (datetime.now() - start_time).total_seconds()

        # Check if this is a rate limit error and handle it specially
        if (
            "RATE_LIMIT_EXCEEDED" in error_msg
            or "rate limit" in error_msg.lower()
            or "too many requests" in error_msg.lower()
        ):
            # Log rate limit error to Supabase
            if supabase_logger.enabled:
                try:
                    supabase_logger.log_query(
                        query=request.query,
                        generated_code="",
                        result=None,
                        error="Rate limit exceeded",
                        success=False,
                        execution_time=execution_time,
                        dataset_info={"error": "rate_limit_exceeded"},
                        user_id=request.user_id,
                    )
                except Exception as log_error:
                    print(f"Failed to log rate limit error to Supabase: {log_error}")

            return QueryResponse(
                success=False,
                result=None,
                generated_code="",
                error="RATE_LIMIT_EXCEEDED: Too many requests. Please wait a few seconds and try again.",
                execution_time=execution_time,
            )

        # Log error to Supabase
        if supabase_logger.enabled:
            try:
                supabase_logger.log_query(
                    query=request.query,
                    generated_code="",
                    result=None,
                    error=error_msg,
                    success=False,
                    execution_time=execution_time,
                    dataset_info={"error": "processing_failed"},
                    user_id=request.user_id,
                )
            except Exception as log_error:
                print(f"Failed to log error to Supabase: {log_error}")

        print(f"Query processing error: {e}")
        print(traceback.format_exc())

        return QueryResponse(
            success=False,
            result=None,
            generated_code="",
            error=error_msg,
            execution_time=execution_time,
        )


@app.get("/api/files/{file_id}/datatypes")
async def get_file_datatypes(
    file_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False)),
):
    """Get column datatypes for a specific file"""
    try:
        # Require authentication
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")

        # Get authenticated user
        user = await get_current_user(credentials)
        user_id = user.get("sub", user.get("user_id"))

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid user authentication")

        # Check if file exists
        if file_id not in user_files:
            raise HTTPException(status_code=404, detail="File not found")

        file_info = user_files[file_id]

        # Validate user access to file
        if file_info["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied to this file")

        df = file_info["dataframe"]

        # Get column datatypes
        datatypes = {}
        for column in df.columns:
            dtype = str(df[column].dtype)
            # Simplify datatype names for better readability
            if dtype.startswith("int"):
                simplified_type = "Integer"
            elif dtype.startswith("float"):
                simplified_type = "Float"
            elif dtype == "object":
                # Check if it's actually string or mixed
                if df[column].apply(lambda x: isinstance(x, str)).all():
                    simplified_type = "Text"
                else:
                    simplified_type = "Mixed"
            elif dtype.startswith("datetime"):
                simplified_type = "DateTime"
            elif dtype == "bool":
                simplified_type = "Boolean"
            else:
                simplified_type = dtype.capitalize()

            datatypes[column] = {
                "type": simplified_type,
                "raw_type": dtype,
                "null_count": int(df[column].isnull().sum()),
                "non_null_count": int(df[column].notna().sum()),
            }

        return {"success": True, "file_name": file_info["name"], "datatypes": datatypes}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting file datatypes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/files")
async def get_user_files(user_id: str, current_user: dict = Depends(get_current_user)):
    """Get user's uploaded files"""
    user_file_list = []
    for file_id, file_info in user_files.items():
        if file_info["user_id"] == user_id:
            user_file_list.append(
                {
                    "id": file_info["id"],
                    "name": file_info["name"],
                    "size": file_info["size"],
                    "upload_date": file_info["upload_date"],
                    "columns": file_info["columns"],
                    "shape": file_info["shape"],
                }
            )

    return {"files": user_file_list}


@app.post("/api/feedback")
async def submit_feedback(
    feedback: FeedbackRequest, current_user: dict = Depends(get_current_user)
):
    """Submit or update user feedback"""
    if not supabase_logger.enabled:
        return FeedbackResponse(success=False, message="Feedback system not available")

    try:
        # Validate ratings
        for rating_field in ["accuracy_rating", "speed_rating", "overall_rating"]:
            rating_value = getattr(feedback, rating_field)
            if rating_value is not None and (rating_value < 1 or rating_value > 5):
                print(f"Invalid rating value: {rating_field} = {rating_value}")
                return FeedbackResponse(
                    success=False, message=f"{rating_field} must be between 1 and 5"
                )

        print(f"Attempting to submit feedback for user: {feedback.user_id}")
        print(
            f"Feedback data: accuracy={feedback.accuracy_rating}, speed={feedback.speed_rating}, overall={feedback.overall_rating}"
        )

        # Check if user already has feedback
        existing_feedback = (
            supabase_logger.client.table("feedback")
            .select("*")
            .eq("user_id", feedback.user_id)
            .execute()
        )
        print(f"Existing feedback query result: {existing_feedback}")

        feedback_data = {
            "user_id": feedback.user_id,
            "accuracy_rating": feedback.accuracy_rating,
            "speed_rating": feedback.speed_rating,
            "overall_rating": feedback.overall_rating,
            "text_feedback": feedback.text_feedback,
            "updated_at": datetime.utcnow().isoformat(),
        }

        if existing_feedback.data:
            # Update existing feedback
            print(f"Updating existing feedback for user: {feedback.user_id}")
            result = (
                supabase_logger.client.table("feedback")
                .update(feedback_data)
                .eq("user_id", feedback.user_id)
                .execute()
            )
            print(f"Update result: {result}")
            feedback_id = existing_feedback.data[0]["id"]
            message = "Feedback updated successfully"
        else:
            # Insert new feedback
            print(f"Inserting new feedback for user: {feedback.user_id}")
            feedback_data["created_at"] = datetime.utcnow().isoformat()
            result = (
                supabase_logger.client.table("feedback").insert(feedback_data).execute()
            )
            print(f"Insert result: {result}")
            feedback_id = result.data[0]["id"] if result.data else None
            message = "Feedback submitted successfully"

        print(
            f"Final feedback response: success=True, message={message}, feedback_id={feedback_id}"
        )
        return FeedbackResponse(success=True, message=message, feedback_id=feedback_id)

    except Exception as e:
        print(f"Failed to submit feedback: {e}")
        print(f"Exception details: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return FeedbackResponse(
            success=False, message=f"Failed to submit feedback: {str(e)}"
        )


@app.get("/api/feedback/{user_id}")
async def get_user_feedback(
    user_id: str, current_user: dict = Depends(get_current_user)
):
    """Get user's existing feedback"""
    if not supabase_logger.enabled:
        return {"feedback": None, "message": "Feedback system not available"}

    try:
        print(f"Getting feedback for user: {user_id}")
        result = (
            supabase_logger.client.table("feedback")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )
        print(f"Get feedback result: {result}")

        if result.data:
            print(f"Found existing feedback: {result.data[0]}")
            return {"feedback": result.data[0]}
        else:
            print("No existing feedback found")
            return {"feedback": None}

    except Exception as e:
        print(f"Failed to get user feedback: {e}")
        print(f"Exception details: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"feedback": None, "error": str(e)}


@app.get("/api/query/history")
async def get_query_history(
    user_id: str, limit: int = 50, current_user: dict = Depends(get_current_user)
):
    """Get user's query history from Supabase"""
    if not supabase_logger.enabled:
        return {"queries": [], "message": "Query history not available"}

    try:
        # Get query history from Supabase for this user
        history = supabase_logger.get_query_history(limit=limit, user_id=user_id)
        return {"queries": history or []}
    except Exception as e:
        print(f"Failed to get query history: {e}")
        return {"queries": [], "error": str(e)}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
