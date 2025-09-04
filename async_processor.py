"""
Async Processing Module với ASYNC/AWAIT + SEMAPHORE PATTERN
Thay thế hệ thống parallel processing cũ bằng async processing hiệu quả hơn
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import traceback

# Import config làm default, nhưng cho phép override
from config import (
    MAX_CONCURRENT_REQUESTS as DEFAULT_MAX_CONCURRENT_REQUESTS, 
    ASYNC_BATCH_SIZE as DEFAULT_ASYNC_BATCH_SIZE, 
    ASYNC_CHUNK_SIZE as DEFAULT_ASYNC_CHUNK_SIZE,
    ASYNC_RATE_LIMIT_RPM as DEFAULT_ASYNC_RATE_LIMIT_RPM, 
    ASYNC_TIMEOUT as DEFAULT_ASYNC_TIMEOUT, 
    ASYNC_MAX_RETRIES as DEFAULT_ASYNC_MAX_RETRIES,
    ASYNC_RETRY_DELAY as DEFAULT_ASYNC_RETRY_DELAY, 
    ASYNC_ENABLE_RATE_LIMITER as DEFAULT_ASYNC_ENABLE_RATE_LIMITER
)

# Setup logging
logger = logging.getLogger(__name__)

# Exception classes
class AsyncProcessingError(Exception):
    """Custom exception cho async processing errors"""
    def __init__(self, message: str, status: int = None, response_message: str = None):
        super().__init__(message)
        self.status = status
        self.response_message = response_message

@dataclass
class AsyncProcessingResult:
    """Kết quả xử lý async"""
    success: bool
    result: Any
    error: Optional[str] = None
    processing_time: float = 0.0
    retry_count: int = 0

class AsyncRateLimiter:
    """Dynamic Rate Limiter với async support"""
    
    def __init__(self, rate_limit_rpm: int = None, enable_rate_limiter: bool = None):
        self.rate_limit_rpm = rate_limit_rpm or DEFAULT_ASYNC_RATE_LIMIT_RPM
        self.enable_rate_limiter = enable_rate_limiter if enable_rate_limiter is not None else DEFAULT_ASYNC_ENABLE_RATE_LIMITER
        self.interval = 60.0 / self.rate_limit_rpm  # Seconds between requests
        self._lock = asyncio.Lock()
        self._last_request_time = 0.0
        self._requests_this_minute = 0
        self._minute_start = time.time()
        
        logger.info(f"🎯 AsyncRateLimiter initialized: {self.rate_limit_rpm} RPM (interval: {self.interval:.3f}s)")
    
    async def acquire(self):
        """Chờ cho đến khi có thể gửi request tiếp theo"""
        if not self.enable_rate_limiter:
            return
            
        async with self._lock:
            current_time = time.time()
            
            # Reset counter nếu đã qua 1 phút
            if current_time - self._minute_start >= 60:
                self._requests_this_minute = 0
                self._minute_start = current_time
            
            # Kiểm tra rate limit
            if self._requests_this_minute >= self.rate_limit_rpm:
                wait_time = 60 - (current_time - self._minute_start)
                if wait_time > 0:
                    logger.debug(f"⏳ Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    self._requests_this_minute = 0
                    self._minute_start = time.time()
            
            # Kiểm tra interval giữa requests
            time_since_last = current_time - self._last_request_time
            if time_since_last < self.interval:
                wait_time = self.interval - time_since_last
                logger.debug(f"⏳ Interval wait: {wait_time:.3f}s")
                await asyncio.sleep(wait_time)
            
            self._last_request_time = time.time()
            self._requests_this_minute += 1

class AsyncAPIClient:
    """Async API Client cho Gemini và OpenAI"""
    
    def __init__(self, api_provider: str, api_key: str, model_name: str, 
                 max_concurrent_requests: int = None,
                 rate_limit_rpm: int = None,
                 timeout: int = None,
                 max_retries: int = None,
                 retry_delay: int = None,
                 enable_rate_limiter: bool = None):
        self.api_provider = api_provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        
        # Sử dụng tham số truyền vào hoặc default từ config
        self.max_concurrent_requests = max_concurrent_requests or DEFAULT_MAX_CONCURRENT_REQUESTS
        self.timeout = timeout or DEFAULT_ASYNC_TIMEOUT
        self.max_retries = max_retries or DEFAULT_ASYNC_MAX_RETRIES
        self.retry_delay = retry_delay or DEFAULT_ASYNC_RETRY_DELAY
        
        self.rate_limiter = AsyncRateLimiter(rate_limit_rpm, enable_rate_limiter)
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Setup endpoints và headers
        if self.api_provider == "gemini":
            self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            self.headers = {"Content-Type": "application/json"}
            # Cho Gemini, nếu api_key là placeholder, lấy từ environment hoặc genai config
            if api_key == "from_genai_config":
                import google.generativeai as genai
                # Lấy API key từ genai configure (nếu có)
                self.api_key = getattr(genai, '_client_manager', {}).get('api_key', api_key)
        elif self.api_provider == "openai":
            self.base_url = "https://api.openai.com/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")
        
        logger.info(f"🚀 AsyncAPIClient initialized: {api_provider} - {model_name}")
        logger.info(f"📊 Concurrent limit: {self.max_concurrent_requests}, Rate limit: {self.rate_limiter.rate_limit_rpm} RPM")
    
    def _prepare_gemini_payload(self, prompt: str) -> Dict[str, Any]:
        """Chuẩn bị payload cho Gemini API"""
        return {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.0,  # Giảm temperature để ổn định hơn
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 30000,  # Tăng maxOutputTokens cho JSON phức tạp
                "candidateCount": 1
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        }
    
    def _prepare_openai_payload(self, prompt: str) -> Dict[str, Any]:
        """Chuẩn bị payload cho OpenAI API"""
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 10000
        }
    
    def _extract_response_text(self, response_data: Dict[str, Any]) -> str:
        """Trích xuất text từ response"""
        try:
            if self.api_provider == "gemini":
                # Debug: Log response structure
                logger.debug(f"🔍 Gemini response structure: {list(response_data.keys())}")
                
                # Kiểm tra response structure
                if "candidates" not in response_data:
                    logger.error(f"❌ Gemini response missing 'candidates': {response_data}")
                    return f"API Error: Missing candidates in response"
                
                if not response_data["candidates"]:
                    logger.error(f"❌ Gemini response has empty candidates: {response_data}")
                    return f"API Error: Empty candidates array"
                
                candidate = response_data["candidates"][0]
                if "content" not in candidate:
                    logger.error(f"❌ Gemini candidate missing 'content': {candidate}")
                    return f"API Error: Missing content in candidate"
                
                if "parts" not in candidate["content"]:
                    logger.error(f"❌ Gemini content missing 'parts': {candidate['content']}")
                    return f"API Error: Missing parts in content"
                
                if not candidate["content"]["parts"]:
                    logger.error(f"❌ Gemini parts is empty: {candidate['content']}")
                    return f"API Error: Empty parts array"
                
                text = candidate["content"]["parts"][0]["text"]
                if not text or text.strip() == "":
                    logger.warning("⚠️ Empty response text received from Gemini API")
                    logger.debug(f"🔍 Full Gemini response: {response_data}")
                    return "Empty response"
                
                logger.debug(f"✅ Extracted text from Gemini: {text[:100]}...")
                return text
                
            elif self.api_provider == "openai":
                text = response_data["choices"][0]["message"]["content"]
                if not text or text.strip() == "":
                    logger.warning("⚠️ Empty response text received from OpenAI API")
                    return "Empty response"
                return text
                
        except (KeyError, IndexError) as e:
            logger.error(f"❌ Không thể parse response structure: {e}")
            logger.error(f"🔍 Full response data: {response_data}")
            return f"Parse error: {str(e)}"
        except Exception as e:
            logger.error(f"❌ Unexpected error extracting response text: {e}")
            logger.error(f"🔍 Full response data: {response_data}")
            return f"Extraction error: {str(e)}"
    
    def _parse_response_with_json_support(self, response_text: str, use_json: bool = False) -> str:
        """Parse response với JSON support"""
        if not use_json:
            return response_text.strip()
        
        try:
            # Import JSON utilities động
            from utils import parse_response_with_fallback
            return parse_response_with_fallback(response_text, use_json=True)
        except ImportError:
            # Fallback nếu không import được utils
            logger.warning("⚠️ Không thể import JSON utilities, sử dụng text parsing")
            return response_text.strip()
    
    async def process_single_request(self, prompt: str, item_id: Optional[str] = None, use_json: bool = False) -> AsyncProcessingResult:
        """Xử lý một request đơn lẻ"""
        start_time = time.time()
        retry_count = 0
        
        # Prepare payload
        if self.api_provider == "gemini":
            payload = self._prepare_gemini_payload(prompt)
            url = f"{self.base_url}?key={self.api_key}"
        else:
            payload = self._prepare_openai_payload(prompt)
            url = self.base_url
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting và semaphore
                await self.rate_limiter.acquire()
                
                async with self.semaphore:
                    timeout = aiohttp.ClientTimeout(total=self.timeout)
                    
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(url, headers=self.headers, json=payload) as response:
                            
                            # Xử lý rate limit response
                            if response.status == 429:
                                retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                                logger.warning(f"⚠️ Rate limit hit for {item_id}, waiting {retry_after}s")
                                await asyncio.sleep(retry_after)
                                retry_count += 1
                                continue
                            
                            # Xử lý lỗi HTTP khác
                            if response.status != 200:
                                error_text = await response.text()
                                raise aiohttp.ClientResponseError(
                                    request_info=response.request_info,
                                    history=response.history,
                                    status=response.status,
                                    message=error_text
                                )
                            
                            # Parse response
                            response_data = await response.json()
                            result_text = self._extract_response_text(response_data)
                            
                            # Áp dụng JSON parsing nếu được yêu cầu
                            parsed_result = self._parse_response_with_json_support(result_text, use_json)
                            
                            processing_time = time.time() - start_time
                            logger.debug(f"✅ Success {item_id}: {processing_time:.2f}s, retries: {retry_count}")
                            
                            return AsyncProcessingResult(
                                success=True,
                                result=parsed_result,
                                processing_time=processing_time,
                                retry_count=retry_count
                            )
            
            except asyncio.TimeoutError:
                retry_count += 1
                error_msg = f"Timeout after {self.timeout}s"
                logger.warning(f"⏰ {error_msg} for {item_id} (attempt {attempt + 1})")
                
            except aiohttp.ClientError as e:
                retry_count += 1
                error_msg = f"HTTP error: {str(e)}"
                logger.warning(f"🌐 {error_msg} for {item_id} (attempt {attempt + 1})")
                
            except Exception as e:
                retry_count += 1
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"❌ {error_msg} for {item_id} (attempt {attempt + 1})")
                logger.debug(traceback.format_exc())
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                logger.debug(f"⏳ Retrying {item_id} in {wait_time}s")
                await asyncio.sleep(wait_time)
        
        # Tất cả attempts đều thất bại
        processing_time = time.time() - start_time
        final_error = f"Failed after {self.max_retries} attempts"
        logger.error(f"💥 {final_error} for {item_id}")
        
        return AsyncProcessingResult(
            success=False,
            result=f"Lỗi: {final_error}",
            error=final_error,
            processing_time=processing_time,
            retry_count=retry_count
        )
    
    async def process_batch_request(self, batch_prompt: str, batch_id: str, expected_count: int, use_json: bool = False) -> AsyncProcessingResult:
        """Xử lý batch request - gửi nhiều items trong 1 API call"""
        
        start_time = time.time()
        retry_count = 0
        
        for attempt in range(self.max_retries):
            try:
                # Acquire semaphore và rate limit
                async with self.semaphore:
                    await self.rate_limiter.acquire()
                    
                    # Prepare payload
                    if self.api_provider.lower() == 'gemini':
                        payload = self._prepare_gemini_payload(batch_prompt)
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
                    else:  # OpenAI
                        payload = self._prepare_openai_payload(batch_prompt)
                        url = "https://api.openai.com/v1/chat/completions"
                        headers = {"Authorization": f"Bearer {self.api_key}"}
                    
                    # Make request với timeout
                    timeout = aiohttp.ClientTimeout(total=self.timeout * 2)  # Batch cần timeout lớn hơn
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            url, 
                            json=payload, 
                            headers=getattr(self, 'headers', {})
                        ) as response:
                            
                            if not response.ok:
                                error_text = await response.text()
                                raise AsyncProcessingError(
                                    f"HTTP {response.status}",
                                    status=response.status,
                                    response_message=error_text
                                )
                            
                            # Parse response
                            response_data = await response.json()
                            result_text = self._extract_response_text(response_data)
                            
                            # Parse batch results
                            batch_results = self._parse_batch_response(result_text, expected_count)
                            
                            processing_time = time.time() - start_time
                            logger.debug(f"✅ Batch Success {batch_id}: {processing_time:.2f}s, {len(batch_results)} items, retries: {retry_count}")
                            
                            return AsyncProcessingResult(
                                success=True,
                                result=batch_results,
                                processing_time=processing_time,
                                retry_count=retry_count
                            )
            
            except asyncio.TimeoutError:
                retry_count += 1
                error_msg = f"Batch timeout after {self.timeout * 2}s"
                logger.warning(f"⏰ {error_msg} for {batch_id} (attempt {attempt + 1})")
                
            except aiohttp.ClientError as e:
                retry_count += 1
                error_msg = f"Batch HTTP error: {str(e)}"
                logger.warning(f"🌐 {error_msg} for {batch_id} (attempt {attempt + 1})")
                
            except Exception as e:
                retry_count += 1
                error_msg = f"Batch unexpected error: {str(e)}"
                logger.error(f"❌ {error_msg} for {batch_id} (attempt {attempt + 1})")
                logger.debug(traceback.format_exc())
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)
                logger.debug(f"⏳ Retrying batch {batch_id} in {wait_time}s")
                await asyncio.sleep(wait_time)
        
        # Tất cả attempts đều thất bại
        processing_time = time.time() - start_time
        final_error = f"Batch failed after {self.max_retries} attempts"
        logger.error(f"💥 {final_error} for {batch_id}")
        
        return AsyncProcessingResult(
            success=False,
            result=[f"Lỗi batch: {final_error}"] * expected_count,
            error=final_error,
            processing_time=processing_time,
            retry_count=retry_count
        )
    
    def _parse_batch_response(self, response_text: str, expected_count: int) -> List[str]:
        """Parse batch response thành list results - ƯU TIÊN JSON ARRAY VỚI ITEM_ID"""
        
        try:
            # Clean response text
            response_text = response_text.strip()
            if not response_text:
                logger.warning("⚠️ Empty response text received")
                return [f"Empty response - Batch processing failed"] * expected_count
            
            # Kiểm tra nếu response chỉ chứa "Empty response"
            if response_text.lower() == "empty response":
                logger.warning("⚠️ Batch response contains only 'Empty response'")
                return [f"Empty response - API error"] * expected_count
            
            import re
            import json
            
            # METHOD 1: JSON ARRAY PARSING (MỚI - ƯU TIÊN NHẤT)
            # Tìm JSON array với item_id
            json_array_pattern = r'\[\s*\{.*?\}\s*\]'
            json_array_matches = re.findall(json_array_pattern, response_text, re.DOTALL)
            
            if json_array_matches:
                for array_match in json_array_matches:
                    try:
                        # Parse JSON array
                        json_array = json.loads(array_match)
                        
                        if isinstance(json_array, list) and len(json_array) >= expected_count:
                            # Sort by item_id if available
                            if all('item_id' in item for item in json_array if isinstance(item, dict)):
                                sorted_items = sorted(json_array, key=lambda x: x.get('item_id', 0))
                                
                                # Convert back to JSON strings, remove item_id
                                results = []
                                for item in sorted_items[:expected_count]:
                                    if isinstance(item, dict):
                                        # Remove item_id before returning
                                        clean_item = {k: v for k, v in item.items() if k != 'item_id'}
                                        formatted_json = json.dumps(clean_item, ensure_ascii=False, indent=2)
                                        results.append(formatted_json)
                                    else:
                                        results.append(json.dumps(item, ensure_ascii=False, indent=2))
                                
                                return results
                            else:
                                # JSON array nhưng không có item_id, dùng thứ tự
                                results = []
                                for item in json_array[:expected_count]:
                                    formatted_json = json.dumps(item, ensure_ascii=False, indent=2)
                                    results.append(formatted_json)
                                return results
                    except json.JSONDecodeError:
                        continue
            
            # METHOD 2: INDIVIDUAL JSON OBJECTS PARSING (CŨ)
            # Tìm các JSON objects riêng lẻ trong response
            json_results = []
            
            # Pattern để tìm JSON objects
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            if json_matches:
                # Validate và parse các JSON objects
                for match in json_matches:
                    try:
                        # Try to parse as JSON
                        parsed_json = json.loads(match)
                        
                        # Remove item_id if present (để đảm bảo clean output)
                        if isinstance(parsed_json, dict) and 'item_id' in parsed_json:
                            clean_json = {k: v for k, v in parsed_json.items() if k != 'item_id'}
                            formatted_json = json.dumps(clean_json, ensure_ascii=False, indent=2)
                        else:
                            formatted_json = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                        
                        json_results.append(formatted_json)
                    except json.JSONDecodeError:
                        # Nếu không parse được, thử clean và parse lại
                        try:
                            # Clean common issues
                            cleaned_match = match.strip()
                            # Fix trailing commas
                            cleaned_match = re.sub(r',(\s*[}\]])', r'\1', cleaned_match)
                            # Fix single quotes to double quotes
                            cleaned_match = re.sub(r"'([^']*)':", r'"\1":', cleaned_match)
                            
                            parsed_json = json.loads(cleaned_match)
                            
                            # Remove item_id if present
                            if isinstance(parsed_json, dict) and 'item_id' in parsed_json:
                                clean_json = {k: v for k, v in parsed_json.items() if k != 'item_id'}
                                formatted_json = json.dumps(clean_json, ensure_ascii=False, indent=2)
                            else:
                                formatted_json = json.dumps(parsed_json, ensure_ascii=False, indent=2)
                            
                            json_results.append(formatted_json)
                        except:
                            continue
                
                # Nếu có đủ JSON objects, return
                if len(json_results) >= expected_count:
                    results = json_results[:expected_count]
                    return results
                elif len(json_results) > 0:
                    # Có 1 số JSON nhưng không đủ
                    # Pad thiếu với error messages
                    while len(json_results) < expected_count:
                        json_results.append('{"error": "JSON object missing from batch response"}')
                    return json_results[:expected_count]
            
            # METHOD 3: TEXT PATTERN PARSING (FALLBACK)
            # Try multiple patterns to handle various AI response formats
            patterns = [
                r'Item\s+(\d+):\s*(.+?)(?=\n\s*Item\s+\d+:|$)',  # Standard format with newline
                r'Item\s+(\d+):\s*(.+?)(?=Item\s+\d+:|$)',       # Standard format
                r'(\d+)\.\s*(.+?)(?=\n\s*\d+\.|$)',             # Numbered list format
                r'Item\s*(\d+)\s*[:\-]\s*(.+?)(?=Item\s*\d+|$)', # Flexible separators
            ]
            
            results = []
            for pattern in patterns:
                matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
                
                if matches and len(matches) >= expected_count * 0.7:  # At least 70% matches
                    # Sort theo item number và extract results
                    try:
                        sorted_matches = sorted(matches, key=lambda x: int(x[0]))
                        results = [match[1].strip() for match in sorted_matches]
                        
                        # Clean results - remove common prefixes/suffixes
                        cleaned_results = []
                        for result in results:
                            # Remove common AI response artifacts
                            result = re.sub(r'^(Kết quả|Result):\s*', '', result, flags=re.IGNORECASE)
                            result = re.sub(r'\s*$', '', result)  # Remove trailing whitespace
                            if result:  # Only add non-empty results
                                cleaned_results.append(result)
                        
                        results = cleaned_results
                        break
                        
                    except (ValueError, IndexError):
                        continue
            
            # METHOD 4: Fallback - Split by common delimiters
            if not results or len(results) < expected_count * 0.5:
                # Try splitting by various delimiters
                delimiters = ['\n\n', '\n---', '\n-', '\n•', '\n*', '\n']
                
                for delimiter in delimiters:
                    lines = [line.strip() for line in response_text.split(delimiter) if line.strip()]
                    
                    # Filter out obvious non-result lines
                    filtered_lines = []
                    for line in lines:
                        # Skip lines that look like headers, instructions, etc.
                        if not re.match(r'^(Kết quả|Result|Item|Hãy|Please|Note)', line, re.IGNORECASE):
                            if len(line) > 5:  # Skip very short lines
                                filtered_lines.append(line)
                    
                    if len(filtered_lines) >= expected_count * 0.5:
                        results = filtered_lines[:expected_count]
                        break
            
            # METHOD 5: Last resort - intelligent splitting
            if not results:
                # Remove common prefixes and split intelligently
                cleaned_text = re.sub(r'^.*?(?:Kết quả|Result):\s*', '', response_text, flags=re.IGNORECASE | re.DOTALL)
                
                # Try to split by sentence boundaries that might separate results
                sentences = re.split(r'[.!?]\s+(?=[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ])', cleaned_text)
                
                if len(sentences) >= expected_count:
                    results = [s.strip() for s in sentences[:expected_count]]
                else:
                    # Split the text evenly
                    text_length = len(cleaned_text)
                    chunk_size = max(1, text_length // expected_count)
                    results = []
                    
                    for i in range(expected_count):
                        start = i * chunk_size
                        end = min((i + 1) * chunk_size, text_length)
                        chunk = cleaned_text[start:end].strip()
                        if chunk:
                            results.append(chunk)
                        else:
                            results.append(f"Text chunk {i+1} empty")
            
            # Ensure we have exactly expected_count results
            if len(results) > expected_count:
                results = results[:expected_count]
            elif len(results) < expected_count:
                while len(results) < expected_count:
                    results.append("Parse failed - missing result")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch parsing failed: {str(e)}")
            return [f"Parse error: {str(e)}"] * expected_count

class AsyncDataProcessor:
    """Main async data processor"""
    
    def __init__(self, api_client: AsyncAPIClient):
        self.api_client = api_client
        
    def _prepare_prompt(self, item: Any, prompt_template: str, is_multicolumn: bool = False, column_names: List[str] = None) -> str:
        """Chuẩn bị prompt từ item data"""
        if is_multicolumn and column_names:
            # Multi-column processing
            data_structure = "\n".join([
                f"{i+1}. {col_name}: {str(item.get(col_name, 'N/A'))}"
                for i, col_name in enumerate(column_names)
            ])
            
            return f"""{prompt_template}

DỮ LIỆU CẦN XỬ LÝ:
{data_structure}

Kết quả:"""
        else:
            # Single column processing
            text = str(item) if not isinstance(item, dict) else str(item.get(column_names[0] if column_names else 'MESSAGE', ''))
            return f"""{prompt_template}

Nội dung cần xử lý:
{text}

Kết quả:"""
    
    def _prepare_batch_prompt(self, items: List[Any], prompt_template: str, is_multicolumn: bool = False, column_names: List[str] = None, use_json: bool = False) -> str:
        """Chuẩn bị prompt cho batch items - hỗ trợ JSON với item_id"""
        
        batch_data = []
        for idx, item in enumerate(items):
            if is_multicolumn and column_names:
                # Multi-column processing
                data_structure = "\n".join([
                    f"  {col_name}: {str(item.get(col_name, 'N/A'))}"
                    for col_name in column_names
                ])
                batch_data.append(f"Item {idx + 1}:\n{data_structure}")
            else:
                # Single column processing
                text = str(item) if not isinstance(item, dict) else str(item.get(column_names[0] if column_names else 'MESSAGE', ''))
                batch_data.append(f"Item {idx + 1}: {text}")
        
        batch_text = "\n\n".join(batch_data)
        
        if use_json:
            # JSON format với item_id để track
            return f"""{prompt_template}

XỬ LÝ BATCH {len(items)} ITEMS:
{batch_text}

QUAN TRỌNG: Hãy trả về kết quả cho từng item theo format JSON array:
[
  {{"item_id": 1, ...}},
  {{"item_id": 2, ...}},
  ...,
  {{"item_id": {len(items)}, ...}}
]

Mỗi JSON object phải có field "item_id" để xác định thứ tự.
CHỈ trả về JSON array, KHÔNG có text khác."""
        else:
            # Text format truyền thống
            return f"""{prompt_template}

XỬ LÝ BATCH {len(items)} ITEMS:
{batch_text}

Hãy trả về kết quả cho từng item theo format:
Item 1: [kết quả item 1]
Item 2: [kết quả item 2]
...
Item {len(items)}: [kết quả item {len(items)}]

Kết quả:"""
    
    async def process_batch_async(self, 
                                items: List[Any], 
                                prompt_template: str,
                                is_multicolumn: bool = False,
                                column_names: List[str] = None,
                                use_json: bool = False) -> List[AsyncProcessingResult]:
        """Xử lý batch items với async - SỬ DỤNG ASYNC_BATCH_SIZE"""
        
        logger.info(f"🚀 Starting async batch processing: {len(items)} items với batch size {DEFAULT_ASYNC_BATCH_SIZE}")
        
        # Chia items thành batches theo ASYNC_BATCH_SIZE
        all_results = []
        batches = [items[i:i + DEFAULT_ASYNC_BATCH_SIZE] for i in range(0, len(items), DEFAULT_ASYNC_BATCH_SIZE)]
        
        logger.info(f"📦 Chia thành {len(batches)} batches, mỗi batch {DEFAULT_ASYNC_BATCH_SIZE} items")
        
        # Tạo tasks cho từng batch
        tasks = []
        batch_metadata = []  # Track batch info for proper ordering
        
        for batch_idx, batch in enumerate(batches):
            if DEFAULT_ASYNC_BATCH_SIZE == 1:
                # Single item processing
                for item_idx, item in enumerate(batch):
                    prompt = self._prepare_prompt(item, prompt_template, is_multicolumn, column_names)
                    item_id = f"batch_{batch_idx}_item_{item_idx}"
                    task = self.api_client.process_single_request(prompt, item_id, use_json)
                    tasks.append(task)
                    batch_metadata.append({
                        'type': 'single',
                        'batch_idx': batch_idx,
                        'item_idx': item_idx,
                        'expected_count': 1
                    })
            else:
                # True batch processing - gửi nhiều items trong 1 request
                batch_prompt = self._prepare_batch_prompt(batch, prompt_template, is_multicolumn, column_names, use_json)
                batch_id = f"batch_{batch_idx}"
                task = self.api_client.process_batch_request(batch_prompt, batch_id, len(batch), use_json)
                tasks.append(task)
                batch_metadata.append({
                    'type': 'batch',
                    'batch_idx': batch_idx,
                    'expected_count': len(batch)
                })
        
        # 🔥 FIX: Sử dụng asyncio.gather() thay vì as_completed() để đảm bảo thứ tự
        total_expected_results = len(items)
        logger.info(f"🚀 Starting {len(tasks)} async tasks...")
        
        try:
            # Chạy tất cả tasks đồng thời và đảm bảo thứ tự
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results theo đúng thứ tự
            completed_results = []
            
            for idx, (result, metadata) in enumerate(zip(results, batch_metadata)):
                if isinstance(result, Exception):
                    # Handle exception
                    logger.error(f"❌ Task {idx} failed: {str(result)}")
                    error_count = metadata['expected_count']
                    error_results = [
                        AsyncProcessingResult(success=False, result=f"Task error: {str(result)}")
                        for _ in range(error_count)
                    ]
                    completed_results.extend(error_results)
                else:
                    # Handle successful result
                    if metadata['type'] == 'single':
                        # Single result
                        completed_results.append(result)
                    else:
                        # Batch result - cần parse thành multiple results
                        if result.success and isinstance(result.result, list):
                            batch_results = [
                                AsyncProcessingResult(success=True, result=r) for r in result.result
                            ]
                            completed_results.extend(batch_results)
                        else:
                            # Batch failed, tạo error results
                            error_count = metadata['expected_count']
                            error_results = [
                                AsyncProcessingResult(success=False, result=f"Batch error: {result.error}")
                                for _ in range(error_count)
                            ]
                            completed_results.extend(error_results)
                
                # Log progress
                if len(completed_results) % 50 == 0:
                    success_count = sum(1 for r in completed_results if r.success)
                    logger.info(f"📊 Progress: {len(completed_results)}/{total_expected_results} ({success_count} success)")
            
            # Final validation
            if len(completed_results) != total_expected_results:
                logger.warning(f"⚠️ Result count mismatch: expected {total_expected_results}, got {len(completed_results)}")
                # Pad or trim results to match expected count
                if len(completed_results) < total_expected_results:
                    padding_needed = total_expected_results - len(completed_results)
                    padding_results = [
                        AsyncProcessingResult(success=False, result="Missing result")
                        for _ in range(padding_needed)
                    ]
                    completed_results.extend(padding_results)
                else:
                    completed_results = completed_results[:total_expected_results]
            
            logger.info(f"✅ Async batch processing completed: {len(completed_results)} results")
            return completed_results
            
        except Exception as e:
            logger.error(f"💥 Async batch processing failed: {str(e)}")
            # Return error results for all items
            error_results = [
                AsyncProcessingResult(success=False, result=f"Batch processing error: {str(e)}")
                for _ in range(total_expected_results)
            ]
            return error_results
    
    async def process_data_chunked(self,
                                 data: List[Any],
                                 prompt_template: str,
                                 is_multicolumn: bool = False,
                                 column_names: List[str] = None,
                                 checkpoint_callback=None,
                                 checkpoint_interval: int = None,
                                 use_json: bool = False,
                                 chunk_size: int = None) -> List[str]:
        """Xử lý data theo chunks để tránh memory issues với checkpoint support"""
        
        total_items = len(data)
        chunk_size = chunk_size or DEFAULT_ASYNC_CHUNK_SIZE
        all_results = []
        
        logger.info(f"📦 Processing {total_items} items in chunks of {chunk_size}")
        
        # Chia thành chunks
        chunks = [data[i:i + chunk_size] for i in range(0, total_items, chunk_size)]
        
        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"🔄 Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} items)")
            
            try:
                # Xử lý chunk
                chunk_results = await self.process_batch_async(
                    chunk, prompt_template, is_multicolumn, column_names, use_json
                )
                
                # Extract results
                chunk_texts = [r.result for r in chunk_results]
                all_results.extend(chunk_texts)
                
                # Log chunk summary
                success_count = sum(1 for r in chunk_results if r.success)
                avg_time = sum(r.processing_time for r in chunk_results) / len(chunk_results)
                total_retries = sum(r.retry_count for r in chunk_results)
                
                logger.info(f"✅ Chunk {chunk_idx + 1} completed: {success_count}/{len(chunk)} success, "
                          f"avg time: {avg_time:.2f}s, total retries: {total_retries}")
                
                # 🔥 CHECKPOINT MECHANISM - Lưu tiến trình sau mỗi chunk
                if checkpoint_callback and checkpoint_interval:
                    processed_so_far = (chunk_idx + 1) * chunk_size
                    if processed_so_far % checkpoint_interval == 0 or chunk_idx == len(chunks) - 1:
                        try:
                            # Gọi callback để lưu checkpoint với results hiện tại
                            checkpoint_callback(all_results, chunk_idx + 1, len(chunks))
                            logger.info(f"💾 Checkpoint saved after chunk {chunk_idx + 1}")
                        except Exception as checkpoint_error:
                            logger.warning(f"⚠️ Failed to save checkpoint: {checkpoint_error}")
                
            except Exception as e:
                logger.error(f"❌ Chunk {chunk_idx + 1} failed: {str(e)}")
                # Tạo error results cho chunk này
                error_results = [f"Lỗi chunk: {str(e)}"] * len(chunk)
                all_results.extend(error_results)
                
                # Lưu checkpoint ngay cả khi có lỗi
                if checkpoint_callback:
                    try:
                        checkpoint_callback(all_results, chunk_idx + 1, len(chunks))
                        logger.info(f"💾 Emergency checkpoint saved after chunk {chunk_idx + 1} failure")
                    except Exception as checkpoint_error:
                        logger.warning(f"⚠️ Failed to save emergency checkpoint: {checkpoint_error}")
        
        logger.info(f"🎉 All chunks processed: {len(all_results)} total results")
        return all_results

# Utility functions
async def process_data_async(api_provider: str,
                           api_key: str, 
                           model_name: str,
                           data: List[Any],
                           prompt_template: str,
                           is_multicolumn: bool = False,
                           column_names: List[str] = None,
                           checkpoint_callback=None,
                           checkpoint_interval: int = None,
                           use_json: bool = False,
                           # Thêm các tham số config
                           max_concurrent_requests: int = None,
                           async_batch_size: int = None,
                           async_chunk_size: int = None,
                           rate_limit_rpm: int = None,
                           timeout: int = None,
                           max_retries: int = None,
                           retry_delay: int = None,
                           enable_rate_limiter: bool = None) -> List[str]:
    """Main entry point cho async processing với checkpoint support và config tùy chỉnh"""
    
    try:
        # Initialize client với config tùy chỉnh
        api_client = AsyncAPIClient(
            api_provider, 
            api_key, 
            model_name,
            max_concurrent_requests=max_concurrent_requests,
            rate_limit_rpm=rate_limit_rpm,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            enable_rate_limiter=enable_rate_limiter
        )
        processor = AsyncDataProcessor(api_client)
        
        # Process data với checkpoint support và custom chunk size
        start_time = time.time()
        results = await processor.process_data_chunked(
            data, 
            prompt_template, 
            is_multicolumn, 
            column_names,
            checkpoint_callback=checkpoint_callback,
            checkpoint_interval=checkpoint_interval,
            use_json=use_json,
            chunk_size=async_chunk_size
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"🎉 Async processing hoàn thành trong {total_time:.2f}s")
        logger.info(f"📊 Trung bình: {total_time/len(data):.2f}s/item")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Async processing failed: {str(e)}")
        logger.debug(traceback.format_exc())
        raise 