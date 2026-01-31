import pandas as pd
import numpy as np
import re
import ast
import sys
from io import StringIO
import contextlib
import traceback
from typing import Dict, Any, Optional, List

try:
    from backend.utils.adaptive_fuzzy_matcher import AdaptiveFuzzyMatcher
    ADAPTIVE_FUZZY_AVAILABLE = True
except ImportError:
    ADAPTIVE_FUZZY_AVAILABLE = False

try:
    from backend.utils.entity_classifier import SmartEntityExtractor
    ENTITY_CLASSIFIER_AVAILABLE = True
except ImportError:
    ENTITY_CLASSIFIER_AVAILABLE = False

try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

FUZZY_MATCHING_AVAILABLE = ADAPTIVE_FUZZY_AVAILABLE and ENTITY_CLASSIFIER_AVAILABLE


class QueryProcessor:
    """Process natural language queries using PandasAI with Groq"""
    
    def __init__(self):
        # Initialize the fuzzy matcher if available
        if FUZZY_MATCHING_AVAILABLE:
            self.adaptive_matcher = AdaptiveFuzzyMatcher()
            self.smart_extractor = SmartEntityExtractor()
            self.fuzzy_matching_enabled = True
        else:
            self.adaptive_matcher = None
            self.smart_extractor = None
            self.fuzzy_matching_enabled = False
        
        self.safe_functions = {
            # Pandas functions
            'df', 'pd', 'np', 'len', 'sum', 'mean', 'median', 'std', 'var', 'min', 'max',
            'head', 'tail', 'describe', 'info', 'shape', 'columns', 'dtypes', 'count',
            'groupby', 'sort_values', 'drop_duplicates', 'fillna', 'dropna', 'isnull',
            'value_counts', 'unique', 'nunique', 'corr', 'cov', 'quantile', 'rank',
            'rolling', 'expanding', 'resample', 'pivot_table', 'melt', 'merge', 'concat',
            'apply', 'map', 'filter', 'query', 'loc', 'iloc', 'at', 'iat', 'where',
            'select_dtypes', 'astype', 'copy', 'reset_index', 'set_index', 'reindex',
            
            # Python built-ins (safe ones)
            'abs', 'round', 'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple',
            'range', 'enumerate', 'zip', 'sorted', 'reversed', 'any', 'all', 'print',
            
            # Numpy functions
            'array', 'zeros', 'ones', 'arange', 'linspace', 'log', 'exp', 'sqrt', 'sin', 'cos'
        }
        
        # Built-in functions that should be available
        self.safe_builtins = {
            'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
            'callable', 'chr', 'classmethod', 'complex', 'dict', 'dir', 'divmod',
            'enumerate', 'filter', 'float', 'format', 'frozenset', 'getattr',
            'hasattr', 'hash', 'hex', 'id', 'int', 'isinstance', 'issubclass',
            'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'object',
            'oct', 'ord', 'pow', 'print', 'property', 'range', 'repr',
            'reversed', 'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod',
            'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip'
        }
        
        self.dangerous_patterns = [
            r'import\s+', r'exec\s*\(', r'eval\s*\(', r'__.*__', r'open\s*\(',
            r'file\s*\(', r'input\s*\(', r'raw_input\s*\(', r'compile\s*\(',
            r'reload\s*\(', r'__import__', r'globals\s*\(', r'locals\s*\(',
            r'vars\s*\(', r'dir\s*\(', r'subprocess',
            r'os\.', r'sys\.', r'shutil\.', r'pickle\.', r'marshal\.',
            r'types\.', r'gc\.', r'inspect\.'
        ]
    
    def _fuzzy_match_columns(self, df: pd.DataFrame, user_terms: List[str],
                             min_similarity: int = 70) -> Dict[str, Any]:
        """
        Dynamically match user terms to column names using fuzzy matching.
        Handles misspellings and variations without hardcoding.
        
        Args:
            df: DataFrame to analyze
            user_terms: List of terms user might use
            min_similarity: Minimum similarity score (0-100)
            
        Returns:
            Dict with matched columns and suggestions
        """
        if not FUZZYWUZZY_AVAILABLE:
            return {'matches': {}, 'suggestions': []}
        
        from fuzzywuzzy import fuzz, process
        
        column_matches = {}
        suggestions = []
        
        # Get all column names
        columns = list(df.columns)
        
        for term in user_terms:
            term_lower = term.lower().strip()
            
            # Skip very short terms
            if len(term_lower) < 2:
                continue
            
            # Find best matches using fuzzy matching
            matches = process.extract(term_lower, 
                                    [col.lower() for col in columns], 
                                    limit=3, 
                                    scorer=fuzz.partial_ratio)
            
            best_matches = []
            for match_text, score in matches:
                if score >= min_similarity:
                    # Find the original column name
                    original_col = next(col for col in columns 
                                      if col.lower() == match_text)
                    best_matches.append({
                        'column': original_col,
                        'score': score,
                        'method': 'fuzzy_partial'
                    })
            
            # Also try exact substring matching
            for col in columns:
                if term_lower in col.lower():
                    exact_score = 100 if term_lower == col.lower() else 95
                    best_matches.append({
                        'column': col,
                        'score': exact_score,
                        'method': 'substring'
                    })
            
            # Remove duplicates and sort by score
            seen_columns = set()
            unique_matches = []
            for match in sorted(best_matches, key=lambda x: x['score'], reverse=True):
                if match['column'] not in seen_columns:
                    unique_matches.append(match)
                    seen_columns.add(match['column'])
            
            if unique_matches:
                # Take the best match
                best_match = unique_matches[0]
                
                if best_match['score'] >= 90:
                    # High confidence - auto-map
                    column_matches[term] = best_match['column']
                elif best_match['score'] >= min_similarity:
                    # Medium confidence - suggest
                    suggestions.append({
                        'original_term': term,
                        'suggestions': unique_matches[:3],  # Top 3 matches
                        'message': f"Did you mean one of these for '{term}'?"
                    })
        
        return {
            'matches': column_matches,
            'suggestions': suggestions
        }
    
    def _generate_dynamic_column_mapping(self, df: pd.DataFrame) -> str:
        """
        Generate dynamic column mapping based on actual column names.
        Uses fuzzy matching to handle variations and misspellings.
        """
        if not FUZZYWUZZY_AVAILABLE:
            return "Fuzzy matching not available - using exact column names only"
        
        # Common terms users might use
        common_terms = [
            'protein', 'fat', 'sugar', 'calorie', 'calories', 'carb', 'carbs',
            'name', 'food', 'item', 'product', 'price', 'cost', 'amount',
            'date', 'time', 'category', 'type', 'status', 'city', 'location',
            'income', 'salary', 'payment', 'transaction', 'merchant', 'channel'
        ]
        
        # Get fuzzy matches for these common terms
        match_result = self._fuzzy_match_columns(df, common_terms, min_similarity=60)
        
        mapping_lines = []
        
        # Add successful matches
        if match_result['matches']:
            mapping_lines.append("AUTOMATIC COLUMN MAPPING (for common variations):")
            for term, column in match_result['matches'].items():
                mapping_lines.append(f"- '{term}' -> use column '{column}'")
        
        # Add available columns for reference
        mapping_lines.append("\nAVAILABLE COLUMNS (use exact names):")
        for col in df.columns:
            mapping_lines.append(f"- '{col}'")
        
        # Add fuzzy matching guidance
        mapping_lines.append("\nFUZZY MATCHING AVAILABLE:")
        mapping_lines.append("- System can handle misspellings and variations")
        mapping_lines.append("- Example: 'protien' -> 'Protein_g', 'calries' -> 'Calories'")
        
        return "\n".join(mapping_lines)
    
    def _generate_currency_price_guidance(self, df: pd.DataFrame) -> str:
        """
        Generate specific guidance for currency/price columns that need cleaning
        """
        guidance_lines = []
        
        # Check for price-like columns
        price_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['price', 'cost', 'amount', 'fee', 'charge']):
                # Check if it contains currency symbols or commas
                sample_values = df[col].dropna().astype(str).head(5)
                if any(any(char in str(val) for char in ['₹', '$', '€', '£', ',']) for val in sample_values):
                    price_columns.append({
                        'column': col,
                        'samples': sample_values.tolist()[:3]
                    })
        
        if price_columns:
            guidance_lines.append("CURRENCY/PRICE COLUMN HANDLING:")
            guidance_lines.append("CRITICAL: Some columns contain currency symbols and commas!")
            
            for col_info in price_columns:
                col = col_info['column']
                samples = col_info['samples']
                guidance_lines.append(f"- Column '{col}' contains: {samples}")
                
                # Generate comprehensive cleaning pattern for all currencies
                currencies_found = []
                if any('₹' in str(s) for s in samples): currencies_found.append('₹')
                if any('$' in str(s) for s in samples): currencies_found.append('$')
                if any('€' in str(s) for s in samples): currencies_found.append('€')
                if any('£' in str(s) for s in samples): currencies_found.append('£')
                if any('¥' in str(s) for s in samples): currencies_found.append('¥')
                if any('CHF' in str(s) for s in samples): currencies_found.append('CHF')
                if any('CAD' in str(s) for s in samples): currencies_found.append('CAD')
                
                # Create comprehensive cleaning code
                clean_code = f"df['{col}'].str.replace(r'[₹$€£¥]', '', regex=True)"
                if any(',' in str(s) for s in samples):
                    clean_code += ".str.replace(',', '')"
                if any('CHF' in str(s) or 'CAD' in str(s) for s in samples):
                    clean_code += ".str.replace(r'CHF|CAD', '', regex=True)"
                if any('\u2009' in str(s) for s in samples):
                    clean_code += ".str.replace('\\u2009', '')"
                clean_code += ".astype(float)"
                
                guidance_lines.append(f"  → To use '{col}' in calculations: {clean_code}")
                
                # Add specific examples for common queries
                guidance_lines.append(f"  → For price comparisons: ({clean_code} < 5000)")
                guidance_lines.append(f"  → For currency filtering: df[df['{col}'].str.contains('₹|$|€', regex=True)]")
        
        # Add general numeric handling guidance
        guidance_lines.append("\nNUMERIC DATA HANDLING:")
        guidance_lines.append("CRITICAL: For numeric comparisons, always clean text columns first!")
        guidance_lines.append("Common patterns:")
        guidance_lines.append("- Remove currency: .str.replace(r'[₹$€£¥]', '', regex=True)")
        guidance_lines.append("- Remove commas: .str.replace(',', '')")
        guidance_lines.append("- Remove units: .str.replace(r'[A-Za-z%]+', '', regex=True)")
        guidance_lines.append("- Convert to float: .astype(float)")
        guidance_lines.append("Example: df['Price'].str.replace(r'[₹$€£¥,]', '', regex=True).astype(float) < 5000")
        
        # Check for columns with non-breaking spaces (common in data files)
        unicode_space_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(10)
                if any('\u2009' in str(val) for val in sample_values):
                    unicode_space_columns.append({
                        'column': col,
                        'samples': [val for val in sample_values.tolist()[:3] if '\u2009' in str(val)]
                    })
        
        if unicode_space_columns:
            guidance_lines.append("UNICODE SPACE HANDLING:")
            guidance_lines.append("CRITICAL: Some columns contain non-breaking spaces (\\u2009)!")
            
            for col_info in unicode_space_columns:
                col = col_info['column']
                samples = col_info['samples']
                guidance_lines.append(f"- Column '{col}' contains non-breaking spaces: {samples}")
                guidance_lines.append(f"  → Normalize spaces first: df['{col}'].str.replace('\\u2009', ' ')")
                guidance_lines.append(f"  → For pattern matching: df['{col}'].str.replace('\\u2009', ' ').str.contains('pattern')")
        
        # Check for compound columns (containing multiple comma-separated values)
        compound_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().astype(str).head(10)
                # Check if values contain multiple comma-separated parts
                comma_count = sum(str(val).count(',') for val in sample_values)
                if comma_count > len(sample_values):  # More commas than samples = compound data
                    compound_columns.append({
                        'column': col,
                        'samples': sample_values.tolist()[:2],
                        'pattern_suggestions': self._generate_compound_patterns(col, sample_values)
                    })
        
        # Check for transposed data structure (specifications as rows)
        transposed_structure = self._detect_transposed_structure(df)
        
        if transposed_structure:
            guidance_lines.append("TRANSPOSED DATA STRUCTURE DETECTED:")
            guidance_lines.append("CRITICAL: This data has specifications as ROWS, not columns!")
            guidance_lines.append("Data structure: First column contains specification names")
            guidance_lines.append("Other columns contain values for different products/models")
            guidance_lines.append("")
            guidance_lines.append("CORRECT QUERY PATTERNS:")
            guidance_lines.append("- To get specific spec for a product:")
            guidance_lines.append("  df.loc[df['column_name'].str.contains('SPEC_NAME'), 'PRODUCT_NAME'].iloc[0]")
            guidance_lines.append("- Example for price: df.loc[df['Specification'].str.contains('Price'), 'Product'].iloc[0]")
            guidance_lines.append("- Example for model: df.loc[df['Specification'].str.contains('Model'), 'Product'].iloc[0]")
            guidance_lines.append("")
        
        if compound_columns:
            guidance_lines.append("COMPOUND COLUMN HANDLING:")
            guidance_lines.append("CRITICAL: Some columns contain multiple comma-separated values!")
            guidance_lines.append("IMPORTANT: Use these EXACT patterns for filtering:")
            
            for col_info in compound_columns:
                col = col_info['column']
                samples = col_info['samples']
                patterns = col_info['pattern_suggestions']
                
                guidance_lines.append(f"- Column '{col}' is compound data: {samples}")
                guidance_lines.append("  → REQUIRED patterns (copy exactly):")
                
                for pattern_desc, pattern_code in patterns.items():
                    guidance_lines.append(f"    * {pattern_desc}: {pattern_code}")
            
            # Add specific examples for smartphone data
            guidance_lines.append("")
            guidance_lines.append("SMARTPHONE SPECIFIC EXAMPLES:")
            guidance_lines.append("- For 6GB RAM: df['ram'].str.replace('\\u2009', ' ').str.contains(r'6\\s*GB\\s*RAM', regex=True)")
            guidance_lines.append("- For 5000 mAh: df['battery'].str.replace('\\u2009', ' ').str.contains(r'5000\\s*mAh', regex=True)")
            guidance_lines.append("- For Fast Charging: df['battery'].str.contains('Fast Charging')")
            guidance_lines.append("- For 8GB+128GB: df['ram'].str.replace('\\u2009', ' ').str.contains(r'8\\s*GB\\s*RAM.*128\\s*GB', regex=True)")
        
        if price_columns:
                col = price_columns[0]['column']
                guidance_lines.append(f"# For filtering {col} < 20000:")
                guidance_lines.append(f"df[df['{col}'].str.replace('₹', '').str.replace(',', '').astype(float) < 20000]")
                guidance_lines.append("")
        
        return "\n".join(guidance_lines) if guidance_lines else ""
    
    def _generate_compound_patterns(self, column_name: str, sample_values) -> dict:
        """
        Generate regex patterns for compound columns based on common use cases
        """
        patterns = {}
        
        # Analyze the column name and samples to suggest appropriate patterns
        col_lower = column_name.lower()
        
        if 'ram' in col_lower:
            patterns["For 6GB RAM"] = f"df['{column_name}'].str.replace('\\u2009', ' ').str.contains(r'6\\s*GB\\s*RAM', regex=True)"
            patterns["For 8GB RAM"] = f"df['{column_name}'].str.replace('\\u2009', ' ').str.contains(r'8\\s*GB\\s*RAM', regex=True)"
            patterns["For specific storage"] = f"df['{column_name}'].str.replace('\\u2009', ' ').str.contains(r'128\\s*GB', regex=True)"
            
        elif 'battery' in col_lower:
            patterns["For 5000 mAh"] = f"df['{column_name}'].str.replace('\\u2009', ' ').str.contains(r'5000\\s*mAh', regex=True)"
            patterns["For fast charging"] = f"df['{column_name}'].str.contains('Fast Charging')"
            patterns["For specific wattage"] = f"df['{column_name}'].str.contains(r'\\d+W')"
            
        elif 'sim' in col_lower or 'network' in col_lower:
            patterns["For 5G support"] = f"df['{column_name}'].str.contains('5G')"
            patterns["For dual SIM"] = f"df['{column_name}'].str.contains('Dual Sim')"
            patterns["For WiFi"] = f"df['{column_name}'].str.contains('Wi-Fi')"
            
        elif 'processor' in col_lower or 'cpu' in col_lower:
            patterns["For Snapdragon"] = f"df['{column_name}'].str.contains('Snapdragon')"
            patterns["For specific GHz"] = f"df['{column_name}'].str.contains(r'\\d+\\.\\d+\\s*GHz')"
            patterns["For Octa Core"] = f"df['{column_name}'].str.contains('Octa Core')"
        
        else:
            # Generic patterns for any compound column
            patterns["Contains specific term"] = f"df['{column_name}'].str.contains('TERM')"
            patterns["Multiple terms (AND)"] = f"df['{column_name}'].str.contains('TERM1') & df['{column_name}'].str.contains('TERM2')"
            patterns["Any of multiple terms (OR)"] = f"df['{column_name}'].str.contains('TERM1|TERM2', regex=True)"
        
        return patterns
    
    def _detect_transposed_structure(self, df: pd.DataFrame) -> bool:
        """
        Detect if the DataFrame has a transposed structure (specifications as rows)
        """
        # Check if first column contains specification-like terms
        if len(df.columns) >= 2 and df.shape[0] > 5:
            first_col = df.iloc[:, 0].astype(str)
            
            # Common specification terms
            spec_terms = ['model', 'price', 'processor', 'memory', 'storage', 'display', 
                         'graphics', 'battery', 'weight', 'dimensions', 'specification']
            
            # Check if many values in first column match specification terms
            matches = sum(1 for val in first_col.str.lower() 
                         if any(term in val for term in spec_terms))
            
            # If more than 30% of first column values are specification-like
            return matches > (len(first_col) * 0.3)
        
        return False

    def process_query(self, df: pd.DataFrame, query: str, groq_client) -> Dict[str, Any]:
        """
        Process natural language query and return pandas code and results
        
        Args:
            df: Input DataFrame
            query: Natural language query
            groq_client: Groq client instance
            
        Returns:
            Dict containing success status, result, code, and error if any
        """
        generated_code = ""
        try:
            # Generate pandas code using Groq
            generated_code = self._generate_pandas_code(df, query, groq_client)
            
            if not generated_code:
                return {
                    'success': False,
                    'error': 'Failed to generate pandas code',
                    'code': '',
                    'result': None
                }
            
            # Validate and execute the code
            result = self._execute_safe_code(df, generated_code)
            
            return {
                'success': True,
                'result': result,
                'code': generated_code,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'code': generated_code,
                'result': None
            }
    
    def process_query_with_fuzzy_matching(self, df: pd.DataFrame, query: str, 
                                        groq_client) -> Dict[str, Any]:
        """
        Process query with intelligent fuzzy matching for category names
        
        Args:
            df: Input DataFrame
            query: Natural language query
            groq_client: Groq client instance
            
        Returns:
            Dict containing success status, suggestions, or results
        """
        if not self.fuzzy_matching_enabled:
            # Fallback to regular processing if fuzzy matching not available
            return self.process_query(df, query, groq_client)
        
        try:
            # Step 1: Preprocess query for corrections
            preprocessing_result = self._preprocess_query_fuzzy(query, df)
            
            # Step 2: Check if we need user clarification
            if preprocessing_result['needs_clarification']:
                return {
                    'success': False,
                    'needs_clarification': True,
                    'suggestions': preprocessing_result['suggestions'],
                    'corrected_query': preprocessing_result['corrected_query'],
                    'corrections_made': preprocessing_result['corrections_made']
                }
            
            # Step 3: Use corrected query if available
            final_query = preprocessing_result['corrected_query']
            
            # Step 4: Process the corrected query
            result = self.process_query(df, final_query, groq_client)
            
            # Step 5: Add correction information to result
            if preprocessing_result['corrections_made']:
                result['corrections_made'] = preprocessing_result['corrections_made']
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Enhanced processing failed: {str(e)}",
                'fallback_available': True
            }
    
    def _preprocess_query_fuzzy(self, query: str, df: pd.DataFrame) -> Dict:
        """Preprocess query to fix category names using fuzzy matching"""
        
        corrections = {
            'corrected_query': query,
            'corrections_made': [],
            'suggestions': [],
            'needs_clarification': False
        }
        
        if not self.adaptive_matcher:
            return corrections
        
        # Only run fuzzy matching if query suggests category filtering
        category_indicators = [
            'in', 'from', 'category', 'type', 'kind', 'section', 'of'
        ]
        query_lower = query.lower()
        
        # Skip fuzzy matching if no category indicators present
        has_indicators = any(indicator in query_lower for indicator in category_indicators)
        if not has_indicators:
            return corrections
        
        # Also skip if query contains words that suggest it's not about categories
        non_category_phrases = [
            'name or description', 'their name', 'priced under', 'priced above',
            'have an', 'availability', 'word', 'in their', 'or description'
        ]
        if any(phrase in query_lower for phrase in non_category_phrases):
            return corrections
        
        # Extract potential category mentions using smart classification
        if self.smart_extractor:
            extraction_result = self.smart_extractor.extract_and_classify(query, df)
            category_entities = extraction_result['category_entities']
        else:
            # Fallback to simple extraction if smart extractor not available
            entities = self._extract_entities_from_query(query)
            category_entities = [{'entity': e, 'confidence': 0.5, 'reasons': ['fallback']} for e in entities]
        
        # Skip if no meaningful entities found
        if not category_entities:
            return corrections
        
        for entity_info in category_entities:
            entity = entity_info['entity']
            confidence = entity_info['confidence']
            reasons = entity_info['reasons']
            
            # Only process high-confidence category entities
            if confidence < 0.7:
                continue
            
            # Try to match entity against dataset
            match_result = self.adaptive_matcher.smart_match(entity, df)
            
            if match_result['confidence'] >= 85:
                # High confidence - auto-replace
                if isinstance(match_result['match'], str):
                    corrected_query = corrections['corrected_query'].replace(
                        entity, match_result['match']
                    )
                    corrections['corrected_query'] = corrected_query
                    corrections['corrections_made'].append({
                        'original': entity,
                        'corrected': match_result['match'],
                        'confidence': match_result['confidence'],
                        'method': match_result.get('method', 'unknown')
                    })
                else:
                    # Multiple matches - need clarification
                    corrections['needs_clarification'] = True
                    corrections['suggestions'].append({
                        'type': 'multiple_matches',
                        'original': entity,
                        'matches': match_result['match'],
                        'message': f"Multiple categories match '{entity}'. Choose:"
                    })
                    
            elif match_result['confidence'] >= 60:
                # Medium confidence - suggest
                corrections['needs_clarification'] = True
                corrections['suggestions'].append({
                    'type': 'suggestion',
                    'original': entity,
                    'suggested': match_result['match'],
                    'confidence': match_result['confidence'],
                    'alternatives': match_result.get('suggestions', [])
                })
                
            # If no good match found, just skip this entity
            # No hardcoded suggestions - let the system work with what it has
        
        return corrections
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential category/value entities from query - more selective"""
        
        # More specific patterns focused on actual category references
        patterns = [
            r'(?:in|from|of|category)\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s&\']{3,40})(?:\s+(?:category|section|department|that|where|with|and|\.|,|$))',
            r'(?:category|type|kind|section)\s+(?:of\s+)?([a-zA-Z][a-zA-Z\s&\']{3,40})(?:\s+(?:that|where|with|products|items|and|\.|,|$))',
            r'"([^"]{4,40})"',  # Quoted text - likely specific categories (min 4 chars)
            r'([A-Z][a-z]+(?:\s+&\s+[A-Z][a-z]+)+)',  # Pattern like "Beauty & Personal Care"
        ]
        
        entities = []
        
        # Common non-category words to exclude
        exclude_words = {
            'products', 'items', 'records', 'data', 'all', 'top', 'best', 'most',
            'show', 'find', 'get', 'display', 'list', 'give', 'tell', 'above',
            'below', 'less', 'more', 'than', 'have', 'with', 'their', 'brand',
            'price', 'priced', 'units', 'stock', 'availability', 'status',
            'but', 'and', 'also', 'that', 'are', 'is', 'was', 'were', 'me'
        }
        
        for pattern in patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entity = match.group(1).strip()
                
                # More strict filtering
                entity_words = entity.lower().split()
                
                # Skip if entity contains too many common words
                common_word_count = sum(1 for word in entity_words if word in exclude_words)
                if common_word_count > len(entity_words) / 2:
                    continue
                
                # Only keep entities that look like category names
                if (len(entity) > 2 and len(entity) < 50 and  # Reasonable length
                    not any(char.isdigit() for char in entity) and  # No numbers
                    not any(symbol in entity for symbol in ['$', '%', '>', '<', '=']) and  # No operators
                    len(entity_words) <= 4):  # Not too many words
                    entities.append(entity)
        
        return list(set(entities))
    
    def record_user_choice(self, original_entity: str, chosen_option: str, 
                          available_options: List[str], df: pd.DataFrame):
        """Record user choice for future learning"""
        
        if self.adaptive_matcher:
            dataset_id = self.adaptive_matcher._get_dataset_id(df)
            self.adaptive_matcher.record_user_choice(
                original_entity, 
                chosen_option, 
                available_options, 
                dataset_id
            )
    
    def _generate_pandas_code(self, df: pd.DataFrame, query: str, groq_client) -> str:
        """Generate pandas code using Groq LLM"""
        try:
            # Create context about the DataFrame
            df_info = self._get_dataframe_context(df)
            
            # Create the prompt
            prompt = self._create_prompt(df_info, query)
            
            # Get response from Groq
            response = groq_client.generate_code(prompt)
            
            # Extract and clean the code
            code = self._extract_code_from_response(response)
            
            return code
            
        except Exception as e:
            # Preserve rate limit errors and other specific errors
            error_msg = str(e)
            if "RATE_LIMIT_EXCEEDED" in error_msg:
                # Re-raise rate limit errors as-is to preserve the specific error type
                raise e
            else:
                raise Exception(f"Error generating code: {error_msg}")
    
    def _get_dataframe_context(self, df: pd.DataFrame) -> str:
        """Create context information about the DataFrame"""
        try:
            # Analyze column types more thoroughly
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
            
            # Identify potential date columns in object columns
            potential_date_cols = []
            date_conversion_guide = {}
            for col in text_cols:
                sample_values = df[col].dropna().astype(str).head(3).tolist()
                if any(self._looks_like_date(val) for val in sample_values):
                    potential_date_cols.append(col)
                    # Determine the date format
                    sample_val = sample_values[0] if sample_values else ""
                    if 'T' in sample_val or ':' in sample_val:
                        date_conversion_guide[col] = f"pd.to_datetime(df['{col}'])"
                    else:
                        date_conversion_guide[col] = f"pd.to_datetime(df['{col}'])"
            
            # Remove potential date columns from text columns for analysis
            analysis_text_cols = [col for col in text_cols if col not in potential_date_cols]
            
            context = f"""
DataFrame Information:
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {list(df.columns)}
- Data types: {dict(df.dtypes)}
- Sample data (first 3 rows):
{df.head(3).to_string()}

Column Analysis:
- Numeric columns (use for calculations/correlations): {numeric_cols}
- Boolean columns (can be used for filtering): {bool_cols}
- Text/Categorical columns (use for grouping/filtering): {analysis_text_cols}
- Date columns (stored as strings - need conversion): {potential_date_cols}
- Datetime columns (already converted): {datetime_cols}
- All object columns: {text_cols}

COLUMN MEANINGS AND VALUES:
- Available columns: {list(df.columns)}
- Sample data for reference:
{df.head(2).to_string()}

COLUMN NAME MAPPING (dynamic fuzzy matching for user variations):
{self._generate_dynamic_column_mapping(df)}

{self._generate_currency_price_guidance(df)}

SPECIFIC COLUMN GUIDANCE:
- City: Geographic location of customers (values: {list(df['City'].unique()[:5]) if 'City' in df.columns else 'N/A'}...)
- TransactionType: Type of financial transaction (values: {list(df['TransactionType'].unique()) if 'TransactionType' in df.columns else 'N/A'})
- MerchantCategory: Category of merchant/business (values: {list(df['MerchantCategory'].unique()) if 'MerchantCategory' in df.columns else 'N/A'})
- Channel: Payment method used (values: {list(df['Channel'].unique()) if 'Channel' in df.columns else 'N/A'})
- OverdraftUsed: Whether overdraft was used (values: {list(df['OverdraftUsed'].unique()) if 'OverdraftUsed' in df.columns else 'N/A'})
- Flags: Special transaction flags (values: {list(df['Flags'].unique()) if 'Flags' in df.columns else 'N/A'})

CRITICAL: 
1. USE EXACT COLUMN NAMES FROM THE LIST ABOVE 
2. IF USER MENTIONS 'protein', 'fat', 'sugar' etc., USE THE MAPPED COLUMN NAMES
3. DO NOT GUESS - USE THE COLUMN MAPPING PROVIDED

IMPORTANT DISTINCTIONS:
- For UTILITY BILLS: Use MerchantCategory == 'Utilities' (NOT TransactionType)
- For RENT PAYMENTS: Use MerchantCategory == 'Rent' (NOT TransactionType)
- For PAYMENT METHODS: Use 'Channel' column (Online Banking, Mobile App, ATM, etc.)
- For TRANSACTION TYPES: Use 'TransactionType' column (Payment, Transfer, Deposit, etc.)
- For BUSINESS CATEGORIES: Use 'MerchantCategory' column (Utilities, Restaurants, Healthcare, Rent, etc.)
- For OVERDRAFT FEES: Use Flags == 'Overdraft Fee' OR OverdraftUsed == True
- For BANK PAYMENTS: Use Channel in ['Online Banking', 'Mobile App', 'Branch'] (exclude ATM, Phone)

CORRELATION ANALYSIS GUIDANCE:
- For overdraft frequency: Count or sum OverdraftUsed or count Flags == 'Overdraft Fee'
- For income correlation: Use AnnualIncomeUSD column
- Group by customer (CustomerID) when analyzing frequency patterns
- Calculate correlation between income and overdraft metrics per customer

CRITICAL DATA TYPE RULES:
1. For correlation analysis (.corr()), ONLY use numeric columns: {numeric_cols}
2. NEVER run .corr() on the entire DataFrame - it will fail on string columns
3. Use df.select_dtypes(include=['number']) to get only numeric columns before .corr()
4. For filtering, use categorical columns like: {analysis_text_cols}
5. When working with filtered data, always select numeric columns first: filtered_df.select_dtypes(include=['number']).corr()

STRING MATCHING RULES (CRITICAL FOR CONSISTENCY):
- ALWAYS use case=False for string matching unless specifically asked for exact case
- Use .str.contains('pattern', case=False) for partial text matching
- Use .loc[condition, 'column'] instead of [condition]['column'] for better performance
- For multiple patterns: .str.contains('pattern1|pattern2', case=False, regex=True)

CORRECT STRING MATCHING EXAMPLES:
- df.loc[df['Course'].str.contains('aiml', case=False), 'Name']
- df[df['Department'].str.contains('computer|engineering', case=False)]
- df.loc[df['Category'].str.contains('science', case=False), ['Name', 'Score']]

INCORRECT EXAMPLES (DO NOT USE):
- df[df['Course'].str.contains('aiml')]['Name']  # Missing case=False
- df['Course'].str.contains('AIML')  # Case sensitive - won't match 'aiml'

DATE HANDLING RULES:
6. Date columns {potential_date_cols} are stored as strings and MUST be converted before using .dt accessor
7. ALWAYS convert date strings to datetime first: pd.to_datetime(df['date_column'])
8. For date filtering: pd.to_datetime(df['date_column']).dt.year == 2023
9. For quarters: pd.to_datetime(df['date_column']).dt.quarter == 2
10. NEVER use .dt accessor directly on string columns - convert first!

DATE CONVERSION EXAMPLES:
{chr(10).join([f"- {col}: {conversion}" for col, conversion in date_conversion_guide.items()])}

EXAMPLE: Instead of "df['Date'].dt.year", use "pd.to_datetime(df['Date']).dt.year"
"""
            return context
        except Exception as e:
            return f"Error getting DataFrame context: {str(e)}"
    
    def _looks_like_date(self, value: str) -> bool:
        """Check if a string value looks like a date"""
        if not isinstance(value, str) or len(value) < 6:
            return False
        
        # Common date patterns
        date_patterns = [
            r'\d{1,2}-\d{1,2}-\d{4}',  # dd-mm-yyyy or mm-dd-yyyy
            r'\d{4}-\d{1,2}-\d{1,2}',  # yyyy-mm-dd
            r'\d{1,2}/\d{1,2}/\d{4}',  # dd/mm/yyyy or mm/dd/yyyy
            r'\d{4}/\d{1,2}/\d{1,2}',  # yyyy/mm/dd
            r'\d{1,2}\.\d{1,2}\.\d{4}', # dd.mm.yyyy
        ]
        
        import re
        return any(re.match(pattern, value.strip()) for pattern in date_patterns)
    
    def _create_prompt(self, df_info: str, query: str) -> str:
        """Create prompt for the LLM"""
        prompt = f"""
You are a pandas expert. Given a DataFrame with the following information:

{df_info}

User Query: "{query}"

Generate ONLY the pandas code to answer this query. The DataFrame is already loaded as 'df'.

Requirements:
1. Return ONLY executable pandas code
2. Use 'df' as the DataFrame variable name
3. The code should return a result (DataFrame, Series, or scalar value) - do NOT use print()
4. Do not include any imports or DataFrame loading code
5. Do not use any file operations, system calls, or dangerous functions
6. Keep the code simple and focused on the specific query
7. If creating visualizations, return the data for plotting, not the plot itself
8. For sorting operations, use df.sort_values() with ascending=False for highest values
9. For filtering by multiple criteria, use boolean indexing with & operator
10. Always use .head(n) to limit results when asked for "top N" items
11. AVOID mixing transform and aggregation operations in the same chain
12. For factor analysis or correlations, use ONLY numeric columns - exclude date columns
13. When analyzing factors affecting price, calculate correlations between price and other numeric columns only
14. For "across all categories" queries, do NOT group by category - treat as single dataset
15. For "top N across all categories", sort ALL rows and take top N, not N from each category
16. For "from each category" queries, use groupby().head(N) to get N items per group
17. Always check for proper bracket matching in string filtering operations
18. CRITICAL: For correlation analysis, ALWAYS filter to numeric columns first using .select_dtypes(include=['number'])
19. NEVER use print() statements - return the result directly as a single expression or final variable
20. CRITICAL: For string matching operations, ALWAYS use case=False for case-insensitive matching unless specifically asked for exact case
21. ALWAYS use .str.contains('pattern', case=False) instead of .str.contains('pattern') for text filtering
22. For partial string matching, use case=False to handle variations in capitalization

EXAMPLES:
- Instead of: print(result) -> Use: result
- Instead of: df.corr() -> Use: df.select_dtypes(include=['number']).corr()
- For multi-step: Create variables and return the final one: filtered_df = df[condition]; filtered_df.select_dtypes(include=['number']).corr()
- Instead of: df['Column'].str.contains('text') -> Use: df['Column'].str.contains('text', case=False)
- Instead of: df[df['Course'].str.contains('aiml')]['Name'] -> Use: df.loc[df['Course'].str.contains('aiml', case=False), 'Name']

Your code:
"""
        return prompt
    
    def _extract_code_from_response(self, response: str) -> str:
        """Extract clean pandas code from LLM response"""
        try:
            # Remove common markdown code block indicators
            code = response.strip()
            
            # Remove markdown code blocks
            if code.startswith('```python'):
                code = code[9:]
            elif code.startswith('```'):
                code = code[3:]
            
            if code.endswith('```'):
                code = code[:-3]
            
            # Clean up the code
            code = code.strip()
            
            # Remove any import statements or dangerous code
            lines = code.split('\n')
            clean_lines = []
            
            for line in lines:
                line = line.strip()
                if line and not any(re.search(pattern, line, re.IGNORECASE) for pattern in self.dangerous_patterns):
                    clean_lines.append(line)
            
            # Join lines and ensure it's a single expression if possible
            clean_code = '\n'.join(clean_lines)
            
            # If multiple lines, wrap in parentheses for safe evaluation
            if '\n' in clean_code.strip():
                # For multi-line code, return as-is but validate each line
                return clean_code
            
            return clean_code
            
        except Exception as e:
            raise Exception(f"Error extracting code: {str(e)}")
    
    def _execute_safe_code(self, df: pd.DataFrame, code: str) -> Any:
        """Safely execute pandas code"""
        try:
            # Create safe built-ins dictionary
            safe_builtins_dict = {}
            for name in self.safe_builtins:
                try:
                    if hasattr(__builtins__, name):
                        safe_builtins_dict[name] = getattr(__builtins__, name)
                    elif hasattr(__builtins__, '__dict__') and name in __builtins__.__dict__:
                        safe_builtins_dict[name] = __builtins__.__dict__[name]
                    elif name in globals():
                        safe_builtins_dict[name] = globals()[name]
                    else:
                        # Try to get from builtins module directly
                        import builtins
                        if hasattr(builtins, name):
                            safe_builtins_dict[name] = getattr(builtins, name)
                except:
                    pass
            
            # Create a restricted environment
            safe_globals = {
                'df': df,
                'pd': pd,
                'np': np,
                '__builtins__': safe_builtins_dict
            }
            
            # Add built-in functions directly to globals for easier access
            for name in ['int', 'float', 'str', 'bool', 'len', 'max', 'min', 'sum', 'abs', 'round']:
                if name in safe_builtins_dict:
                    safe_globals[name] = safe_builtins_dict[name]
            
            # Additional safety check
            if any(re.search(pattern, code, re.IGNORECASE) for pattern in self.dangerous_patterns):
                raise Exception("Code contains potentially dangerous operations")
            
            # Determine if code is multi-line or single expression
            is_multiline = '\n' in code.strip() or ';' in code
            
            # Capture stdout for operations that print
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                if is_multiline:
                    # Multi-line code - use exec
                    try:
                        compile(code, '<string>', 'exec')
                    except SyntaxError as e:
                        raise Exception(f"Syntax error in generated code: {str(e)}")
                    
                    # Create a local namespace to capture results
                    local_namespace = {}
                    exec(code, safe_globals, local_namespace)
                    
                    # Check for printed output first
                    printed_output = captured_output.getvalue()
                    if printed_output.strip():
                        return printed_output.strip()
                    
                    # Return the last variable defined, or a meaningful result
                    if local_namespace:
                        # Look for common variable names that contain results
                        result_vars = ['result', 'correlations', 'output', 'answer']
                        for var_name in result_vars:
                            if var_name in local_namespace:
                                return local_namespace[var_name]
                        
                        # Return the last defined variable
                        return list(local_namespace.values())[-1]
                    else:
                        return "Code executed successfully (no return value)"
                        
                else:
                    # Single expression - use eval
                    try:
                        compile(code, '<string>', 'eval')
                    except SyntaxError as e:
                        raise Exception(f"Syntax error in generated code: {str(e)}")
                    
                    # Execute the code
                    result = eval(code, safe_globals, {})
                    
                    # If result is None, check if there was printed output
                    if result is None:
                        printed_output = captured_output.getvalue()
                        if printed_output.strip():
                            result = printed_output.strip()
                    
                    return result
                
            finally:
                sys.stdout = old_stdout
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle specific pandas errors with helpful suggestions
            if "cannot combine transform and aggregation operations" in error_msg:
                raise Exception(
                    "Transform/aggregation error: Try breaking the query into simpler steps. "
                    "For complex queries, use df.assign() for new columns or separate operations."
                )
            elif "KeyError" in error_msg and "column" in error_msg.lower():
                raise Exception(
                    f"Column not found: {error_msg}. Please check column names in your data."
                )
            elif "groupby" in error_msg.lower() and "aggregate" in error_msg.lower():
                raise Exception(
                    "GroupBy error: Use proper aggregation functions like .mean(), .sum(), .max() after groupby."
                )
            elif "could not convert string to float" in error_msg:
                raise Exception(
                    "Data type error: Cannot convert text to numbers. Use .select_dtypes(include=['number']) "
                    "to select only numeric columns before mathematical operations like .corr()"
                )
            elif "Can only use .dt accessor with datetimelike values" in error_msg:
                raise Exception(
                    "Date access error: Cannot use .dt accessor on string columns. "
                    "Convert to datetime first: pd.to_datetime(df['date_column']).dt.year"
                )
            else:
                raise Exception(f"Error executing code: {error_msg}")
    
    def _validate_result(self, result: Any) -> Any:
        """Validate and format the result"""
        try:
            if isinstance(result, (pd.DataFrame, pd.Series)):
                # Limit result size for display
                if hasattr(result, 'shape') and len(result) > 1000:
                    return result.head(1000)
                return result
            
            elif isinstance(result, (int, float, str, bool, type(None))):
                return result
            
            elif isinstance(result, (list, tuple, dict)):
                return str(result)
            
            else:
                return str(result)
                
        except Exception as e:
            return f"Error formatting result: {str(e)}"
