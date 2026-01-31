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

    def _detect_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect the intent of the user's query.
        
        Returns:
            Dict with 'intent' ('data', 'conversational', or 'visualization') and 'confidence'
        """
        query_lower = query.lower().strip()
        
        # Visualization intent indicators
        visualization_patterns = [
            # Direct chart/graph requests
            r'\b(plot|chart|graph|visualize|visualise|visualization|draw|sketch)\b',
            r'\b(bar chart|bar graph|pie chart|pie graph|line chart|line graph)\b',
            r'\b(scatter plot|scatter graph|histogram|heatmap|heat map)\b',
            r'\b(box plot|boxplot|area chart|bubble chart|donut chart)\b',
            # Trend visualization
            r'\b(show .+ trend|trend of|trends? over|over time)\b',
            r'\b(time series|temporal|chronological)\b',
            # Distribution visualization
            r'\b(distribution of|spread of|how .+ distributed)\b',
            # Comparison visualization
            r'\b(compare .+ visually|visual comparison|side by side)\b',
            # Relationship visualization
            r'\b(relationship between|correlation .+ visually|how .+ relates?)\b',
            # Composition visualization
            r'\b(breakdown of|composition of|proportion of|percentage breakdown)\b',
            r'\b(share of|market share|pie of)\b',
        ]
        
        # Conversational/analytical intent indicators
        conversational_patterns = [
            # Summarization
            r'\b(summarize|summary|summarise|overview|describe|explain|what is this|about this)\b',
            r'\b(tell me about|give me an overview|what does this|what\'s this)\b',
            # Insights and analysis
            r'\b(insights?|key findings?|main takeaways?|important points?)\b',
            r'\b(analyze|analyse|analysis|interpret|interpretation)\b',
            # Questions about the dataset itself
            r'\b(what kind of|what type of|what sort of)\s+(data|dataset|information)\b',
            r'\b(purpose of|meant for|used for)\s+(this|the)\s+(data|dataset)\b',
            # Recommendations and suggestions
            r'\b(recommend|suggest|advice|should i|what should)\b',
            # Explanations
            r'\b(why is|how come|reason for|explain why|what causes)\b',
            r'\b(meaning of|definition of|what does .+ mean)\b',
            # Comparisons (narrative)
            r'\b(compare and contrast|differences? between|similarities? between)\b',
            # Trends and patterns (narrative)
            r'\b(overall trend|general pattern|big picture|in general)\b',
            # Help and guidance
            r'\b(help me understand|can you explain|what can i learn)\b',
        ]
        
        # Data/tabular intent indicators
        data_patterns = [
            # Direct data retrieval
            r'\b(show|display|list|get|find|fetch|retrieve|give me|return)\b',
            r'\b(select|filter|where|which|whose)\b',
            # Aggregations
            r'\b(count|sum|average|mean|median|total|max|min|std|variance)\b',
            r'\b(group by|grouped by|per|for each|by each)\b',
            # Sorting and ranking
            r'\b(top \d+|bottom \d+|first \d+|last \d+|highest|lowest|most|least)\b',
            r'\b(sort|order|rank|arrange)\b',
            # Specific column references
            r'\b(column|columns|field|fields|row|rows|record|records)\b',
            # Calculations
            r'\b(calculate|compute|multiply|divide|add|subtract)\b',
            # Specific data operations
            r'\b(unique|distinct|duplicates|missing|null|empty)\b',
            r'\b(greater than|less than|equal to|between|above|below)\b',
            # Statistical
            r'\b(correlation|percentile|quartile)\b',
        ]
        
        # Count matches for each intent
        visualization_score = 0
        conversational_score = 0
        data_score = 0
        
        for pattern in visualization_patterns:
            if re.search(pattern, query_lower):
                visualization_score += 1
        
        for pattern in conversational_patterns:
            if re.search(pattern, query_lower):
                conversational_score += 1
        
        for pattern in data_patterns:
            if re.search(pattern, query_lower):
                data_score += 1
        
        # Special cases that strongly indicate visualization
        strong_visualization = [
            'plot', 'chart', 'graph', 'visualize', 'visualization',
            'bar chart', 'pie chart', 'line chart', 'scatter plot',
            'histogram', 'heatmap', 'show me a chart', 'create a graph',
            'draw a', 'plot the', 'graph of', 'chart showing'
        ]
        
        for phrase in strong_visualization:
            if phrase in query_lower:
                visualization_score += 4
        
        # Special cases that strongly indicate conversational
        strong_conversational = [
            'summarize', 'summary', 'overview', 'describe this', 'about this dataset',
            'what is this dataset', 'tell me about', 'explain this', 'insights',
            'what can i learn', 'purpose of this', 'help me understand'
        ]
        
        for phrase in strong_conversational:
            if phrase in query_lower:
                conversational_score += 3
        
        # Special cases that strongly indicate data query
        strong_data = [
            'show me the data', 'list all', 'get all', 'find all', 'filter by',
            'group by', 'sort by', 'top 10', 'count of', 'sum of',
            'average of', 'where', 'which rows', 'how many', 'table of'
        ]
        
        for phrase in strong_data:
            if phrase in query_lower:
                data_score += 3
        
        # Determine intent based on highest score
        scores = {
            'visualization': visualization_score,
            'conversational': conversational_score,
            'data': data_score
        }
        
        total_score = sum(scores.values())
        if total_score == 0:
            # Default to data query if no clear indicators
            return {'intent': 'data', 'confidence': 0.5, 'scores': scores}
        
        # Find the intent with highest score
        max_intent = max(scores, key=scores.get)
        confidence = scores[max_intent] / total_score
        
        return {'intent': max_intent, 'confidence': confidence, 'scores': scores}
    
    def _determine_chart_type(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """
        Intelligently determine the best chart type based on query and data characteristics.
        
        Returns:
            Dict with chart_type, reasoning, and configuration hints
        """
        query_lower = query.lower()
        
        # Analyze data characteristics
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Check for date-like string columns
        for col in categorical_cols[:]:
            sample = df[col].dropna().head(5).tolist()
            if any(self._looks_like_date(str(v)) for v in sample):
                date_cols.append(col)
                categorical_cols.remove(col)
        
        num_numeric = len(numeric_cols)
        num_categorical = len(categorical_cols)
        num_rows = len(df)
        
        # Explicit chart type requests
        chart_keywords = {
            'pie': ['pie chart', 'pie graph', 'pie of', 'breakdown', 'proportion', 'percentage', 'share'],
            'bar': ['bar chart', 'bar graph', 'bars', 'compare categories', 'comparison'],
            'line': ['line chart', 'line graph', 'trend', 'over time', 'time series', 'progression'],
            'scatter': ['scatter plot', 'scatter graph', 'correlation', 'relationship between', 'x vs y'],
            'histogram': ['histogram', 'distribution', 'frequency', 'spread'],
            'heatmap': ['heatmap', 'heat map', 'correlation matrix', 'correlations'],
            'box': ['box plot', 'boxplot', 'outliers', 'quartiles', 'spread by category'],
            'area': ['area chart', 'stacked area', 'cumulative'],
            'donut': ['donut chart', 'donut graph']
        }
        
        # Check for explicit chart type
        for chart_type, keywords in chart_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return {
                        'chart_type': chart_type,
                        'explicit': True,
                        'reasoning': f"User explicitly requested a {chart_type} chart",
                        'data_suitable': True
                    }
        
        # Smart chart type selection based on query intent and data
        reasoning = []
        
        # Distribution analysis
        if any(word in query_lower for word in ['distribution', 'spread', 'frequency', 'how many']):
            if num_numeric > 0:
                return {
                    'chart_type': 'histogram',
                    'explicit': False,
                    'reasoning': "Distribution analysis of numeric data is best shown with a histogram",
                    'data_suitable': True
                }
        
        # Time-based trends
        if any(word in query_lower for word in ['trend', 'over time', 'time series', 'progression', 'growth']):
            if date_cols or any(word in query_lower for word in ['year', 'month', 'date', 'time']):
                return {
                    'chart_type': 'line',
                    'explicit': False,
                    'reasoning': "Time-based trends are best visualized with a line chart",
                    'data_suitable': True
                }
        
        # Composition/proportion analysis
        if any(word in query_lower for word in ['composition', 'proportion', 'percentage', 'share', 'breakdown']):
            if num_categorical > 0:
                # Pie charts work best with fewer categories (< 8)
                for col in categorical_cols:
                    if df[col].nunique() <= 8:
                        return {
                            'chart_type': 'pie',
                            'explicit': False,
                            'reasoning': "Composition/proportion is best shown with a pie chart (limited categories)",
                            'data_suitable': True
                        }
                return {
                    'chart_type': 'bar',
                    'explicit': False,
                    'reasoning': "Too many categories for pie chart, using horizontal bar chart instead",
                    'data_suitable': True
                }
        
        # Comparison between categories
        if any(word in query_lower for word in ['compare', 'comparison', 'versus', 'vs', 'by category', 'per category']):
            if num_categorical > 0 and num_numeric > 0:
                return {
                    'chart_type': 'bar',
                    'explicit': False,
                    'reasoning': "Category comparison is best shown with a bar chart",
                    'data_suitable': True
                }
        
        # Correlation/relationship analysis
        if any(word in query_lower for word in ['correlation', 'relationship', 'relates', 'vs', 'versus']):
            if num_numeric >= 2:
                return {
                    'chart_type': 'scatter',
                    'explicit': False,
                    'reasoning': "Relationship between numeric variables is best shown with a scatter plot",
                    'data_suitable': True
                }
            if num_numeric > 3:
                return {
                    'chart_type': 'heatmap',
                    'explicit': False,
                    'reasoning': "Multiple correlations are best shown with a correlation heatmap",
                    'data_suitable': True
                }
        
        # Default logic based on data structure
        if num_categorical > 0 and num_numeric > 0:
            # Check number of categories
            for col in categorical_cols:
                unique_count = df[col].nunique()
                if unique_count <= 10:
                    return {
                        'chart_type': 'bar',
                        'explicit': False,
                        'reasoning': f"Categorical data with {unique_count} categories, showing as bar chart",
                        'data_suitable': True
                    }
            return {
                'chart_type': 'bar',
                'explicit': False,
                'reasoning': "Categorical with numeric data, using bar chart (top categories)",
                'data_suitable': True
            }
        
        if num_numeric >= 2:
            return {
                'chart_type': 'scatter',
                'explicit': False,
                'reasoning': "Multiple numeric columns, showing relationship with scatter plot",
                'data_suitable': True
            }
        
        if num_numeric == 1:
            return {
                'chart_type': 'histogram',
                'explicit': False,
                'reasoning': "Single numeric column, showing distribution with histogram",
                'data_suitable': True
            }
        
        # Fallback
        return {
            'chart_type': 'bar',
            'explicit': False,
            'reasoning': "Default chart type for general visualization",
            'data_suitable': True
        }
    
    # ========================================================================
    # NEW VISUALIZATION ARCHITECTURE: 4-PHASE APPROACH
    # Phase 1: Data Analyst (Generate Pandas code)
    # Phase 2: Execution Sandbox (Run code, get clean DataFrame)
    # Phase 3: Visualization Architect (Heuristic chart selection)
    # Phase 4: Renderer (Generate Plotly-compatible JSON)
    # ========================================================================
    
    def _generate_viz_data_code(self, df: pd.DataFrame, query: str, 
                                 groq_client) -> str:
        """
        PHASE 1: Data Analyst
        Ask LLM to write Pandas code to prepare data for visualization.
        The LLM should NOT draw charts - only prepare the data.
        """
        import sys as _sys
        
        # Get column info
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            sample = df[col].dropna().head(3).tolist()
            columns_info.append(f"  - '{col}' ({dtype}, {nunique} unique values, sample: {sample[:3]})")
        
        columns_str = "\n".join(columns_info)
        
        prompt = f"""You are a data analyst. Write Pandas code to prepare data for a visualization.

DATASET SCHEMA:
{columns_str}

Total rows: {len(df)}

USER REQUEST: "{query}"

TASK: Write Python code that creates a variable called `viz_data` - a small, clean DataFrame ready for plotting.

RULES:
1. Start with `viz_data = ` - this variable will be used for charting
2. Use groupby, aggregation, filtering, or binning as needed
3. Keep viz_data small (max 20 rows for bar/pie, 100 for scatter/line)
4. For continuous numeric columns, consider binning with pd.cut()
5. Reset index if using groupby: .reset_index()
6. Convert any interval/period types to strings for JSON compatibility
7. Handle missing values with .dropna() or .fillna()
8. DO NOT import anything - pandas is already available as 'df'
9. DO NOT create any plots - only prepare the data

EXAMPLES:
- "sales by category" → viz_data = df.groupby('Category')['Sales'].sum().reset_index()
- "top 5 products by price" → viz_data = df.nlargest(5, 'Price')[['Name', 'Price']]
- "price distribution" → viz_data = df['Price'].value_counts().head(10).reset_index()
- "temperature vs failure" → temp_bins = pd.cut(df['Temperature'], bins=10); viz_data = df.groupby(temp_bins)['Failure'].mean().reset_index(); viz_data['Temperature'] = viz_data['Temperature'].astype(str)

YOUR CODE (only the code, no explanations):
"""
        
        code = groq_client.generate_code(prompt)
        
        # Clean the code
        if code:
            code = code.strip()
            # Remove markdown code blocks
            if code.startswith('```python'):
                code = code[9:]
            elif code.startswith('```'):
                code = code[3:]
            if code.endswith('```'):
                code = code[:-3]
            code = code.strip()
        
        _sys.stderr.write(f"📊 Phase 1 - Data Prep Code:\n{code}\n")
        _sys.stderr.flush()
        
        return code
    
    def _execute_viz_code(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        """
        PHASE 2: Execution Sandbox
        Execute the generated code and return the viz_data DataFrame.
        Includes safety checks and downsampling.
        """
        import sys as _sys
        
        if not code:
            return None
        
        try:
            # Create execution namespace
            local_vars = {'df': df, 'pd': pd, 'np': np}
            
            # Execute the code
            exec(code, {'pd': pd, 'np': np, '__builtins__': {}}, local_vars)
            
            viz_data = local_vars.get('viz_data')
            
            if viz_data is None:
                _sys.stderr.write("⚠️ Phase 2: viz_data not found in executed code\n")
                _sys.stderr.flush()
                return None
            
            # Convert to DataFrame if it's a Series
            if isinstance(viz_data, pd.Series):
                viz_data = viz_data.reset_index()
            
            # Downsample if too large
            if len(viz_data) > 100:
                _sys.stderr.write(f"⚠️ Phase 2: Downsampling from {len(viz_data)} to 100 rows\n")
                viz_data = viz_data.head(100)
            
            _sys.stderr.write(f"✅ Phase 2 - viz_data shape: {viz_data.shape}\n")
            _sys.stderr.write(f"   Columns: {list(viz_data.columns)}\n")
            _sys.stderr.write(f"   Preview:\n{viz_data.head(5).to_string()}\n")
            _sys.stderr.flush()
            
            return viz_data
            
        except Exception as e:
            _sys.stderr.write(f"❌ Phase 2 - Execution error: {e}\n")
            _sys.stderr.flush()
            return None
    
    def _select_chart_type(self, viz_data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """
        PHASE 3: Visualization Architect
        Use heuristics to determine the best chart type based on the prepared data.
        """
        import sys as _sys
        query_lower = query.lower()
        
        if viz_data is None or len(viz_data) == 0:
            return {'chart_type': 'bar', 'x_col': None, 'y_col': None, 'reasoning': 'Default'}
        
        cols = viz_data.columns.tolist()
        numeric_cols = viz_data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = viz_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Check for explicit chart type in query
        explicit_types = {
            'pie': 'pie', 'donut': 'donut', 'scatter': 'scatter',
            'line': 'line', 'bar': 'bar', 'histogram': 'histogram'
        }
        for keyword, chart_type in explicit_types.items():
            if keyword in query_lower:
                x_col = categorical_cols[0] if categorical_cols else cols[0]
                y_col = numeric_cols[0] if numeric_cols else cols[-1]
                return {'chart_type': chart_type, 'x_col': x_col, 'y_col': y_col, 
                        'reasoning': f'Explicit {chart_type} requested'}
        
        # Heuristic selection based on data types
        if len(cols) >= 2:
            x_col = cols[0]
            y_col = cols[1] if len(cols) > 1 else cols[0]
            
            x_is_numeric = x_col in numeric_cols
            y_is_numeric = y_col in numeric_cols
            
            # Categorical X, Numeric Y → Bar Chart
            if not x_is_numeric and y_is_numeric:
                chart_type = 'bar'
                reasoning = f"Categorical '{x_col}' x Numeric '{y_col}' → Bar Chart"
            
            # Both Numeric → Scatter or Line
            elif x_is_numeric and y_is_numeric:
                # Check if X looks like a sequence (for line chart)
                if 'time' in x_col.lower() or 'date' in x_col.lower() or 'index' in x_col.lower():
                    chart_type = 'line'
                    reasoning = f"Time/sequence '{x_col}' x Numeric '{y_col}' → Line Chart"
                else:
                    chart_type = 'scatter'
                    reasoning = f"Numeric '{x_col}' x Numeric '{y_col}' → Scatter Plot"
            
            # Single column or both categorical
            else:
                chart_type = 'bar'
                reasoning = "Default to Bar Chart"
        
        elif len(cols) == 1:
            x_col = cols[0]
            y_col = 'count'
            if x_col in numeric_cols:
                chart_type = 'histogram'
                reasoning = f"Single numeric column '{x_col}' → Histogram"
            else:
                chart_type = 'bar'
                reasoning = f"Single categorical column '{x_col}' → Bar Chart (counts)"
        
        else:
            chart_type = 'bar'
            x_col, y_col = None, None
            reasoning = "No columns found, default to Bar"
        
        _sys.stderr.write(f"📊 Phase 3 - Chart Selection: {chart_type}\n")
        _sys.stderr.write(f"   X: {x_col}, Y: {y_col}\n")
        _sys.stderr.write(f"   Reasoning: {reasoning}\n")
        _sys.stderr.flush()
        
        return {'chart_type': chart_type, 'x_col': x_col, 'y_col': y_col, 'reasoning': reasoning}
    
    def _generate_chart_config(self, viz_data: pd.DataFrame, chart_info: Dict, 
                                query: str, groq_client) -> Dict[str, Any]:
        """
        PHASE 4: Renderer
        Generate chart configuration (Plotly-compatible JSON) from the prepared data.
        """
        import sys as _sys
        
        chart_type = chart_info['chart_type']
        x_col = chart_info['x_col']
        y_col = chart_info['y_col']
        
        if viz_data is None or len(viz_data) == 0:
            return self._empty_chart_config(chart_type, "No data available")
        
        cols = viz_data.columns.tolist()
        
        # Ensure we have valid columns
        if x_col not in cols:
            x_col = cols[0]
        if y_col not in cols and len(cols) > 1:
            y_col = cols[1]
        elif y_col not in cols:
            y_col = cols[0]
        
        try:
            # Build labels and data based on chart type
            if chart_type in ['bar', 'pie', 'donut', 'line']:
                labels = [str(x)[:40] for x in viz_data[x_col].tolist()]
                
                # Get numeric values
                if y_col in viz_data.columns and y_col != x_col:
                    data = viz_data[y_col].tolist()
                else:
                    # If y_col is same as x_col or not found, use counts
                    data = [1] * len(labels)
                
                # Clean data - ensure all values are JSON-serializable numbers
                data = [round(float(v), 2) if pd.notna(v) and isinstance(v, (int, float, np.number)) else 0 for v in data]
                
                # Generate insights
                if data:
                    max_idx = data.index(max(data))
                    min_idx = data.index(min(data))
                    avg_val = sum(data) / len(data)
                    insights = [
                        f"Highest: {labels[max_idx]} ({data[max_idx]:,.2f})",
                        f"Lowest: {labels[min_idx]} ({data[min_idx]:,.2f})",
                        f"Average: {avg_val:,.2f}"
                    ]
                else:
                    insights = []
                
                config = {
                    'chart_type': chart_type,
                    'title': self._generate_chart_title(query, x_col, y_col),
                    'labels': labels,
                    'datasets': [{
                        'label': str(y_col),
                        'data': data,
                        'insight': insights[0] if insights else ''
                    }],
                    'x_axis_label': str(x_col),
                    'y_axis_label': str(y_col),
                    'insights': insights
                }
            
            elif chart_type == 'scatter':
                # For scatter, we need x,y pairs
                x_vals = viz_data[x_col].tolist()
                y_vals = viz_data[y_col].tolist() if y_col in viz_data.columns else x_vals
                
                scatter_data = []
                for x, y in zip(x_vals, y_vals):
                    if pd.notna(x) and pd.notna(y):
                        scatter_data.append({
                            'x': round(float(x), 2) if isinstance(x, (int, float, np.number)) else 0,
                            'y': round(float(y), 2) if isinstance(y, (int, float, np.number)) else 0
                        })
                
                # Calculate correlation if possible
                try:
                    corr = viz_data[x_col].corr(viz_data[y_col])
                    corr_insight = f"Correlation: {corr:.3f}"
                except:
                    corr_insight = "Correlation: N/A"
                
                config = {
                    'chart_type': 'scatter',
                    'title': self._generate_chart_title(query, x_col, y_col),
                    'labels': [],
                    'datasets': [{
                        'label': f'{x_col} vs {y_col}',
                        'data': scatter_data[:100],  # Limit to 100 points
                        'insight': corr_insight
                    }],
                    'x_axis_label': str(x_col),
                    'y_axis_label': str(y_col),
                    'insights': [corr_insight, f"Showing {len(scatter_data[:100])} data points"]
                }
            
            elif chart_type == 'histogram':
                # For histogram, compute bins
                values = viz_data[x_col].dropna()
                if len(values) > 0:
                    counts, bin_edges = np.histogram(values, bins=10)
                    labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(counts))]
                    data = [int(c) for c in counts]
                else:
                    labels, data = [], []
                
                config = {
                    'chart_type': 'histogram',
                    'title': f'Distribution of {x_col}',
                    'labels': labels,
                    'datasets': [{
                        'label': 'Frequency',
                        'data': data,
                        'insight': f"Total: {sum(data)} values"
                    }],
                    'x_axis_label': str(x_col),
                    'y_axis_label': 'Frequency',
                    'insights': [
                        f"Mean: {values.mean():.2f}" if len(values) > 0 else "No data",
                        f"Std Dev: {values.std():.2f}" if len(values) > 0 else ""
                    ]
                }
            
            else:
                config = self._empty_chart_config(chart_type, "Unknown chart type")
            
            _sys.stderr.write(f"✅ Phase 4 - Chart config generated\n")
            _sys.stderr.write(f"   Labels: {len(config.get('labels', []))}\n")
            _sys.stderr.write(f"   Data points: {len(config.get('datasets', [{}])[0].get('data', []))}\n")
            _sys.stderr.flush()
            
            return config
            
        except Exception as e:
            _sys.stderr.write(f"❌ Phase 4 - Config generation error: {e}\n")
            _sys.stderr.flush()
            return self._empty_chart_config(chart_type, str(e))
    
    def _generate_chart_title(self, query: str, x_col: str, y_col: str) -> str:
        """Generate a descriptive chart title"""
        query_lower = query.lower()
        
        # Try to extract intent from query
        if 'top' in query_lower:
            return f"Top {y_col} by {x_col}"
        elif 'distribution' in query_lower:
            return f"Distribution of {x_col}"
        elif 'vs' in query_lower or 'versus' in query_lower:
            return f"{x_col} vs {y_col}"
        elif 'by' in query_lower:
            return f"{y_col} by {x_col}"
        elif 'relation' in query_lower:
            return f"Relationship: {x_col} and {y_col}"
        else:
            return f"{y_col} by {x_col}"
    
    def _empty_chart_config(self, chart_type: str, message: str) -> Dict[str, Any]:
        """Return an empty chart configuration with error message"""
        return {
            'chart_type': chart_type,
            'title': 'Chart Error',
            'labels': [],
            'datasets': [{'label': 'No Data', 'data': [], 'insight': message}],
            'x_axis_label': '',
            'y_axis_label': '',
            'insights': [message]
        }
    
    def _generate_visualization(self, df: pd.DataFrame, query: str, 
                                groq_client) -> Dict[str, Any]:
        """
        Main visualization pipeline using the 4-phase architecture.
        
        Phase 1: Data Analyst - LLM writes Pandas code to prepare data
        Phase 2: Execution Sandbox - Run code, get clean DataFrame
        Phase 3: Visualization Architect - Heuristic chart selection
        Phase 4: Renderer - Generate chart JSON config
        """
        import sys as _sys
        
        _sys.stderr.write(f"\n{'='*60}\n")
        _sys.stderr.write(f"📊 VISUALIZATION PIPELINE START\n")
        _sys.stderr.write(f"   Query: {query}\n")
        _sys.stderr.write(f"{'='*60}\n")
        _sys.stderr.flush()
        
        try:
            # ============================================================
            # PHASE 1: Data Analyst - Generate Pandas code for data prep
            # ============================================================
            viz_code = self._generate_viz_data_code(df, query, groq_client)
            
            # ============================================================
            # PHASE 2: Execution Sandbox - Execute code, get viz_data
            # ============================================================
            viz_data = self._execute_viz_code(df, viz_code)
            
            # If Phase 2 failed, try a simpler fallback
            if viz_data is None:
                _sys.stderr.write("⚠️ Phase 2 failed, using fallback data prep\n")
                _sys.stderr.flush()
                viz_data = self._fallback_data_prep(df, query)
            
            # ============================================================
            # PHASE 3: Visualization Architect - Select chart type
            # ============================================================
            chart_info = self._select_chart_type(viz_data, query)
            
            # ============================================================
            # PHASE 4: Renderer - Generate chart config
            # ============================================================
            chart_config = self._generate_chart_config(viz_data, chart_info, query, groq_client)
            
            _sys.stderr.write(f"\n{'='*60}\n")
            _sys.stderr.write(f"✅ VISUALIZATION PIPELINE COMPLETE\n")
            _sys.stderr.write(f"{'='*60}\n\n")
            _sys.stderr.flush()
            
            return {
                'success': True,
                'result': chart_config,
                'response_type': 'visualization',
                'code': viz_code,
                'error': None,
                'chart_reasoning': chart_info['reasoning']
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            _sys.stderr.write(f"❌ Visualization pipeline error: {e}\n")
            _sys.stderr.write(f"{error_details}\n")
            _sys.stderr.flush()
            
            return {
                'success': False,
                'result': None,
                'response_type': 'visualization',
                'code': None,
                'error': f"Failed to generate visualization: {str(e)}",
                'error_details': error_details
            }
    
    def _fallback_data_prep(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Fallback data preparation when LLM-generated code fails.
        Uses simple heuristics based on query keywords.
        """
        import sys as _sys
        query_lower = query.lower()
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Find columns mentioned in query
        target_cols = self._find_columns_from_query(df, query)
        
        _sys.stderr.write(f"📊 Fallback data prep\n")
        _sys.stderr.write(f"   Target columns: {target_cols}\n")
        _sys.stderr.flush()
        
        try:
            # Top N pattern
            top_n = self._parse_top_n_query(query)
            if top_n['is_top_n'] and numeric_cols:
                sort_col = None
                label_col = None
                
                for tc in target_cols:
                    if tc in numeric_cols and not sort_col:
                        sort_col = tc
                    elif tc in categorical_cols and not label_col:
                        label_col = tc
                
                if not sort_col:
                    sort_col = numeric_cols[0]
                if not label_col:
                    # Find a name-like column
                    for col in df.columns:
                        if any(x in col.lower() for x in ['name', 'title', 'product', 'item']):
                            label_col = col
                            break
                    if not label_col and categorical_cols:
                        label_col = categorical_cols[0]
                
                n = min(top_n['n'], len(df))
                if top_n['order'] == 'desc':
                    viz_data = df.nlargest(n, sort_col)
                else:
                    viz_data = df.nsmallest(n, sort_col)
                
                if label_col:
                    viz_data = viz_data[[label_col, sort_col]].reset_index(drop=True)
                else:
                    viz_data = viz_data[[sort_col]].reset_index()
                
                return viz_data
            
            # Two columns mentioned - relationship/groupby
            if len(target_cols) >= 2:
                col1, col2 = target_cols[0], target_cols[1]
                
                if col1 in categorical_cols and col2 in numeric_cols:
                    viz_data = df.groupby(col1)[col2].mean().reset_index()
                    return viz_data.head(15)
                elif col1 in numeric_cols and col2 in categorical_cols:
                    viz_data = df.groupby(col2)[col1].mean().reset_index()
                    return viz_data.head(15)
                elif col1 in numeric_cols and col2 in numeric_cols:
                    viz_data = df[[col1, col2]].dropna().head(100)
                    return viz_data
            
            # Single column mentioned
            if target_cols:
                col = target_cols[0]
                if col in categorical_cols:
                    viz_data = df[col].value_counts().head(10).reset_index()
                    viz_data.columns = [col, 'count']
                    return viz_data
                elif col in numeric_cols:
                    # Find a label column
                    for lc in df.columns:
                        if any(x in lc.lower() for x in ['name', 'title', 'product']):
                            viz_data = df.nlargest(10, col)[[lc, col]].reset_index(drop=True)
                            return viz_data
                    viz_data = df[[col]].head(20)
                    return viz_data
            
            # Default: first categorical column counts
            if categorical_cols:
                col = categorical_cols[0]
                viz_data = df[col].value_counts().head(10).reset_index()
                viz_data.columns = [col, 'count']
                return viz_data
            
            # Last resort: first numeric column
            if numeric_cols:
                viz_data = df[numeric_cols[:2]].head(20)
                return viz_data
            
            return df.head(10)
            
        except Exception as e:
            _sys.stderr.write(f"❌ Fallback prep error: {e}\n")
            _sys.stderr.flush()
            return df.head(10)
    
    def _validate_chart_config(self, config: Dict, expected_type: str) -> Dict:
        """Validate and fix chart configuration"""
        
        # Ensure chart_type is set
        if 'chart_type' not in config:
            config['chart_type'] = expected_type
        
        # Ensure title exists
        if 'title' not in config:
            config['title'] = f"{expected_type.capitalize()} Chart"
        
        # Ensure datasets exist and are properly formatted
        if 'datasets' not in config:
            config['datasets'] = []
        
        # Ensure insights exist
        if 'insights' not in config:
            config['insights'] = []
        
        # Validate data values are numbers
        for dataset in config.get('datasets', []):
            if 'data' in dataset:
                cleaned_data = []
                for val in dataset['data']:
                    if isinstance(val, dict):
                        # Scatter plot data point
                        cleaned_data.append({
                            'x': float(val.get('x', 0)) if val.get('x') is not None else 0,
                            'y': float(val.get('y', 0)) if val.get('y') is not None else 0
                        })
                    elif val is None:
                        cleaned_data.append(0)
                    else:
                        try:
                            cleaned_data.append(float(val))
                        except (ValueError, TypeError):
                            cleaned_data.append(0)
                dataset['data'] = cleaned_data
        
        return config
    
    def _find_columns_from_query(self, df: pd.DataFrame, query: str) -> List[str]:
        """
        Find ALL relevant columns mentioned in the query text.
        Returns a list of column names found, in order of appearance.
        """
        query_lower = query.lower()
        columns = df.columns.tolist()
        found_columns = []
        found_positions = []  # Track position in query for ordering
        
        # First pass: exact match (case-insensitive)
        for col in columns:
            col_lower = col.lower()
            pos = query_lower.find(col_lower)
            if pos != -1 and col not in found_columns:
                found_columns.append(col)
                found_positions.append(pos)
        
        # Second pass: partial word match for columns not yet found
        for col in columns:
            if col in found_columns:
                continue
            col_words = col.lower().replace('_', ' ').replace('-', ' ').split()
            for word in col_words:
                if len(word) > 2:
                    pos = query_lower.find(word)
                    if pos != -1:
                        found_columns.append(col)
                        found_positions.append(pos)
                        break
        
        # Third pass: fuzzy matching for remaining columns
        if not found_columns:
            try:
                from rapidfuzz import process
                query_words = query_lower.replace('_', ' ').replace('-', ' ').split()
                for word in query_words:
                    if len(word) > 3:
                        matches = process.extractBests(word, [c.lower() for c in columns], 
                                                       score_cutoff=75, limit=1)
                        if matches:
                            matched_col_lower = matches[0][0]
                            for col in columns:
                                if col.lower() == matched_col_lower and col not in found_columns:
                                    pos = query_lower.find(word)
                                    found_columns.append(col)
                                    found_positions.append(pos)
                                    break
            except:
                pass
        
        # Sort by position in query (earlier mentions first)
        if found_positions:
            sorted_pairs = sorted(zip(found_positions, found_columns))
            found_columns = [col for _, col in sorted_pairs]
        
        return found_columns
    
    def _find_column_from_query(self, df: pd.DataFrame, query: str) -> Optional[str]:
        """
        Find the most relevant column from the query text.
        Returns the first matched column.
        """
        columns = self._find_columns_from_query(df, query)
        return columns[0] if columns else None
    
    def _parse_relationship_query(self, query: str) -> tuple:
        """
        Parse queries like "X vs Y", "relation between X and Y", "X by Y"
        Returns (x_term, y_term) or (None, None) if not a relationship query.
        """
        query_lower = query.lower()
        
        # Patterns for relationship queries
        patterns = [
            r'(\w+)\s+vs\.?\s+(\w+)',  # "X vs Y"
            r'(\w+)\s+versus\s+(\w+)',  # "X versus Y"
            r'relation(?:ship)?\s+(?:between\s+)?(\w+)\s+and\s+(\w+)',  # "relation between X and Y"
            r'(\w+)\s+and\s+(\w+)\s+relation',  # "X and Y relation"
            r'(\w+)\s+by\s+(\w+)',  # "X by Y"
            r'(\w+)\s+against\s+(\w+)',  # "X against Y"
            r'compare\s+(\w+)\s+(?:and|with|to)\s+(\w+)',  # "compare X and Y"
            r'(\w+)\s+over\s+(\w+)',  # "X over Y"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1), match.group(2)
        
        return None, None
    
    def _parse_top_n_query(self, query: str) -> Dict[str, Any]:
        """
        Parse queries like "top 5 products by price", "bottom 10 items with lowest stock"
        Returns dict with: {is_top_n: bool, n: int, order: 'desc'/'asc', sort_column: str, label_column: str}
        """
        query_lower = query.lower()
        result = {'is_top_n': False, 'n': 5, 'order': 'desc', 'sort_column': None, 'label_column': None}
        
        # Patterns for top/bottom N queries
        top_patterns = [
            r'top\s+(\d+)',                          # "top 5"
            r'(\d+)\s+(?:most|highest|largest|best|top)',  # "5 most", "5 highest"
            r'highest\s+(\d+)',                       # "highest 5"
        ]
        bottom_patterns = [
            r'bottom\s+(\d+)',                        # "bottom 5"
            r'(\d+)\s+(?:least|lowest|smallest|worst|bottom)',  # "5 lowest"
            r'lowest\s+(\d+)',                        # "lowest 5"
        ]
        
        # Check for top N patterns
        for pattern in top_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['is_top_n'] = True
                result['n'] = int(match.group(1))
                result['order'] = 'desc'  # Top = descending (highest first)
                break
        
        # Check for bottom N patterns (overrides top if found)
        for pattern in bottom_patterns:
            match = re.search(pattern, query_lower)
            if match:
                result['is_top_n'] = True
                result['n'] = int(match.group(1))
                result['order'] = 'asc'  # Bottom = ascending (lowest first)
                break
        
        # Also check for keywords without explicit number
        if not result['is_top_n']:
            if any(word in query_lower for word in ['top ', 'highest', 'most expensive', 'maximum', 'best']):
                result['is_top_n'] = True
                result['n'] = 5
                result['order'] = 'desc'
            elif any(word in query_lower for word in ['bottom ', 'lowest', 'cheapest', 'minimum', 'worst']):
                result['is_top_n'] = True
                result['n'] = 5
                result['order'] = 'asc'
        
        # Determine sort column from keywords
        if result['is_top_n']:
            sort_keywords = {
                'price': ['price', 'cost', 'expensive', 'cheap', 'costly'],
                'stock': ['stock', 'inventory', 'quantity', 'available'],
                'rating': ['rating', 'rated', 'score', 'review'],
                'sales': ['sales', 'sold', 'revenue', 'selling'],
            }
            for col_hint, keywords in sort_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    result['sort_column'] = col_hint
                    break
        
        return result
    
    def _generate_fallback_chart(self, df: pd.DataFrame, chart_type: str, 
                                  query: str) -> Dict[str, Any]:
        """
        Generate chart data directly from DataFrame when LLM fails.
        This is a fallback mechanism to ensure visualization always works.
        """
        import sys as _sys
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Find ALL columns mentioned in query
        target_columns = self._find_columns_from_query(df, query)
        
        # Also parse relationship patterns like "X vs Y"
        x_term, y_term = self._parse_relationship_query(query)
        
        _sys.stderr.write(f"📊 Fallback chart generation:\n")
        _sys.stderr.write(f"   Chart type: {chart_type}\n")
        _sys.stderr.write(f"   Target columns: {target_columns}\n")
        _sys.stderr.write(f"   Relationship terms: x={x_term}, y={y_term}\n")
        _sys.stderr.flush()
        
        chart_config = {
            'chart_type': chart_type,
            'title': f'{chart_type.capitalize()} Chart - Data Overview',
            'labels': [],
            'datasets': [{'label': 'Value', 'data': [], 'insight': ''}],
            'x_axis_label': '',
            'y_axis_label': '',
            'insights': []
        }
        
        try:
            # CASE 0: Top N / Bottom N query (highest priority)
            top_n_info = self._parse_top_n_query(query)
            
            if top_n_info['is_top_n']:
                _sys.stderr.write(f"   Top N query detected: n={top_n_info['n']}, order={top_n_info['order']}\n")
                _sys.stderr.flush()
                
                # Find the numeric column to sort by
                sort_col = None
                label_col = None
                
                # First, try to match sort_column hint to actual column
                if top_n_info['sort_column']:
                    for col in numeric_cols:
                        if top_n_info['sort_column'] in col.lower():
                            sort_col = col
                            break
                
                # Also check target_columns for the sort column
                if not sort_col and target_columns:
                    for tc in target_columns:
                        if tc in numeric_cols:
                            sort_col = tc
                            break
                
                # Fallback: use first numeric column
                if not sort_col and numeric_cols:
                    sort_col = numeric_cols[0]
                
                # Find label column (usually 'Name', 'Product', 'Title', or first categorical)
                label_candidates = ['name', 'product', 'title', 'item', 'description']
                for col in df.columns:
                    if col.lower() in label_candidates or any(lc in col.lower() for lc in label_candidates):
                        label_col = col
                        break
                
                # Fallback to first categorical or index
                if not label_col and categorical_cols:
                    label_col = categorical_cols[0]
                
                if sort_col:
                    ascending = (top_n_info['order'] == 'asc')
                    n = min(top_n_info['n'], len(df))
                    
                    # Sort and take top/bottom N
                    sorted_df = df.nlargest(n, sort_col) if not ascending else df.nsmallest(n, sort_col)
                    
                    if label_col:
                        labels = [str(x)[:30] for x in sorted_df[label_col].tolist()]  # Truncate long names
                    else:
                        labels = [f"Item {i+1}" for i in range(len(sorted_df))]
                    
                    data = sorted_df[sort_col].tolist()
                    data = [round(float(x), 2) if pd.notna(x) else 0 for x in data]
                    
                    order_word = "Highest" if not ascending else "Lowest"
                    chart_config['labels'] = labels
                    chart_config['datasets'][0]['data'] = data
                    chart_config['datasets'][0]['label'] = sort_col
                    chart_config['title'] = f'Top {n} {order_word} {sort_col}'
                    chart_config['x_axis_label'] = label_col or 'Item'
                    chart_config['y_axis_label'] = sort_col
                    
                    max_val = max(data) if data else 0
                    min_val = min(data) if data else 0
                    avg_val = sum(data) / len(data) if data else 0
                    
                    chart_config['insights'] = [
                        f"Showing top {n} items by {order_word.lower()} {sort_col}",
                        f"Highest: {labels[0] if labels else 'N/A'} ({max_val:,.2f})" if not ascending else f"Lowest: {labels[0] if labels else 'N/A'} ({min_val:,.2f})",
                        f"Average among top {n}: {avg_val:,.2f}"
                    ]
                    
                    _sys.stderr.write(f"   Generated Top N chart: {len(labels)} items by {sort_col}\n")
                    _sys.stderr.flush()
                    return chart_config
            
            # CASE 1: Relationship query with two columns (X vs Y)
            x_col, y_col = None, None
            
            if len(target_columns) >= 2:
                x_col, y_col = target_columns[0], target_columns[1]
            elif x_term and y_term:
                # Match terms to actual columns
                for col in df.columns:
                    col_lower = col.lower()
                    if x_term in col_lower and not x_col:
                        x_col = col
                    elif y_term in col_lower and not y_col:
                        y_col = col
            
            if x_col and y_col:
                _sys.stderr.write(f"   Relationship mode: {x_col} vs {y_col}\n")
                _sys.stderr.flush()
                
                x_is_numeric = x_col in numeric_cols
                y_is_numeric = y_col in numeric_cols
                
                if not x_is_numeric or df[x_col].nunique() <= 20:
                    # X is categorical or has few unique values - group by X, aggregate Y
                    if y_is_numeric:
                        grouped = df.groupby(x_col)[y_col].mean().sort_index()
                        if len(grouped) > 15:
                            grouped = grouped.head(15)
                        
                        chart_config['labels'] = [str(x) for x in grouped.index.tolist()]
                        chart_config['datasets'][0]['data'] = [round(float(x), 2) for x in grouped.values.tolist()]
                        chart_config['datasets'][0]['label'] = f'Average {y_col}'
                        chart_config['title'] = f'{y_col} by {x_col}'
                        chart_config['x_axis_label'] = x_col
                        chart_config['y_axis_label'] = f'Average {y_col}'
                        
                        # Insights
                        max_idx = grouped.idxmax()
                        min_idx = grouped.idxmin()
                        chart_config['insights'] = [
                            f"Highest average {y_col}: {max_idx} ({grouped[max_idx]:,.2f})",
                            f"Lowest average {y_col}: {min_idx} ({grouped[min_idx]:,.2f})",
                            f"Overall average: {df[y_col].mean():,.2f}"
                        ]
                    else:
                        # Both categorical - count combinations
                        grouped = df.groupby(x_col).size().head(15)
                        chart_config['labels'] = [str(x) for x in grouped.index.tolist()]
                        chart_config['datasets'][0]['data'] = [int(x) for x in grouped.values.tolist()]
                        chart_config['title'] = f'Count by {x_col}'
                else:
                    # Both numeric with many values - scatter or binned
                    if chart_type == 'scatter':
                        sample = df[[x_col, y_col]].dropna().head(100)
                        chart_config['datasets'][0]['data'] = [
                            {'x': round(float(row[x_col]), 2), 'y': round(float(row[y_col]), 2)}
                            for _, row in sample.iterrows()
                        ]
                        chart_config['title'] = f'{x_col} vs {y_col}'
                        chart_config['x_axis_label'] = x_col
                        chart_config['y_axis_label'] = y_col
                        corr = df[x_col].corr(df[y_col])
                        chart_config['insights'] = [f"Correlation: {corr:.3f}"]
                    else:
                        # Bin X column for bar/line chart
                        try:
                            df_temp = df[[x_col, y_col]].dropna()
                            df_temp['x_binned'] = pd.cut(df_temp[x_col], bins=10)
                            grouped = df_temp.groupby('x_binned')[y_col].mean()
                            
                            chart_config['labels'] = [str(x) for x in grouped.index.tolist()]
                            chart_config['datasets'][0]['data'] = [round(float(x), 2) for x in grouped.values.tolist()]
                            chart_config['title'] = f'{y_col} by {x_col} Range'
                        except:
                            pass
                
                return chart_config
            
            # CASE 2: Single column query
            target_col = target_columns[0] if target_columns else None
            
            if target_col:
                col_dtype = df[target_col].dtype
                _sys.stderr.write(f"   Single column mode: {target_col} (dtype: {col_dtype})\n")
                _sys.stderr.flush()
                
                unique_count = df[target_col].nunique()
                
                if str(col_dtype) == 'object' or unique_count <= 20:
                    # This is a categorical column - do value counts
                    value_counts = df[target_col].value_counts().head(15)
                    
                    chart_config['labels'] = [str(x) for x in value_counts.index.tolist()]
                    chart_config['datasets'][0]['data'] = [int(x) for x in value_counts.values.tolist()]
                    chart_config['datasets'][0]['label'] = 'Count'
                    chart_config['title'] = f'Distribution of {target_col}'
                    chart_config['x_axis_label'] = target_col
                    chart_config['y_axis_label'] = 'Count'
                    
                    # Generate insights
                    total = value_counts.sum()
                    top_label = value_counts.index[0] if len(value_counts) > 0 else 'N/A'
                    top_count = value_counts.values[0] if len(value_counts) > 0 else 0
                    top_pct = (top_count / total * 100) if total > 0 else 0
                    
                    chart_config['insights'] = [
                        f"Total records: {total:,}",
                        f"Most common: '{top_label}' with {top_count:,} ({top_pct:.1f}%)",
                        f"Number of unique values: {unique_count}"
                    ]
                    
                    _sys.stderr.write(f"   Generated categorical chart with {len(value_counts)} categories\n")
                    _sys.stderr.flush()
                    return chart_config
                
                elif np.issubdtype(col_dtype, np.number):
                    # Numeric column - histogram for distribution
                    if chart_type in ['pie', 'donut', 'bar']:
                        # For pie/bar with numeric, bin the data
                        values = df[target_col].dropna()
                        counts, bin_edges = np.histogram(values, bins=8)
                        labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(counts))]
                        
                        chart_config['labels'] = labels
                        chart_config['datasets'][0]['data'] = [int(x) for x in counts.tolist()]
                        chart_config['datasets'][0]['label'] = 'Frequency'
                        chart_config['title'] = f'Distribution of {target_col}'
                        chart_config['x_axis_label'] = f'{target_col} Range'
                        chart_config['y_axis_label'] = 'Count'
                        
                        chart_config['insights'] = [
                            f"Mean: {values.mean():,.2f}",
                            f"Range: {values.min():,.2f} to {values.max():,.2f}",
                            f"Total records: {len(values):,}"
                        ]
                        return chart_config
            
            # Default fallback logic if no target column found
            if chart_type in ['bar', 'pie', 'donut']:
                # For distribution queries, prefer categorical columns with value_counts
                if categorical_cols:
                    cat_col = categorical_cols[0]
                    value_counts = df[cat_col].value_counts().head(10)
                    
                    chart_config['labels'] = [str(x) for x in value_counts.index.tolist()]
                    chart_config['datasets'][0]['data'] = [int(x) for x in value_counts.values.tolist()]
                    chart_config['datasets'][0]['label'] = 'Count'
                    chart_config['title'] = f'Distribution of {cat_col}'
                    chart_config['x_axis_label'] = cat_col
                    chart_config['y_axis_label'] = 'Count'
                    
                    total = value_counts.sum()
                    top_label = value_counts.index[0] if len(value_counts) > 0 else 'N/A'
                    top_count = value_counts.values[0] if len(value_counts) > 0 else 0
                    top_pct = (top_count / total * 100) if total > 0 else 0
                    
                    chart_config['insights'] = [
                        f"Total records: {total:,}",
                        f"Most common: '{top_label}' with {top_count:,} ({top_pct:.1f}%)",
                        f"Showing top {len(value_counts)} categories"
                    ]
                elif numeric_cols:
                    # Numeric only - show histogram-style bar chart
                    num_col = numeric_cols[0]
                    values = df[num_col].dropna()
                    counts, bin_edges = np.histogram(values, bins=10)
                    labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(counts))]
                    
                    chart_config['labels'] = labels
                    chart_config['datasets'][0]['data'] = [int(x) for x in counts.tolist()]
                    chart_config['title'] = f'Distribution of {num_col}'
                    
            elif chart_type == 'line':
                if numeric_cols:
                    num_col = target_col if target_col and target_col in numeric_cols else numeric_cols[0]
                    values = df[num_col].head(20).tolist()
                    chart_config['labels'] = [str(i+1) for i in range(len(values))]
                    chart_config['datasets'][0]['data'] = [round(float(x), 2) if pd.notna(x) else 0 for x in values]
                    chart_config['datasets'][0]['label'] = num_col
                    chart_config['title'] = f'Trend of {num_col}'
                    chart_config['x_axis_label'] = 'Index'
                    chart_config['y_axis_label'] = num_col
                    
                    mean_val = df[num_col].mean()
                    chart_config['insights'] = [
                        f"Average value: {mean_val:,.2f}",
                        f"Showing first {len(values)} data points"
                    ]
                    
            elif chart_type == 'histogram':
                if numeric_cols:
                    num_col = target_col if target_col and target_col in numeric_cols else numeric_cols[0]
                    values = df[num_col].dropna()
                    
                    counts, bin_edges = np.histogram(values, bins=10)
                    labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(counts))]
                    
                    chart_config['labels'] = labels
                    chart_config['datasets'][0]['data'] = [int(x) for x in counts.tolist()]
                    chart_config['datasets'][0]['label'] = 'Frequency'
                    chart_config['title'] = f'Distribution of {num_col}'
                    chart_config['x_axis_label'] = num_col
                    chart_config['y_axis_label'] = 'Frequency'
                    
                    chart_config['insights'] = [
                        f"Mean: {values.mean():,.2f}",
                        f"Std Dev: {values.std():,.2f}",
                        f"Range: {values.min():,.2f} to {values.max():,.2f}"
                    ]
                    
            elif chart_type == 'scatter':
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                    sample = df[[x_col, y_col]].dropna().head(50)
                    
                    chart_config['datasets'][0]['data'] = [
                        {'x': round(float(row[x_col]), 2), 'y': round(float(row[y_col]), 2)}
                        for _, row in sample.iterrows()
                    ]
                    chart_config['datasets'][0]['label'] = f'{x_col} vs {y_col}'
                    chart_config['title'] = f'Relationship: {x_col} vs {y_col}'
                    chart_config['x_axis_label'] = x_col
                    chart_config['y_axis_label'] = y_col
                    
                    corr = df[x_col].corr(df[y_col])
                    chart_config['insights'] = [
                        f"Correlation coefficient: {corr:.3f}",
                        f"Showing {len(sample)} data points"
                    ]
                    
        except Exception as e:
            _sys.stderr.write(f"⚠️ Fallback chart generation error: {e}\n")
            import traceback
            _sys.stderr.write(f"{traceback.format_exc()}\n")
            _sys.stderr.flush()
            chart_config['insights'] = ['Unable to generate detailed chart from data']
        
        return chart_config
    
    def _generate_conversational_response(self, df: pd.DataFrame, query: str, 
                                          groq_client) -> Dict[str, Any]:
        """
        Generate a conversational/textual response for analytical queries.
        
        Args:
            df: Input DataFrame
            query: User's query
            groq_client: Groq client instance
            
        Returns:
            Dict with success status and textual response
        """
        try:
            # Create comprehensive dataset context
            dataset_context = self._get_dataset_summary_context(df)
            
            # Create the prompt for conversational response
            prompt = f"""You are a helpful data analyst assistant. The user has uploaded a dataset and is asking a question about it.

DATASET INFORMATION:
{dataset_context}

USER QUESTION: "{query}"

INSTRUCTIONS:
1. Provide a clear, helpful, and informative response in plain English
2. Use the dataset information provided to give accurate insights
3. Format your response as a well-structured paragraph or bullet points as appropriate
4. If the question asks for a summary, provide key insights about the data
5. Include relevant statistics or observations from the dataset context
6. Be concise but comprehensive
7. Do NOT generate any code - provide only a textual response
8. If you mention specific numbers or statistics, reference the actual data

YOUR RESPONSE:"""

            # Get response from Groq
            response = groq_client.generate_code(prompt)
            
            # Clean up the response
            cleaned_response = response.strip()
            
            # Remove any code blocks if accidentally included
            if '```' in cleaned_response:
                # Extract text outside code blocks
                parts = re.split(r'```[\w]*\n?', cleaned_response)
                cleaned_response = ' '.join(part.strip() for part in parts if part.strip())
            
            return {
                'success': True,
                'result': cleaned_response,
                'response_type': 'conversational',
                'code': None,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'result': None,
                'response_type': 'conversational',
                'code': None,
                'error': str(e)
            }
    
    def _get_dataset_summary_context(self, df: pd.DataFrame) -> str:
        """
        Generate a comprehensive summary of the dataset for conversational queries.
        """
        try:
            lines = []
            
            # Basic info
            lines.append(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            lines.append(f"\nColumn Names: {list(df.columns)}")
            
            # Column types
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            lines.append(f"\nNumeric Columns ({len(numeric_cols)}): {numeric_cols}")
            lines.append(f"Text/Categorical Columns ({len(text_cols)}): {text_cols}")
            if date_cols:
                lines.append(f"Date Columns ({len(date_cols)}): {date_cols}")
            
            # Sample data
            lines.append(f"\nFirst 3 rows sample:\n{df.head(3).to_string()}")
            
            # Basic statistics for numeric columns
            if numeric_cols:
                lines.append(f"\nNumeric Column Statistics:")
                stats = df[numeric_cols].describe().round(2)
                lines.append(stats.to_string())
            
            # Value counts for categorical columns (first 2 only)
            if text_cols:
                lines.append(f"\nCategorical Column Value Counts (top 5 each):")
                for col in text_cols[:3]:  # Limit to first 3 categorical columns
                    try:
                        value_counts = df[col].value_counts().head(5)
                        lines.append(f"\n{col}:")
                        for val, count in value_counts.items():
                            lines.append(f"  - {val}: {count}")
                    except:
                        pass
            
            # Missing values summary
            missing = df.isnull().sum()
            missing_cols = missing[missing > 0]
            if len(missing_cols) > 0:
                lines.append(f"\nMissing Values:")
                for col, count in missing_cols.items():
                    pct = (count / len(df)) * 100
                    lines.append(f"  - {col}: {count} ({pct:.1f}%)")
            else:
                lines.append(f"\nNo missing values in the dataset.")
            
            return '\n'.join(lines)
            
        except Exception as e:
            return f"Error generating dataset summary: {str(e)}"

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
        rephrased_query = query
        
        try:
            # Step 1: Detect user intent (data query vs conversational vs visualization)
            intent_result = self._detect_query_intent(query)
            intent = intent_result['intent']
            intent_confidence = intent_result['confidence']
            
            # Log intent detection for debugging
            import sys as _sys
            _sys.stderr.write(f"\n🎯 Intent Detection:\n")
            _sys.stderr.write(f"   Query: {query[:50]}...\n")
            _sys.stderr.write(f"   Intent: {intent} (confidence: {intent_confidence:.2f})\n")
            _sys.stderr.write(f"   Scores: {intent_result['scores']}\n")
            _sys.stderr.flush()
            
            # Step 2: Handle based on intent
            if intent == 'conversational':
                # Generate textual/analytical response
                result = self._generate_conversational_response(df, query, groq_client)
                result['original_query'] = query
                result['rephrased_query'] = query  # No rephrasing for conversational
                result['intent'] = 'conversational'
                result['intent_confidence'] = intent_confidence
                return result
            
            if intent == 'visualization':
                # Generate visualization response
                result = self._generate_visualization(df, query, groq_client)
                result['original_query'] = query
                result['rephrased_query'] = query  # No rephrasing for visualization
                result['intent'] = 'visualization'
                result['intent_confidence'] = intent_confidence
                return result
            
            # Step 3: For data queries, proceed with rephrasing and code generation
            rephrased_query = self._rephrase_query(df, query, groq_client)
            
            # Generate pandas code using the rephrased query
            generated_code = self._generate_pandas_code(df, rephrased_query, groq_client)
            
            if not generated_code:
                return {
                    'success': False,
                    'error': 'Failed to generate pandas code',
                    'code': '',
                    'result': None,
                    'original_query': query,
                    'rephrased_query': rephrased_query,
                    'response_type': 'data',
                    'intent': 'data',
                    'intent_confidence': intent_confidence
                }
            
            # Validate and execute the code
            result = self._execute_safe_code(df, generated_code)
            
            return {
                'success': True,
                'result': result,
                'code': generated_code,
                'error': None,
                'original_query': query,
                'rephrased_query': rephrased_query,
                'response_type': 'data',
                'intent': 'data',
                'intent_confidence': intent_confidence
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'code': generated_code,
                'result': None,
                'original_query': query,
                'rephrased_query': rephrased_query,
                'response_type': 'data',
                'intent': 'data'
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
    
    def _rephrase_query(self, df: pd.DataFrame, query: str, groq_client) -> str:
        """
        Rephrase a vague query into a precise, structured query using DataFrame metadata.
        
        Examples:
        - "give me value for nic code 123 and which is orally taken" 
          -> "What is the value where nic_code equals 123 and consumption equals 'oral'?"
        - "show products in electronics that cost less than 500"
          -> "Show all products where category equals 'Electronics' and price is less than 500"
        
        Args:
            df: Input DataFrame for context
            query: Original user query
            groq_client: Groq client instance
            
        Returns:
            Rephrased query string
        """
        try:
            # Get column names and sample values for context
            column_info = self._get_column_metadata(df)
            
            # Create the rephrasing prompt
            rephrase_prompt = f"""You are a query clarification assistant. Your job is to rephrase vague or informal user queries into clear, precise queries that map to database columns.

DATASET METADATA:
{column_info}

USER'S ORIGINAL QUERY: "{query}"

INSTRUCTIONS:
1. Identify what the user is asking for (which columns/values they want)
2. Map informal terms to actual column names from the metadata above
3. Rephrase the query to be clear and precise, using actual column names
4. Keep the same intent but make it unambiguous
5. If the query is already clear, return it as-is with minor improvements

EXAMPLES:
- "give me value for nic code 123 and orally taken" → "Show records where NIC_Code equals 123 and Consumption_Type equals 'Oral'"
- "products under 500 in electronics" → "Show all products where Category equals 'Electronics' and Price is less than 500"
- "highest selling items last month" → "Show items with highest Sales for the last month, sorted by Sales descending"

Return ONLY the rephrased query, nothing else. Do not include explanations or code.

REPHRASED QUERY:"""

            # Get response from Groq
            response = groq_client.generate_code(rephrase_prompt)
            
            # Clean up the response
            rephrased = response.strip()
            
            # Remove quotes if wrapped
            if (rephrased.startswith('"') and rephrased.endswith('"')) or \
               (rephrased.startswith("'") and rephrased.endswith("'")):
                rephrased = rephrased[1:-1]
            
            # If rephrasing failed or returned empty, use original
            if not rephrased or len(rephrased) < 5:
                return query
            
            # Log the rephrasing for debugging
            import sys as _sys
            _sys.stderr.write(f"\n🔄 Query Rephrasing:\n")
            _sys.stderr.write(f"   Original: {query}\n")
            _sys.stderr.write(f"   Rephrased: {rephrased}\n")
            _sys.stderr.flush()
            
            return rephrased
            
        except Exception as e:
            # If rephrasing fails, just use the original query
            import sys as _sys
            _sys.stderr.write(f"⚠️ Query rephrasing failed: {e}, using original query\n")
            _sys.stderr.flush()
            return query
    
    def _get_column_metadata(self, df: pd.DataFrame) -> str:
        """
        Generate detailed column metadata for query rephrasing.
        Includes column names, types, and sample values.
        """
        metadata_lines = []
        metadata_lines.append("COLUMNS AND THEIR DETAILS:")
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            unique_count = df[col].nunique()
            
            # Get sample values (up to 5 unique values for categorical, or range for numeric)
            if df[col].dtype == 'object' or unique_count <= 10:
                sample_values = df[col].dropna().unique()[:5].tolist()
                sample_str = f"Sample values: {sample_values}"
            elif pd.api.types.is_numeric_dtype(df[col]):
                min_val = df[col].min()
                max_val = df[col].max()
                sample_str = f"Range: {min_val} to {max_val}"
            else:
                sample_str = f"Unique values: {unique_count}"
            
            # Add common alternative names/aliases
            col_lower = col.lower().replace('_', ' ').replace('-', ' ')
            metadata_lines.append(f"  - Column: '{col}' (type: {col_type})")
            metadata_lines.append(f"    {sample_str}")
            
            # Suggest possible user terms for this column
            possible_terms = self._generate_column_aliases(col)
            if possible_terms:
                metadata_lines.append(f"    User might refer to this as: {possible_terms}")
        
        return "\n".join(metadata_lines)
    
    def _generate_column_aliases(self, column_name: str) -> str:
        """Generate possible user terms for a column name"""
        aliases = []
        col_lower = column_name.lower()
        
        # Common mappings
        alias_mappings = {
            'nic': ['nic code', 'nic-code', 'niccode'],
            'code': ['code', 'id', 'number'],
            'consumption': ['consumption', 'taken', 'how taken', 'method'],
            'oral': ['oral', 'orally', 'by mouth'],
            'price': ['price', 'cost', 'amount', 'value'],
            'category': ['category', 'type', 'kind', 'section'],
            'name': ['name', 'title', 'label'],
            'date': ['date', 'time', 'when'],
            'quantity': ['quantity', 'qty', 'count', 'number', 'amount'],
            'status': ['status', 'state', 'condition'],
        }
        
        for key, terms in alias_mappings.items():
            if key in col_lower:
                aliases.extend([t for t in terms if t not in aliases])
        
        # Add variations of the column name itself
        variations = [
            col_lower,
            col_lower.replace('_', ' '),
            col_lower.replace('-', ' '),
            col_lower.replace('_', ''),
        ]
        for v in variations:
            if v not in aliases:
                aliases.append(v)
        
        return ', '.join(aliases[:5]) if aliases else ''

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
            # Debug: Log the actual code being executed
            import sys as _debug_sys
            _debug_sys.stderr.write(f"\n{'='*50}\n")
            _debug_sys.stderr.write(f"📝 Code to execute:\n{code}\n")
            _debug_sys.stderr.write(f"{'='*50}\n")
            _debug_sys.stderr.flush()
            
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
            
            # Check if it's a multi-line expression (chained methods) vs multi-statement
            # Multi-line expressions start with df. or df[ and are chained method calls
            code_stripped = code.strip()
            first_line = code_stripped.split('\n')[0].strip()
            
            # Check for assignment at the start (like "result = df..." or "x = df...")
            # But not for things like ".assign(col=..." which are method calls
            has_variable_assignment = (
                '=' in first_line and 
                not first_line.startswith('df') and
                not first_line.startswith('.')
            )
            
            # Check if code starts with df (any accessor like df., df[, df(, etc.)
            starts_with_df = code_stripped.startswith('df')
            
            is_chained_expression = (
                is_multiline and 
                starts_with_df and
                not has_variable_assignment and
                not any(line.strip().startswith(('result ', 'result=', 'output ', 'output=', 
                                                  'answer ', 'answer=', 'import ', 'print('))
                       for line in code_stripped.split('\n'))
            )
            
            # Debug logging to stderr (won't be captured by stdout redirect)
            import sys as _sys
            _sys.stderr.write(f"🔍 Code execution debug:\n")
            _sys.stderr.write(f"   is_multiline: {is_multiline}\n")
            _sys.stderr.write(f"   starts_with_df: {starts_with_df}\n")
            _sys.stderr.write(f"   is_chained_expression: {is_chained_expression}\n")
            _sys.stderr.write(f"   first_line: {first_line[:80]}...\n")
            _sys.stderr.write(f"   has_variable_assignment: {has_variable_assignment}\n")
            _sys.stderr.flush()
            
            # Capture stdout for operations that print
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                if is_chained_expression:
                    # Multi-line chained expression - wrap and use eval
                    # Join the lines and try to evaluate as single expression
                    single_line_code = ' '.join(line.strip() for line in code_stripped.split('\n'))
                    _sys.stderr.write(f"   Trying eval with joined code...\n")
                    _sys.stderr.flush()
                    try:
                        compile(single_line_code, '<string>', 'eval')
                        result = eval(single_line_code, safe_globals, {})
                        _sys.stderr.write(f"   ✅ Eval succeeded, result type: {type(result)}\n")
                        _sys.stderr.flush()
                        return result
                    except SyntaxError as se:
                        _sys.stderr.write(f"   Eval failed with SyntaxError: {se}, trying wrapped exec...\n")
                        _sys.stderr.flush()
                        # Fall back to exec with result capture
                        wrapped_code = f"__result__ = (\n{code_stripped}\n)"
                        local_namespace = {}
                        exec(wrapped_code, safe_globals, local_namespace)
                        if '__result__' in local_namespace:
                            return local_namespace['__result__']
                        # If still no result, the code might be invalid
                        raise Exception("Failed to capture result from chained expression")
                        
                elif is_multiline:
                    _sys.stderr.write(f"   ⚠️ Going into is_multiline branch (not chained)\n")
                    _sys.stderr.flush()
                    # Multi-line code - use exec
                    try:
                        compile(code, '<string>', 'exec')
                    except SyntaxError as e:
                        raise Exception(f"Syntax error in generated code: {str(e)}")
                    
                    # Create a local namespace to capture results
                    local_namespace = {}
                    exec(code, safe_globals, local_namespace)
                    
                    _sys.stderr.write(f"   local_namespace keys: {list(local_namespace.keys())}\n")
                    _sys.stderr.flush()
                    
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
