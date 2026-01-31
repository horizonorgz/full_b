"""
Smart Entity Classifier
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any
from fuzzywuzzy import fuzz


class SmartEntityExtractor:
    """Extract and classify entities from natural language queries"""
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        
        # Strong query terms that are almost never categories
        self.strong_query_terms = {
            'show', 'find', 'get', 'list', 'display', 'identify', 'all', 
            'any', 'each', 'the', 'that', 'are', 'have', 'with', 'their', 
            'name', 'description', 'priced', 'above', 'below', 'under', 
            'over', 'than', 'products', 'items', 'top', 'most', 'expensive', 
            'cheap', 'best', 'highest', 'lowest'
        }
        
        # Business category indicators
        self.category_indicators = {
            'electronics', 'clothing', 'apparel', 'beauty', 'personal', 
            'care', 'sports', 'outdoors', 'accessories', 'equipment', 
            'books', 'toys', 'games', 'shoes', 'footwear', 'bags', 
            'jewelry', 'watches', 'furniture', 'home', 'garden', 'food', 
            'drinks', 'health', 'automotive', 'tools', 'office', 'supplies'
        }
    
    def extract_and_classify(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Main method to extract and classify entities from query"""
        
        # Extract vocabulary from dataset
        vocab_info = self._extract_vocabulary(df)
        
        # Extract potential entities from query
        entities = self._extract_entities(query)
        
        # Classify each entity
        classifications = {}
        category_entities = []
        
        for entity in entities:
            classification = self._classify_entity(entity, query, vocab_info)
            classifications[entity] = classification
            
            if classification['classification'] == 'category':
                category_entities.append({
                    'entity': entity,
                    'confidence': classification['confidence'],
                    'reasons': classification['reasons']
                })
        
        return {
            'category_entities': category_entities,
            'all_classifications': classifications,
            'vocabulary_stats': vocab_info['stats']
        }
    
    def _extract_vocabulary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract vocabulary information from the dataset"""
        
        vocab = {
            'category_values': set(),
            'column_names': set(df.columns.str.lower()),
            'all_text_values': set(),
            'stats': {}
        }
        
        # Find category columns
        category_columns = [col for col in df.columns 
                          if any(indicator in col.lower() 
                                for indicator in ['category', 'type', 'class', 'group', 'section'])]
        
        if not category_columns and len(df.columns) > 0:
            # Use string columns
            string_columns = df.select_dtypes(include=['object']).columns
            category_columns = string_columns[:2] 
        
        for col in category_columns:
            if col in df.columns:
                unique_values = df[col].dropna().unique()
                for val in unique_values:
                    if isinstance(val, str):
                        vocab['category_values'].add(val.lower())
        
        # Calculate stats
        if vocab['category_values']:
            word_counts = [len(cat.split()) for cat in vocab['category_values']]
            vocab['stats'] = {
                'total_categories': len(vocab['category_values']),
                'avg_word_count': np.mean(word_counts),
                'max_word_count': max(word_counts),
                'min_word_count': min(word_counts)
            }
        else:
            vocab['stats'] = {'total_categories': 0}
        
        return vocab
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from the query - more selective approach"""
        
        entities = set()
        
        # Method 1: Extract quoted phrases (high confidence)
        quoted_phrases = re.findall(r"'([^']*)'", query)
        entities.update(quoted_phrases)
        
        # Method 2: Extract phrases explicitly marked as categories
        category_context_patterns = [
            r'(?:in|from|of)\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s&\'-]*)\s+(?:category|section|department)',
            r'(?:category|section|type|kind)\s+(?:of\s+)?([a-zA-Z][a-zA-Z\s&\'-]*)',
        ]
        
        for pattern in category_context_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.update([match.strip() for match in matches if len(match.strip()) > 2])
        
        # Method 3: Extract capitalized business-like phrases (more selective)
        capitalized_phrases = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+&\s+[A-Z][a-zA-Z]*)*\b', query)
        for phrase in capitalized_phrases:
            # Only include if it looks like a business category
            if (len(phrase.split()) <= 3 and 
                any(indicator in phrase.lower() for indicator in self.category_indicators)):
                entities.add(phrase.strip())
        
        # Method 4: Only extract individual words if they are strong category indicators
        words = query.split()
        for word in words:
            if (word.lower() in self.category_indicators and 
                len(word) > 3 and 
                word.isalpha()):
                entities.add(word)
        
        # Filter out obvious non-categories
        filtered_entities = []
        for entity in entities:
            if self._is_likely_category_entity(entity, query):
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def _is_likely_category_entity(self, entity: str, query: str) -> bool:
        """Pre-filter entities to avoid processing obvious non-categories"""
        
        entity_lower = entity.lower().strip()
        
        # Skip if it's a strong query term
        if entity_lower in self.strong_query_terms:
            return False
        
        # Skip medical/domain-specific terms that are clearly not categories
        medical_terms = {
            'patients', 'diagnosis', 'hospital', 'admitted', 'emergencies',
            'satisfaction', 'score', 'checkups', 'routine', 'treatment',
            'doctor', 'nurse', 'medicine', 'therapy', 'surgery'
        }
        if entity_lower in medical_terms:
            return False
        
        # Skip if it contains descriptive patterns
        if any(pattern in entity_lower for pattern in [
            'who were', 'that are', 'which are', 'more than', 'less than',
            'compared to', 'average', 'most common'
        ]):
            return False
        
        # Must have some indication it could be a category
        has_category_context = any(word in query.lower() for word in [
            'category', 'section', 'type', 'kind', 'department', 'from', 'in'
        ])
        
        has_business_indicators = any(indicator in entity_lower 
                                    for indicator in self.category_indicators)
        
        # Only proceed if there's some category context or business indicators
        return has_category_context or has_business_indicators
    
    def _classify_entity(self, entity: str, query: str, 
                        vocab_info: Dict) -> Dict[str, Any]:
        """Classify a single entity"""
        
        entity_lower = entity.lower().strip()
        confidence = 0.0
        reasons = []
        
        # Quick rejection: obvious query terms
        if entity_lower in self.strong_query_terms:
            return {
                'classification': 'query_term',
                'confidence': 0.9,
                'reasons': ['strong_query_term']
            }
        
        # Strong positive: exact match with dataset categories
        if entity_lower in vocab_info['category_values']:
            return {
                'classification': 'category',
                'confidence': 0.95,
                'reasons': ['exact_dataset_match']
            }
        
        # Fuzzy match with dataset categories
        best_fuzzy_score = 0
        for cat_val in vocab_info['category_values']:
            score = fuzz.ratio(entity_lower, cat_val) / 100.0
            best_fuzzy_score = max(best_fuzzy_score, score)
        
        if best_fuzzy_score > 0.8:
            confidence += 0.7
            reasons.append('high_fuzzy_match')
        elif best_fuzzy_score > 0.6:
            confidence += 0.5
            reasons.append('moderate_fuzzy_match')
        
        # Check for category indicators
        if any(indicator in entity_lower for indicator in self.category_indicators):
            confidence += 0.4
            reasons.append('contains_category_indicator')
        
        # Format analysis
        entity_words = entity_lower.split()
        if (entity.istitle() and len(entity_words) <= 3 and 
            not any(word in self.strong_query_terms for word in entity_words)):
            confidence += 0.3
            reasons.append('category_format')
        
        # Category symbols
        if ('&' in entity or "'" in entity or '-' in entity) and len(entity_words) <= 4:
            confidence += 0.2
            reasons.append('category_symbols')
        
        # Context analysis
        context_score = self._analyze_context(entity, query)
        confidence += context_score['score']
        reasons.extend(context_score['reasons'])
        
        # Negative signals
        if self._is_descriptive_phrase(entity):
            confidence -= 0.4
            reasons.append('descriptive_phrase_penalty')
        
        if len(entity_words) > 4:
            confidence -= 0.2
            reasons.append('too_long_penalty')
        
        # Final classification
        confidence = max(0.0, min(1.0, confidence))
        
        if confidence >= self.confidence_threshold:
            classification = 'category'
        elif confidence < 0.3:
            classification = 'query_term'
        else:
            classification = 'ambiguous'
        
        return {
            'classification': classification,
            'confidence': confidence,
            'reasons': reasons
        }
    
    def _analyze_context(self, entity: str, query: str) -> Dict[str, Any]:
        """Analyze the context around the entity in the query"""
        
        result = {'score': 0.0, 'reasons': []}
        
        query_lower = query.lower()
        entity_lower = entity.lower()
        
        # Find entity position
        entity_pos = query_lower.find(entity_lower)
        if entity_pos == -1:
            return result
        
        # Check preceding context
        preceding = query_lower[:entity_pos].strip()
        preceding_words = preceding.split()[-3:] if preceding else []
        
        category_contexts = ['in', 'from', 'of', 'category', 'type', 'section', 'the']
        if any(ctx in preceding_words for ctx in category_contexts):
            result['score'] += 0.3
            result['reasons'].append('category_context')
        
        # Check descriptive following text
        following = query_lower[entity_pos + len(entity_lower):].strip()
        if following.startswith(('name', 'description', 'that are', 'which are')):
            result['score'] -= 0.3
            result['reasons'].append('descriptive_context')
        
        return result
    
    def _is_descriptive_phrase(self, entity: str) -> bool:
        """Check if entity is a descriptive phrase"""
        
        descriptive_patterns = [
            'their name', 'their description', 'brand and', 'name or description',
            'are priced', 'units in stock', 'availability status', 'that are',
            'which are', 'priced above', 'priced below', 'less than', 'more than'
        ]
        
        entity_lower = entity.lower()
        return any(pattern in entity_lower for pattern in descriptive_patterns)
