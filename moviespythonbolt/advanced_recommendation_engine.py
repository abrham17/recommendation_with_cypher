#!/usr/bin/env python3
"""
Advanced Hybrid Recommendation Engine for Neo4j Movies
Implements sophisticated ML algorithms including matrix factorization,
ensemble methods, and context-aware recommendations.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import json
import logging
from typing import List, Dict, Tuple, Optional
from neo4j import GraphDatabase
from collections import defaultdict
import math

class AdvancedHybridRecommendationEngine:
    """
    Sophisticated hybrid recommendation engine combining multiple ML approaches
    """
    
    def __init__(self, driver, database):
        self.driver = driver
        self.database = database
        self.user_item_matrix = None
        self.item_features_matrix = None
        self.nmf_model = None
        self.tfidf_vectorizer = None
        self.content_similarity_matrix = None
        self.ensemble_weights = {
            'collaborative': 0.35,
            'content': 0.25,
            'matrix_factorization': 0.25,
            'context_aware': 0.15
        }
        self.logger = logging.getLogger(__name__)
        
    def initialize_models(self):
        """Initialize and train all ML models"""
        try:
            self._build_user_item_matrix()
            self._build_content_features()
            self._train_matrix_factorization()
            self._compute_content_similarity()
            self.logger.info("All recommendation models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
    
    def _build_user_item_matrix(self):
        """Build user-item rating matrix for collaborative filtering"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (u:User)-[r:RATED]->(m:Movie)
                RETURN u.userId as userId, m.title as movieTitle, r.rating as rating
            """)
            
            ratings_data = []
            for record in result:
                ratings_data.append({
                    'userId': record['userId'],
                    'movieTitle': record['movieTitle'],
                    'rating': record['rating']
                })
            
            if not ratings_data:
                self.logger.warning("No rating data found")
                return
            
            # Create pivot table
            df = pd.DataFrame(ratings_data)
            self.user_item_matrix = df.pivot_table(
                index='userId', 
                columns='movieTitle', 
                values='rating', 
                fill_value=0
            )
            
            self.logger.info(f"Built user-item matrix: {self.user_item_matrix.shape}")
    
    def _build_content_features(self):
        """Build content feature matrix using movie metadata"""
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (m:Movie)
                OPTIONAL MATCH (m)-[:IN_GENRE]->(g:Genre)
                OPTIONAL MATCH (m)<-[:DIRECTED]-(d:Director)
                OPTIONAL MATCH (m)<-[:ACTED_IN]-(a:Actor)
                WITH m, 
                     collect(DISTINCT g.name) as genres,
                     collect(DISTINCT d.name) as directors,
                     collect(DISTINCT a.name) as actors
                RETURN m.title as title,
                       m.summary as summary,
                       m.tagline as tagline,
                       genres,
                       directors,
                       actors,
                       m.released as year
            """)
            
            movies_data = []
            for record in result:
                # Combine text features
                text_features = []
                if record['summary']:
                    text_features.append(record['summary'])
                if record['tagline']:
                    text_features.append(record['tagline'])
                
                # Add categorical features as text
                text_features.extend(record['genres'] or [])
                text_features.extend(record['directors'] or [])
                text_features.extend((record['actors'] or [])[:5])  # Top 5 actors
                
                movies_data.append({
                    'title': record['title'],
                    'features': ' '.join(text_features),
                    'year': record['year'] or 2000
                })
            
            if not movies_data:
                self.logger.warning("No movie content data found")
                return
            
            # Create TF-IDF features
            df = pd.DataFrame(movies_data)
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self.item_features_matrix = self.tfidf_vectorizer.fit_transform(df['features'])
            self.movie_titles = df['title'].tolist()
            
            self.logger.info(f"Built content features matrix: {self.item_features_matrix.shape}")
    
    def _train_matrix_factorization(self):
        """Train Non-negative Matrix Factorization model"""
        if self.user_item_matrix is None or self.user_item_matrix.empty:
            self.logger.warning("No user-item matrix available for matrix factorization")
            return
        
        # Use NMF for matrix factorization
        n_components = min(50, min(self.user_item_matrix.shape) - 1)
        self.nmf_model = NMF(
            n_components=n_components,
            init='random',
            random_state=42,
            max_iter=200
        )
        
        # Fit the model
        self.nmf_model.fit(self.user_item_matrix.values)
        self.logger.info(f"Trained NMF model with {n_components} components")
    
    def _compute_content_similarity(self):
        """Compute content-based similarity matrix"""
        if self.item_features_matrix is None:
            self.logger.warning("No content features available")
            return
        
        self.content_similarity_matrix = cosine_similarity(self.item_features_matrix)
        self.logger.info("Computed content similarity matrix")
    
    def collaborative_filtering_advanced(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Advanced collaborative filtering using user similarity and matrix factorization"""
        if self.user_item_matrix is None or user_id not in self.user_item_matrix.index:
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index.tolist()
        
        if not unrated_movies:
            return []
        
        # Method 1: User-based collaborative filtering
        user_similarities = {}
        for other_user in self.user_item_matrix.index:
            if other_user != user_id:
                similarity = self._calculate_user_similarity(user_id, other_user)
                if similarity > 0.1:  # Threshold for similarity
                    user_similarities[other_user] = similarity
        
        # Method 2: Matrix factorization predictions
        mf_predictions = {}
        if self.nmf_model is not None:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            user_factors = self.nmf_model.transform(self.user_item_matrix.iloc[[user_idx]])
            item_factors = self.nmf_model.components_
            
            for movie in unrated_movies:
                if movie in self.user_item_matrix.columns:
                    movie_idx = self.user_item_matrix.columns.get_loc(movie)
                    predicted_rating = np.dot(user_factors[0], item_factors[:, movie_idx])
                    mf_predictions[movie] = predicted_rating
        
        # Combine predictions
        recommendations = []
        for movie in unrated_movies:
            # User-based CF prediction
            cf_score = self._predict_rating_collaborative(user_id, movie, user_similarities)
            
            # Matrix factorization prediction
            mf_score = mf_predictions.get(movie, 3.0)
            
            # Combine scores
            combined_score = (cf_score * 0.6 + mf_score * 0.4)
            
            recommendations.append({
                'title': movie,
                'predicted_rating': combined_score,
                'method': 'advanced_collaborative'
            })
        
        # Sort and return top recommendations
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:n_recommendations]
    
    def content_based_advanced(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Advanced content-based filtering using TF-IDF and user preferences"""
        if self.content_similarity_matrix is None:
            return []
        
        # Get user's highly rated movies
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (u:User {userId: $userId})-[r:RATED]->(m:Movie)
                WHERE r.rating >= 4.0
                RETURN m.title as title, r.rating as rating
                ORDER BY r.rating DESC, r.timestamp DESC
                LIMIT 10
            """, userId=user_id)
            
            liked_movies = [record['title'] for record in result]
        
        if not liked_movies:
            return []
        
        # Calculate content-based scores
        movie_scores = defaultdict(float)
        
        for liked_movie in liked_movies:
            if liked_movie in self.movie_titles:
                movie_idx = self.movie_titles.index(liked_movie)
                similarities = self.content_similarity_matrix[movie_idx]
                
                for i, similarity in enumerate(similarities):
                    candidate_movie = self.movie_titles[i]
                    if candidate_movie != liked_movie and similarity > 0.1:
                        movie_scores[candidate_movie] += similarity
        
        # Get movies user hasn't rated
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (m:Movie)
                WHERE NOT EXISTS((m)<-[:RATED]-(:User {userId: $userId}))
                RETURN m.title as title
            """, userId=user_id)
            
            unrated_movies = {record['title'] for record in result}
        
        # Filter and sort recommendations
        recommendations = []
        for movie, score in movie_scores.items():
            if movie in unrated_movies:
                recommendations.append({
                    'title': movie,
                    'content_score': score,
                    'method': 'advanced_content'
                })
        
        recommendations.sort(key=lambda x: x['content_score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def context_aware_recommendations(self, user_id: str, context: Dict, n_recommendations: int = 10) -> List[Dict]:
        """Context-aware recommendations considering time, mood, etc."""
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()  # 0=Monday, 6=Sunday
        
        # Define context-based genre preferences
        time_genre_mapping = {
            'morning': ['Family', 'Comedy', 'Animation'],
            'afternoon': ['Action', 'Adventure', 'Comedy'],
            'evening': ['Drama', 'Romance', 'Thriller'],
            'night': ['Horror', 'Thriller', 'Sci-Fi']
        }
        
        # Determine time of day
        if 6 <= current_hour < 12:
            time_period = 'morning'
        elif 12 <= current_hour < 17:
            time_period = 'afternoon'
        elif 17 <= current_hour < 22:
            time_period = 'evening'
        else:
            time_period = 'night'
        
        preferred_genres = time_genre_mapping.get(time_period, [])
        
        # Weekend vs weekday preferences
        is_weekend = current_day >= 5
        if is_weekend:
            preferred_genres.extend(['Action', 'Adventure', 'Comedy'])
        
        # Get context-appropriate movies
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (m:Movie)-[:IN_GENRE]->(g:Genre)
                WHERE g.name IN $genres
                AND NOT EXISTS((m)<-[:RATED]-(:User {userId: $userId}))
                WITH m, collect(g.name) as movieGenres, m.avgRating as avgRating
                WHERE avgRating >= 3.5
                RETURN m.title as title, movieGenres, avgRating, m.popularity as popularity
                ORDER BY avgRating DESC, popularity DESC
                LIMIT $limit
            """, userId=user_id, genres=preferred_genres, limit=n_recommendations * 2)
            
            recommendations = []
            for record in result:
                # Calculate context score based on genre match and time appropriateness
                genre_match_score = len(set(record['movieGenres']) & set(preferred_genres))
                context_score = (
                    record['avgRating'] * 0.4 +
                    genre_match_score * 0.3 +
                    (record['popularity'] / 100) * 0.3
                )
                
                recommendations.append({
                    'title': record['title'],
                    'context_score': context_score,
                    'genres': record['movieGenres'],
                    'time_period': time_period,
                    'method': 'context_aware'
                })
        
        recommendations.sort(key=lambda x: x['context_score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def ensemble_recommendations(self, user_id: str, context: Dict = None, n_recommendations: int = 15) -> List[Dict]:
        """
        Ensemble method combining all recommendation approaches with dynamic weighting
        """
        if context is None:
            context = {}
        
        # Get recommendations from all methods
        collab_recs = self.collaborative_filtering_advanced(user_id, n_recommendations)
        content_recs = self.content_based_advanced(user_id, n_recommendations)
        context_recs = self.context_aware_recommendations(user_id, context, n_recommendations)
        
        # Get user's rating history to adjust weights
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (u:User {userId: $userId})-[r:RATED]->()
                RETURN count(r) as ratingCount, avg(r.rating) as avgRating
            """, userId=user_id)
            
            user_stats = result.single()
            rating_count = user_stats['ratingCount'] if user_stats else 0
            avg_rating = user_stats['avgRating'] if user_stats else 3.0
        
        # Adjust ensemble weights based on user data availability
        adjusted_weights = self.ensemble_weights.copy()
        
        if rating_count < 5:
            # New user - rely more on content and context
            adjusted_weights['collaborative'] = 0.15
            adjusted_weights['content'] = 0.35
            adjusted_weights['context_aware'] = 0.35
            adjusted_weights['matrix_factorization'] = 0.15
        elif rating_count > 50:
            # Experienced user - rely more on collaborative
            adjusted_weights['collaborative'] = 0.45
            adjusted_weights['matrix_factorization'] = 0.35
            adjusted_weights['content'] = 0.15
            adjusted_weights['context_aware'] = 0.05
        
        # Combine all recommendations
        movie_scores = defaultdict(lambda: {'total_score': 0, 'methods': [], 'details': {}})
        
        # Process collaborative recommendations
        for rec in collab_recs:
            movie = rec['title']
            score = rec['predicted_rating'] * adjusted_weights['collaborative']
            movie_scores[movie]['total_score'] += score
            movie_scores[movie]['methods'].append('collaborative')
            movie_scores[movie]['details']['collaborative_score'] = rec['predicted_rating']
        
        # Process content-based recommendations
        for rec in content_recs:
            movie = rec['title']
            # Normalize content score to 1-5 scale
            normalized_score = min(5.0, max(1.0, rec['content_score'] * 5))
            score = normalized_score * adjusted_weights['content']
            movie_scores[movie]['total_score'] += score
            movie_scores[movie]['methods'].append('content')
            movie_scores[movie]['details']['content_score'] = rec['content_score']
        
        # Process context-aware recommendations
        for rec in context_recs:
            movie = rec['title']
            score = rec['context_score'] * adjusted_weights['context_aware']
            movie_scores[movie]['total_score'] += score
            movie_scores[movie]['methods'].append('context')
            movie_scores[movie]['details']['context_score'] = rec['context_score']
        
        # Create final recommendations
        final_recommendations = []
        for movie, data in movie_scores.items():
            # Boost score for movies recommended by multiple methods
            method_diversity_bonus = len(set(data['methods'])) * 0.1
            final_score = data['total_score'] + method_diversity_bonus
            
            final_recommendations.append({
                'title': movie,
                'ensemble_score': final_score,
                'methods_used': list(set(data['methods'])),
                'method_count': len(set(data['methods'])),
                'details': data['details'],
                'recommendation_type': 'ensemble_hybrid'
            })
        
        # Sort by ensemble score
        final_recommendations.sort(key=lambda x: x['ensemble_score'], reverse=True)
        
        # Add diversity to prevent over-specialization
        diversified_recs = self._add_diversity(final_recommendations, n_recommendations)
        
        return diversified_recs
    
    def _add_diversity(self, recommendations: List[Dict], n_recommendations: int) -> List[Dict]:
        """Add diversity to recommendations to avoid over-specialization"""
        if not recommendations:
            return []
        
        # Get genre information for diversity
        movie_genres = {}
        with self.driver.session(database=self.database) as session:
            for rec in recommendations:
                result = session.run("""
                    MATCH (m:Movie {title: $title})-[:IN_GENRE]->(g:Genre)
                    RETURN collect(g.name) as genres
                """, title=rec['title'])
                
                record = result.single()
                movie_genres[rec['title']] = record['genres'] if record else []
        
        # Select diverse recommendations
        selected = []
        used_genres = set()
        
        # First, select top recommendations ensuring genre diversity
        for rec in recommendations:
            if len(selected) >= n_recommendations:
                break
            
            movie_genre_set = set(movie_genres.get(rec['title'], []))
            
            # If we haven't seen these genres much, or it's a very high-scoring recommendation
            if (len(movie_genre_set & used_genres) <= 1 or 
                rec['ensemble_score'] > 4.0 or 
                len(selected) < n_recommendations // 2):
                
                selected.append(rec)
                used_genres.update(movie_genre_set)
        
        # Fill remaining slots with highest-scoring recommendations
        for rec in recommendations:
            if len(selected) >= n_recommendations:
                break
            if rec not in selected:
                selected.append(rec)
        
        return selected[:n_recommendations]
    
    def _calculate_user_similarity(self, user1: str, user2: str) -> float:
        """Calculate similarity between two users using Pearson correlation"""
        if self.user_item_matrix is None:
            return 0.0
        
        user1_ratings = self.user_item_matrix.loc[user1]
        user2_ratings = self.user_item_matrix.loc[user2]
        
        # Find commonly rated movies
        common_movies = (user1_ratings > 0) & (user2_ratings > 0)
        
        if common_movies.sum() < 2:
            return 0.0
        
        # Calculate Pearson correlation
        user1_common = user1_ratings[common_movies]
        user2_common = user2_ratings[common_movies]
        
        correlation = np.corrcoef(user1_common, user2_common)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _predict_rating_collaborative(self, user_id: str, movie: str, user_similarities: Dict) -> float:
        """Predict rating using collaborative filtering"""
        if not user_similarities:
            return 3.0  # Default rating
        
        weighted_sum = 0.0
        similarity_sum = 0.0
        
        for similar_user, similarity in user_similarities.items():
            if movie in self.user_item_matrix.columns and similar_user in self.user_item_matrix.index:
                rating = self.user_item_matrix.loc[similar_user, movie]
                if rating > 0:
                    weighted_sum += similarity * rating
                    similarity_sum += abs(similarity)
        
        if similarity_sum == 0:
            return 3.0
        
        predicted_rating = weighted_sum / similarity_sum
        return max(1.0, min(5.0, predicted_rating))  # Clamp to valid range
    
    def get_explanation(self, user_id: str, movie_title: str, recommendation_data: Dict) -> str:
        """Generate human-readable explanation for why a movie was recommended"""
        explanations = []
        
        methods = recommendation_data.get('methods_used', [])
        details = recommendation_data.get('details', {})
        
        if 'collaborative' in methods:
            explanations.append("Users with similar taste also enjoyed this movie")
        
        if 'content' in methods:
            explanations.append("This movie has similar themes to movies you've rated highly")
        
        if 'context' in methods:
            explanations.append("This movie fits your current viewing context and time preferences")
        
        if len(methods) > 1:
            explanations.append(f"Recommended by {len(methods)} different algorithms for higher confidence")
        
        return ". ".join(explanations) + "."
