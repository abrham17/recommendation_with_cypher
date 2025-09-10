#!/usr/bin/env python3
"""
Real-time User Interaction Tracking System
Tracks user behavior, analyzes patterns, and provides real-time feedback
for continuous recommendation system improvement.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import threading
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

class InteractionType(Enum):
    VIEW = "view"
    CLICK = "click"
    RATING = "rating"
    SEARCH = "search"
    RECOMMENDATION_VIEW = "recommendation_view"
    RECOMMENDATION_CLICK = "recommendation_click"
    MOVIE_DETAIL_VIEW = "movie_detail_view"
    PROFILE_VIEW = "profile_view"
    LOGIN = "login"
    LOGOUT = "logout"
    SCROLL = "scroll"
    HOVER = "hover"
    FILTER = "filter"
    SORT = "sort"

@dataclass
class UserInteraction:
    """Represents a single user interaction"""
    interaction_id: str
    user_id: str
    session_id: str
    interaction_type: InteractionType
    timestamp: datetime
    movie_title: Optional[str] = None
    rating_value: Optional[float] = None
    search_query: Optional[str] = None
    page_url: str = ""
    user_agent: str = ""
    ip_address: str = ""
    duration: Optional[int] = None  # in seconds
    metadata: Dict[str, Any] = None
    
    def to_dict(self):
        data = asdict(self)
        data['interaction_type'] = self.interaction_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

class RealTimeTracker:
    """
    Real-time user interaction tracking and analytics system
    """
    
    def __init__(self, driver, database):
        self.driver = driver
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # In-memory storage for real-time analytics
        self.recent_interactions = deque(maxlen=10000)  # Last 10k interactions
        self.user_sessions = defaultdict(dict)  # Active user sessions
        self.interaction_counts = defaultdict(int)  # Real-time counters
        self.movie_popularity = defaultdict(int)  # Real-time movie popularity
        
        # Analytics cache
        self.analytics_cache = {}
        self.cache_expiry = {}
        
        # Background thread for processing interactions
        self.processing_queue = deque()
        self.processing_thread = None
        self.stop_processing = False
        
        self.start_background_processing()
    
    def start_background_processing(self):
        """Start background thread for processing interactions"""
        self.processing_thread = threading.Thread(target=self._process_interactions_background)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.logger.info("Started background interaction processing")
    
    def track_interaction(self, user_id: str, interaction_type: InteractionType, 
                         session_id: str = None, **kwargs) -> str:
        """
        Track a user interaction in real-time
        """
        interaction_id = str(uuid.uuid4())
        
        if not session_id:
            session_id = self._get_or_create_session(user_id)
        
        interaction = UserInteraction(
            interaction_id=interaction_id,
            user_id=user_id,
            session_id=session_id,
            interaction_type=interaction_type,
            timestamp=datetime.now(),
            **kwargs
        )
        
        # Add to processing queue
        self.processing_queue.append(interaction)
        
        # Update real-time counters
        self._update_real_time_stats(interaction)
        
        # Add to recent interactions for analytics
        self.recent_interactions.append(interaction)
        
        self.logger.debug(f"Tracked interaction: {interaction_type.value} for user {user_id}")
        return interaction_id
    
    def _get_or_create_session(self, user_id: str) -> str:
        """Get existing session or create new one for user"""
        current_time = datetime.now()
        
        # Check if user has active session (within last 30 minutes)
        if user_id in self.user_sessions:
            session_data = self.user_sessions[user_id]
            last_activity = session_data.get('last_activity')
            
            if last_activity and (current_time - last_activity).seconds < 1800:  # 30 minutes
                session_data['last_activity'] = current_time
                return session_data['session_id']
        
        # Create new session
        session_id = str(uuid.uuid4())
        self.user_sessions[user_id] = {
            'session_id': session_id,
            'start_time': current_time,
            'last_activity': current_time,
            'interaction_count': 0
        }
        
        return session_id
    
    def _update_real_time_stats(self, interaction: UserInteraction):
        """Update real-time statistics"""
        # Update interaction counts
        self.interaction_counts[interaction.interaction_type.value] += 1
        self.interaction_counts['total'] += 1
        
        # Update movie popularity
        if interaction.movie_title:
            self.movie_popularity[interaction.movie_title] += 1
        
        # Update session stats
        if interaction.user_id in self.user_sessions:
            self.user_sessions[interaction.user_id]['interaction_count'] += 1
    
    def _process_interactions_background(self):
        """Background thread to process interactions and store in database"""
        while not self.stop_processing:
            try:
                if self.processing_queue:
                    # Process batch of interactions
                    batch = []
                    for _ in range(min(50, len(self.processing_queue))):  # Process up to 50 at once
                        if self.processing_queue:
                            batch.append(self.processing_queue.popleft())
                    
                    if batch:
                        self._store_interactions_batch(batch)
                
                time.sleep(1)  # Process every second
                
            except Exception as e:
                self.logger.error(f"Error in background processing: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _store_interactions_batch(self, interactions: List[UserInteraction]):
        """Store batch of interactions in Neo4j database"""
        try:
            with self.driver.session(database=self.database) as session:
                for interaction in interactions:
                    # Store interaction in database
                    session.run("""
                        MATCH (u:User {userId: $userId})
                        CREATE (i:Interaction {
                            interactionId: $interactionId,
                            sessionId: $sessionId,
                            type: $type,
                            timestamp: $timestamp,
                            movieTitle: $movieTitle,
                            ratingValue: $ratingValue,
                            searchQuery: $searchQuery,
                            pageUrl: $pageUrl,
                            userAgent: $userAgent,
                            duration: $duration,
                            metadata: $metadata
                        })
                        CREATE (u)-[:PERFORMED]->(i)
                    """, 
                    userId=interaction.user_id,
                    interactionId=interaction.interaction_id,
                    sessionId=interaction.session_id,
                    type=interaction.interaction_type.value,
                    timestamp=interaction.timestamp,
                    movieTitle=interaction.movie_title,
                    ratingValue=interaction.rating_value,
                    searchQuery=interaction.search_query,
                    pageUrl=interaction.page_url,
                    userAgent=interaction.user_agent,
                    duration=interaction.duration,
                    metadata=json.dumps(interaction.metadata) if interaction.metadata else None
                    )
                    
                    # If it's a movie interaction, create relationship
                    if interaction.movie_title:
                        session.run("""
                            MATCH (i:Interaction {interactionId: $interactionId})
                            MATCH (m:Movie {title: $movieTitle})
                            CREATE (i)-[:INVOLVES]->(m)
                        """, 
                        interactionId=interaction.interaction_id,
                        movieTitle=interaction.movie_title
                        )
            
            self.logger.debug(f"Stored {len(interactions)} interactions in database")
            
        except Exception as e:
            self.logger.error(f"Error storing interactions: {e}")
    
    def get_real_time_analytics(self) -> Dict[str, Any]:
        """Get real-time analytics dashboard data"""
        current_time = datetime.now()
        
        # Check cache
        cache_key = "real_time_analytics"
        if (cache_key in self.analytics_cache and 
            cache_key in self.cache_expiry and 
            current_time < self.cache_expiry[cache_key]):
            return self.analytics_cache[cache_key]
        
        # Calculate analytics
        analytics = {
            'current_timestamp': current_time.isoformat(),
            'total_interactions': self.interaction_counts['total'],
            'interaction_breakdown': dict(self.interaction_counts),
            'active_sessions': len([s for s in self.user_sessions.values() 
                                  if (current_time - s['last_activity']).seconds < 1800]),
            'top_movies_now': self._get_trending_movies_real_time(),
            'recent_activity': self._get_recent_activity_summary(),
            'user_engagement': self._calculate_user_engagement(),
        }
        
        # Cache for 30 seconds
        self.analytics_cache[cache_key] = analytics
        self.cache_expiry[cache_key] = current_time + timedelta(seconds=30)
        
        return analytics
    
    def _get_trending_movies_real_time(self) -> List[Dict[str, Any]]:
        """Get currently trending movies based on recent interactions"""
        movie_scores = defaultdict(float)
        current_time = datetime.now()
        
        # Weight recent interactions more heavily
        for interaction in self.recent_interactions:
            if interaction.movie_title:
                # Calculate time decay (more recent = higher weight)
                time_diff = (current_time - interaction.timestamp).total_seconds()
                weight = max(0.1, 1.0 - (time_diff / 3600))  # Decay over 1 hour
                
                # Different interaction types have different weights
                type_weights = {
                    InteractionType.RATING: 3.0,
                    InteractionType.MOVIE_DETAIL_VIEW: 2.0,
                    InteractionType.RECOMMENDATION_CLICK: 2.5,
                    InteractionType.VIEW: 1.0,
                    InteractionType.CLICK: 1.5
                }
                
                interaction_weight = type_weights.get(interaction.interaction_type, 1.0)
                movie_scores[interaction.movie_title] += weight * interaction_weight
        
        # Sort and return top movies
        trending = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        return [{'title': title, 'trend_score': score} for title, score in trending]
    
    def _get_recent_activity_summary(self) -> Dict[str, Any]:
        """Get summary of recent activity"""
        current_time = datetime.now()
        last_hour = current_time - timedelta(hours=1)
        
        recent_interactions = [i for i in self.recent_interactions 
                             if i.timestamp >= last_hour]
        
        activity_by_type = defaultdict(int)
        unique_users = set()
        
        for interaction in recent_interactions:
            activity_by_type[interaction.interaction_type.value] += 1
            unique_users.add(interaction.user_id)
        
        return {
            'total_interactions_last_hour': len(recent_interactions),
            'unique_users_last_hour': len(unique_users),
            'activity_breakdown': dict(activity_by_type)
        }
    
    def _calculate_user_engagement(self) -> Dict[str, float]:
        """Calculate user engagement metrics"""
        if not self.user_sessions:
            return {'average_session_length': 0, 'average_interactions_per_session': 0}
        
        session_lengths = []
        interactions_per_session = []
        current_time = datetime.now()
        
        for session_data in self.user_sessions.values():
            session_length = (session_data['last_activity'] - session_data['start_time']).total_seconds()
            session_lengths.append(session_length)
            interactions_per_session.append(session_data['interaction_count'])
        
        return {
            'average_session_length': sum(session_lengths) / len(session_lengths) if session_lengths else 0,
            'average_interactions_per_session': sum(interactions_per_session) / len(interactions_per_session) if interactions_per_session else 0,
            'active_sessions': len(self.user_sessions)
        }
    
    def get_user_behavior_analysis(self, user_id: str) -> Dict[str, Any]:
        """Analyze specific user's behavior patterns"""
        try:
            with self.driver.session(database=self.database) as session:
                # Get user's interaction history
                result = session.run("""
                    MATCH (u:User {userId: $userId})-[:PERFORMED]->(i:Interaction)
                    RETURN i
                    ORDER BY i.timestamp DESC
                    LIMIT 1000
                """, userId=user_id)
                
                interactions = [dict(record['i']) for record in result]
                
                if not interactions:
                    return {'error': 'No interaction data found for user'}
                
                # Analyze patterns
                analysis = {
                    'total_interactions': len(interactions),
                    'interaction_types': self._analyze_interaction_types(interactions),
                    'activity_patterns': self._analyze_activity_patterns(interactions),
                    'movie_preferences': self._analyze_movie_preferences(user_id),
                    'engagement_score': self._calculate_engagement_score(interactions),
                    'recommendation_effectiveness': self._analyze_recommendation_effectiveness(user_id)
                }
                
                return analysis
                
        except Exception as e:
            self.logger.error(f"Error analyzing user behavior: {e}")
            return {'error': 'Failed to analyze user behavior'}
    
    def _analyze_interaction_types(self, interactions: List[Dict]) -> Dict[str, Any]:
        """Analyze distribution of interaction types"""
        type_counts = defaultdict(int)
        for interaction in interactions:
            type_counts[interaction['type']] += 1
        
        total = len(interactions)
        return {
            'counts': dict(type_counts),
            'percentages': {k: (v/total)*100 for k, v in type_counts.items()}
        }
    
    def _analyze_activity_patterns(self, interactions: List[Dict]) -> Dict[str, Any]:
        """Analyze user activity patterns by time"""
        hour_counts = defaultdict(int)
        day_counts = defaultdict(int)
        
        for interaction in interactions:
            timestamp = datetime.fromisoformat(interaction['timestamp'].replace('Z', '+00:00'))
            hour_counts[timestamp.hour] += 1
            day_counts[timestamp.weekday()] += 1
        
        return {
            'most_active_hour': max(hour_counts.items(), key=lambda x: x[1])[0] if hour_counts else None,
            'most_active_day': max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else None,
            'hourly_distribution': dict(hour_counts),
            'daily_distribution': dict(day_counts)
        }
    
    def _analyze_movie_preferences(self, user_id: str) -> Dict[str, Any]:
        """Analyze user's movie preferences from interactions"""
        try:
            with self.driver.session(database=self.database) as session:
                # Get user's highly rated movies and their genres
                result = session.run("""
                    MATCH (u:User {userId: $userId})-[r:RATED]->(m:Movie)-[:IN_GENRE]->(g:Genre)
                    WHERE r.rating >= 4.0
                    RETURN g.name as genre, count(*) as count, avg(r.rating) as avgRating
                    ORDER BY count DESC, avgRating DESC
                    LIMIT 10
                """, userId=user_id)
                
                preferred_genres = [{'genre': record['genre'], 
                                   'count': record['count'], 
                                   'avg_rating': record['avgRating']} 
                                  for record in result]
                
                # Get most viewed movies
                result = session.run("""
                    MATCH (u:User {userId: $userId})-[:PERFORMED]->(i:Interaction)-[:INVOLVES]->(m:Movie)
                    WHERE i.type IN ['view', 'movie_detail_view', 'click']
                    RETURN m.title as title, count(*) as viewCount
                    ORDER BY viewCount DESC
                    LIMIT 10
                """, userId=user_id)
                
                most_viewed = [{'title': record['title'], 'view_count': record['viewCount']} 
                              for record in result]
                
                return {
                    'preferred_genres': preferred_genres,
                    'most_viewed_movies': most_viewed
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing movie preferences: {e}")
            return {'preferred_genres': [], 'most_viewed_movies': []}
    
    def _calculate_engagement_score(self, interactions: List[Dict]) -> float:
        """Calculate user engagement score based on interaction patterns"""
        if not interactions:
            return 0.0
        
        # Weight different interaction types
        weights = {
            'rating': 5.0,
            'movie_detail_view': 3.0,
            'recommendation_click': 4.0,
            'search': 2.0,
            'view': 1.0,
            'click': 1.5
        }
        
        total_score = 0
        for interaction in interactions:
            interaction_type = interaction['type']
            weight = weights.get(interaction_type, 1.0)
            total_score += weight
        
        # Normalize by number of interactions and time span
        if len(interactions) > 1:
            first_interaction = datetime.fromisoformat(interactions[-1]['timestamp'].replace('Z', '+00:00'))
            last_interaction = datetime.fromisoformat(interactions[0]['timestamp'].replace('Z', '+00:00'))
            time_span_days = max(1, (last_interaction - first_interaction).days)
            
            engagement_score = (total_score / len(interactions)) * (len(interactions) / time_span_days)
        else:
            engagement_score = total_score
        
        return min(10.0, engagement_score)  # Cap at 10
    
    def _analyze_recommendation_effectiveness(self, user_id: str) -> Dict[str, Any]:
        """Analyze how effective recommendations are for this user"""
        try:
            with self.driver.session(database=self.database) as session:
                # Get recommendation clicks vs views
                result = session.run("""
                    MATCH (u:User {userId: $userId})-[:PERFORMED]->(i:Interaction)
                    WHERE i.type IN ['recommendation_view', 'recommendation_click']
                    RETURN i.type as type, count(*) as count
                """, userId=user_id)
                
                rec_stats = {record['type']: record['count'] for record in result}
                
                views = rec_stats.get('recommendation_view', 0)
                clicks = rec_stats.get('recommendation_click', 0)
                
                click_through_rate = (clicks / views * 100) if views > 0 else 0
                
                return {
                    'recommendation_views': views,
                    'recommendation_clicks': clicks,
                    'click_through_rate': click_through_rate,
                    'effectiveness_rating': 'High' if click_through_rate > 15 else 'Medium' if click_through_rate > 5 else 'Low'
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing recommendation effectiveness: {e}")
            return {'recommendation_views': 0, 'recommendation_clicks': 0, 'click_through_rate': 0}
    
    def get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance and health metrics"""
        current_time = datetime.now()
        
        return {
            'processing_queue_size': len(self.processing_queue),
            'recent_interactions_count': len(self.recent_interactions),
            'active_sessions_count': len(self.user_sessions),
            'cache_hit_ratio': self._calculate_cache_hit_ratio(),
            'system_uptime': (current_time - datetime.now()).total_seconds(),  # Placeholder
            'interactions_per_second': self._calculate_interactions_per_second(),
        }
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio for analytics"""
        # This is a simplified implementation
        # In production, you'd track actual cache hits/misses
        return 0.85  # Placeholder
    
    def _calculate_interactions_per_second(self) -> float:
        """Calculate current interactions per second rate"""
        current_time = datetime.now()
        last_minute = current_time - timedelta(minutes=1)
        
        recent_count = len([i for i in self.recent_interactions 
                           if i.timestamp >= last_minute])
        
        return recent_count / 60.0  # Per second
    
    def stop(self):
        """Stop the background processing thread"""
        self.stop_processing = True
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        self.logger.info("Stopped real-time tracker")
