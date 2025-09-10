# 🎬 CineAI - Sophisticated Movie Recommendation System

A production-ready, AI-powered movie recommendation platform built with Neo4j, Flask, and advanced machine learning algorithms. This system combines multiple recommendation techniques including collaborative filtering, content-based filtering, matrix factorization, and real-time behavioral analysis.

## 🌟 Key Features

### Advanced Machine Learning
- **Hybrid Recommendation Engine**: Combines 4+ ML algorithms with intelligent weighting
- **Matrix Factorization**: Non-negative Matrix Factorization for latent feature discovery
- **Content-Based Filtering**: TF-IDF vectorization with cosine similarity
- **Collaborative Filtering**: Advanced user-based and item-based approaches
- **Context-Aware Recommendations**: Time-based and behavioral context integration
- **Real-Time Learning**: Continuous model adaptation based on user interactions

### Sophisticated Analytics
- **Real-Time Tracking**: 14+ interaction types with live processing
- **Behavioral Analysis**: User engagement patterns and preference modeling
- **Performance Monitoring**: System health and recommendation effectiveness metrics
- **Interactive Dashboards**: D3.js and Chart.js powered visualizations
- **A/B Testing Framework**: Algorithm performance comparison and optimization

### Production Features
- **User Authentication**: Secure registration/login with bcrypt hashing
- **Graph Database**: Neo4j with optimized schema and relationships
- **RESTful APIs**: Comprehensive endpoints for all functionality
- **Responsive Design**: Modern UI with Tailwind CSS and interactive elements
- **Scalable Architecture**: Background processing and caching strategies

## 🏗️ System Architecture

\`\`\`
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask App      │    │   Neo4j DB      │
│                 │    │                  │    │                 │
│ • HTML/CSS/JS   │◄──►│ • Authentication │◄──►│ • Movie Graph   │
│ • D3.js Graphs  │    │ • Recommendation │    │ • User Profiles │
│ • Analytics UI  │    │ • Real-time API  │    │ • Interactions  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   ML Engine      │
                       │                  │
                       │ • Hybrid Models  │
                       │ • Matrix Factor. │
                       │ • Content Filter │
                       │ • Collab Filter  │
                       └──────────────────┘
\`\`\`

## 🧠 Recommendation Algorithms

### 1. Content-Based Filtering
Analyzes movie features (genres, directors, actors) using TF-IDF vectorization:

\`\`\`python
def content_based_recommendations(self, movie_title, limit=10):
    # Extract movie features and create TF-IDF vectors
    movie_features = self.get_movie_features(movie_title)
    tfidf_matrix = self.vectorizer.fit_transform(all_features)
    
    # Calculate cosine similarity
    similarity_scores = cosine_similarity(movie_vector, tfidf_matrix)
    return sorted_recommendations[:limit]
\`\`\`

**Strengths**: Works for new users, explainable recommendations
**Use Case**: Cold start problem, genre-based preferences

### 2. Collaborative Filtering
Finds users with similar preferences using Pearson correlation:

\`\`\`python
def collaborative_filtering(self, user_id, limit=10):
    # Find similar users based on rating patterns
    similar_users = self.find_similar_users(user_id)
    
    # Aggregate recommendations from similar users
    recommendations = self.weighted_user_recommendations(similar_users)
    return recommendations[:limit]
\`\`\`

**Strengths**: Captures complex preferences, improves with data
**Use Case**: Active users with sufficient rating history

### 3. Matrix Factorization
Advanced collaborative filtering using Non-negative Matrix Factorization:

\`\`\`python
def matrix_factorization_recommendations(self, user_id, limit=10):
    # Decompose user-item matrix into latent factors
    W, H = self.nmf_model.fit_transform(interaction_matrix)
    
    # Predict ratings for unrated movies
    predicted_ratings = np.dot(W[user_idx], H)
    return top_predictions[:limit]
\`\`\`

**Strengths**: Handles sparse data, discovers latent factors
**Use Case**: Large-scale systems, pattern discovery

### 4. Hybrid Engine
Intelligently combines algorithms with dynamic weighting:

\`\`\`python
def get_hybrid_recommendations(self, user_id, context=None):
    # Calculate algorithm weights based on data availability
    weights = self.calculate_dynamic_weights(user_id)
    
    # Get recommendations from each algorithm
    content_recs = self.content_based_recommendations(user_id)
    collab_recs = self.collaborative_filtering(user_id)
    matrix_recs = self.matrix_factorization_recommendations(user_id)
    
    # Weighted combination with diversity optimization
    return self.ensemble_and_diversify(
        content_recs * weights['content'],
        collab_recs * weights['collaborative'],
        matrix_recs * weights['matrix']
    )
\`\`\`

**Adaptive Weighting**:
- **New Users**: Higher content-based weight (70%)
- **Active Users**: Higher collaborative weight (60%)
- **Sparse Data**: Higher matrix factorization weight (50%)

## 📊 Real-Time Analytics System

### Interaction Tracking
Monitors 14+ interaction types in real-time:

\`\`\`python
INTERACTION_TYPES = [
    'movie_view', 'movie_rating', 'movie_search', 'recommendation_click',
    'profile_update', 'genre_filter', 'actor_click', 'director_click',
    'similar_movie_click', 'watchlist_add', 'favorite_add', 'share_movie',
    'review_submit', 'recommendation_feedback'
]
\`\`\`

### Behavioral Analysis
Advanced user behavior modeling:

\`\`\`python
def analyze_user_behavior(self, user_id):
    interactions = self.get_user_interactions(user_id)
    
    return {
        'engagement_score': self.calculate_engagement_score(interactions),
        'genre_preferences': self.extract_genre_preferences(interactions),
        'viewing_patterns': self.analyze_temporal_patterns(interactions),
        'diversity_preference': self.measure_diversity_preference(interactions),
        'social_influence': self.calculate_social_influence(interactions)
    }
\`\`\`

### Performance Metrics
Comprehensive system monitoring:

- **Accuracy Metrics**: Precision@K, Recall@K, RMSE, MAE
- **Diversity Metrics**: Intra-list diversity, genre coverage
- **Business Metrics**: Click-through rate, user engagement, retention
- **System Metrics**: Response time, cache hit rate, error rate

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- Neo4j Database 4.0+
- 8GB+ RAM (recommended for ML operations)

### 1. Environment Setup
\`\`\`bash
# Clone repository
git clone <repository-url>
cd moviespythonbolt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

### 2. Neo4j Configuration
\`\`\`bash
# Start Neo4j database
neo4j start

# Set environment variables
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export FLASK_SECRET_KEY="your_secret_key"
\`\`\`

### 3. Database Initialization
\`\`\`bash
# Run enhanced schema setup
python scripts/enhanced_schema_setup.py


\`\`\`

### 4. Launch Application
\`\`\`bash
# Start Flask application
python movies_sync.py

# Access at http://localhost:5000
\`\`\`

## 🔧 System Components

### Core Files

#### `movies_sync.py` (840 lines)
Main Flask application with:
- User authentication and session management
- RESTful API endpoints for all functionality
- Real-time interaction tracking integration
- Security decorators and input validation

#### `advanced_recommendation_engine.py` (544 lines)
Sophisticated ML engine featuring:
- Hybrid recommendation algorithms
- Matrix factorization with NMF
- Content-based filtering with TF-IDF
- Collaborative filtering with Pearson correlation
- Context-aware recommendation logic

#### `real_time_tracker.py` (530 lines)
Real-time analytics system with:
- Background thread processing
- Comprehensive interaction tracking
- User behavior analysis algorithms
- Performance monitoring and metrics

#### `scripts/enhanced_schema_setup.py` (225 lines)
Database schema initialization:
- Advanced Neo4j graph relationships
- Performance optimization indexes
- Constraint definitions for data integrity

### Frontend Templates

#### `static/index.html` (187 lines)
Main dashboard featuring:
- Interactive D3.js movie network visualization
- Real-time movie search functionality
- Graph-based movie discovery interface

#### `static/advanced_recommendations.html` (238 lines)
AI-powered recommendation interface:
- Ensemble recommendation results display
- Algorithm transparency and explanations
- Interactive rating and feedback system

#### `static/analytics_dashboard.html` (329 lines)
Comprehensive analytics dashboard:
- Real-time system metrics with auto-refresh
- Interactive Chart.js visualizations
- Performance monitoring and health indicators

#### `static/user_behavior.html` (458 lines)
Personal analytics interface:
- Individual user behavior pattern analysis
- Engagement scoring and preference visualization
- Personalized recommendation insights

## 📡 API Endpoints

### Core Functionality
\`\`\`
GET  /                          # Main dashboard with graph visualization
GET  /search?q=<query>          # Movie search with autocomplete
GET  /movie/<title>             # Detailed movie information
POST /movie/<title>/vote        # Rate/vote for movies
GET  /graph                     # Network graph data for D3.js
\`\`\`

### Recommendations
\`\`\`
GET  /recommend                 # Basic recommendation form
POST /recommend                 # Generate personalized recommendations
GET  /advanced_recommendations  # Advanced ML-powered interface
POST /api/feedback              # Recommendation feedback tracking
\`\`\`

### User Management
\`\`\`
GET  /register                  # User registration page
POST /register                  # Create new user account
GET  /login                     # User login interface
POST /login                     # Authenticate user session
GET  /logout                    # User logout and session cleanup
GET  /profile                   # User profile and preferences
POST /profile/update            # Update user profile data
\`\`\`

### Analytics & Tracking
\`\`\`
GET  /analytics                 # System analytics dashboard
GET  /user_behavior             # Personal behavior analysis
POST /track_interaction         # Real-time interaction tracking
GET  /api/user_stats/<user_id>  # Individual user statistics
GET  /api/system_metrics        # System performance metrics
\`\`\`

## 🎯 Advanced Features

### Context-Aware Recommendations
The system considers multiple contextual factors:

\`\`\`python
def get_contextual_recommendations(self, user_id, context):
    base_recs = self.get_hybrid_recommendations(user_id)
    
    # Time-based adjustments
    if context.get('time_of_day') == 'evening':
        base_recs = self.boost_genres(['Drama', 'Thriller'], base_recs)
    
    # Device-based filtering
    if context.get('device') == 'mobile':
        base_recs = self.filter_by_duration(base_recs, max_duration=120)
    
    # Social context
    if context.get('with_friends'):
        base_recs = self.boost_social_movies(base_recs)
    
    return base_recs
\`\`\`

### Real-Time Model Updates
Continuous learning from user interactions:

\`\`\`python
def update_models_realtime(self, interaction):
    # Update user preferences
    self.update_user_profile(interaction)
    
    # Refresh similarity calculations
    if interaction['type'] in ['rating', 'favorite']:
        self.refresh_user_similarities(interaction['user_id'])
    
    # Update content features
    if interaction['type'] == 'movie_view':
        self.update_movie_popularity(interaction['movie_id'])
\`\`\`

### A/B Testing Framework
Algorithm performance optimization:

\`\`\`python
def ab_test_recommendations(self, user_id):
    test_group = self.get_user_test_group(user_id)
    
    if test_group == 'control':
        return self.standard_hybrid_recommendations(user_id)
    elif test_group == 'experimental':
        return self.deep_learning_recommendations(user_id)
    
    # Track metrics for analysis
    self.track_ab_test_performance(user_id, test_group)
\`\`\`

## 📈 Performance Optimization

### Caching Strategy
- **In-Memory Caching**: Frequent recommendations and user data
- **Redis Integration**: Distributed caching for scalability
- **Precomputed Similarities**: Background calculation of user/item similarities
- **Incremental Updates**: Efficient cache invalidation strategies

### Database Optimization
- **Neo4j Indexing**: Optimized queries with proper indexes
- **Query Optimization**: Efficient Cypher queries for complex relationships
- **Connection Pooling**: Managed database connections
- **Batch Processing**: Bulk operations for data updates

### Machine Learning Optimization
- **Model Caching**: Precomputed model predictions
- **Incremental Learning**: Online model updates
- **Feature Engineering**: Optimized feature extraction
- **Parallel Processing**: Multi-threaded recommendation generation

## 🔍 Evaluation Metrics

### Accuracy Metrics
- **Precision@K**: Relevant items in top-K recommendations
- **Recall@K**: Coverage of relevant items
- **RMSE/MAE**: Rating prediction accuracy
- **NDCG**: Normalized Discounted Cumulative Gain

### Beyond Accuracy
- **Diversity**: Genre and type variety in recommendations
- **Novelty**: Recommendation of less popular items
- **Coverage**: Percentage of recommendable items
- **Serendipity**: Unexpected but relevant discoveries

### Business Metrics
- **Click-Through Rate**: User engagement with recommendations
- **Conversion Rate**: Actions taken on recommendations
- **User Retention**: Long-term platform engagement
- **Session Duration**: Time spent with recommendations

## 🚀 Future Enhancements

### Advanced ML Techniques
- **Deep Learning**: Neural collaborative filtering and autoencoders
- **Graph Neural Networks**: Leverage Neo4j graph structure for GNNs
- **Reinforcement Learning**: Optimize long-term user satisfaction
- **Natural Language Processing**: Analyze reviews and movie descriptions

### Enhanced Features
- **Multi-Modal Recommendations**: Image, video, and text analysis
- **Cross-Domain Recommendations**: Books, music, TV shows integration
- **Social Recommendations**: Friend-based and community features
- **Real-Time Streaming**: Live recommendation updates via WebSockets

### Technical Improvements
- **Microservices Architecture**: Containerized service separation
- **GraphQL API**: Flexible and efficient data querying
- **Edge Computing**: Reduced latency with edge deployments
- **Auto-Scaling**: Dynamic resource allocation based on load

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Neo4j**: Powerful graph database technology
- **scikit-learn**: Machine learning algorithms and utilities
- **Flask**: Lightweight and flexible web framework
- **D3.js**: Beautiful and interactive data visualizations
- **Tailwind CSS**: Modern utility-first CSS framework

---

**CineAI** - Revolutionizing movie discovery through advanced AI and sophisticated graph technology. 🎬✨

*Built with ❤️ for movie enthusiasts and data scientists.*
\`\`\`

```txt file="" isHidden
