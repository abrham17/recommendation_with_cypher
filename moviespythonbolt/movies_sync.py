#!/usr/bin/env python
import logging
import os
from json import dumps
from textwrap import dedent
from typing import cast
import hashlib
from datetime import datetime
import math
from functools import wraps
from dotenv import load_dotenv
import neo4j
from flask import Flask, Response, request, render_template, session, redirect, url_for, flash, jsonify
from neo4j import GraphDatabase, basic_auth
from typing_extensions import LiteralString
from werkzeug.security import generate_password_hash, check_password_hash

from advanced_recommendation_engine import AdvancedHybridRecommendationEngine
from real_time_tracker import RealTimeTracker, InteractionType
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
neo4j_version = os.getenv("NEO4J_VERSION")
database = os.getenv("NEO4J_DATABASE")

port = int(os.getenv("PORT", 8080))

driver = GraphDatabase.driver(url, auth=basic_auth(username, password))

advanced_rec_engine = AdvancedHybridRecommendationEngine(driver, database)
real_time_tracker = RealTimeTracker(driver, database)

# Initialize models on startup (in production, this should be done asynchronously)
try:
    advanced_rec_engine.initialize_models()
    logging.info("Advanced recommendation models initialized")
except Exception as e:
    logging.error(f"Failed to initialize recommendation models: {e}")

def query(q: LiteralString) -> LiteralString:
    # this is a safe transform:
    # no way for cypher injection by trimming whitespace
    # hence, we can safely cast to LiteralString
    return cast(LiteralString, dedent(q).strip())


def login_required(f):
    """Decorator to require login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def track_interaction(interaction_type: InteractionType):
    """Decorator to automatically track user interactions"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Execute the original function first
            result = f(*args, **kwargs)
            
            # Track the interaction if user is logged in
            if 'user_id' in session:
                try:
                    # Get additional context from request
                    movie_title = kwargs.get('title') or request.args.get('movie_title')
                    search_query = request.args.get('q')
                    
                    real_time_tracker.track_interaction(
                        user_id=session['user_id'],
                        interaction_type=interaction_type,
                        movie_title=movie_title,
                        search_query=search_query,
                        page_url=request.url,
                        user_agent=request.headers.get('User-Agent', ''),
                        ip_address=request.remote_addr
                    )
                except Exception as e:
                    logging.error(f"Error tracking interaction: {e}")
            
            return result
        return decorated_function
    return decorator


def get_current_user():
    """Get current logged-in user information"""
    if 'user_id' not in session:
        return None
    
    with driver.session(database=database) as db_session:
        result = db_session.run(query("""
            MATCH (u:User {userId: $userId})
            RETURN u
        """), userId=session['user_id'])
        
        record = result.single()
        return dict(record['u']) if record else None


def create_user(email, password, name, age=None, gender=None):
    """Create a new user account"""
    password_hash = generate_password_hash(password)
    
    with driver.session(database=database) as db_session:
        # Check if user already exists
        existing = db_session.run(query("""
            MATCH (u:User {email: $email})
            RETURN u
        """), email=email).single()
        
        if existing:
            return None, "User with this email already exists"
        
        # Create new user
        result = db_session.run(query("""
            CREATE (u:User {
                userId: randomUUID(),
                email: $email,
                passwordHash: $passwordHash,
                name: $name,
                age: $age,
                gender: $gender,
                registrationDate: datetime(),
                avgRating: 3.5,
                totalRatings: 0,
                preferredGenres: [],
                activityLevel: 'new'
            })
            RETURN u
        """), email=email, passwordHash=password_hash, name=name, age=age, gender=gender)
        
        user = result.single()
        return dict(user['u']) if user else None, None


def authenticate_user(email, password):
    """Authenticate user login"""
    with driver.session(database=database) as db_session:
        result = db_session.run(query("""
            MATCH (u:User {email: $email})
            RETURN u
        """), email=email)
        
        user_record = result.single()
        if not user_record:
            return None, "Invalid email or password"
        
        user = dict(user_record['u'])
        if check_password_hash(user['passwordHash'], password):
            return user, None
        else:
            return None, "Invalid email or password"


@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration"""
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        name = request.form.get('name')
        age = request.form.get('age', type=int)
        gender = request.form.get('gender')
        
        # Validation
        if not email or not password or not name:
            flash('Please fill in all required fields.', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'error')
            return render_template('register.html')
        
        # Create user
        user, error = create_user(email, password, name, age, gender)
        if error:
            flash(error, 'error')
            return render_template('register.html')
        
        # Auto-login after registration
        session['user_id'] = user['userId']
        session['user_name'] = user['name']
        flash('Registration successful! Welcome to Neo4j Movies!', 'success')
        return redirect(url_for('get_index'))
    
    return render_template('register.html')


@app.route("/login", methods=["GET", "POST"])
@track_interaction(InteractionType.LOGIN)
def login():
    """User login"""
    if request.method == "POST":
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Please enter both email and password.', 'error')
            return render_template('login.html')
        
        user, error = authenticate_user(email, password)
        if error:
            flash(error, 'error')
            return render_template('login.html')
        
        # Set session
        session['user_id'] = user['userId']
        session['user_name'] = user['name']
        flash(f'Welcome back, {user["name"]}!', 'success')
        
        # Redirect to intended page or home
        next_page = request.args.get('next')
        return redirect(next_page) if next_page else redirect(url_for('get_index'))
    
    return render_template('login.html')


@app.route("/logout")
@track_interaction(InteractionType.LOGOUT)
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('get_index'))


@app.route("/profile")
@login_required
@track_interaction(InteractionType.PROFILE_VIEW)
def profile():
    """User profile page"""
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    
    # Get user statistics
    with driver.session(database=database) as db_session:
        stats = db_session.run(query("""
            MATCH (u:User {userId: $userId})
            OPTIONAL MATCH (u)-[r:RATED]->(m:Movie)
            OPTIONAL MATCH (u)-[v:VIEWED]->(m2:Movie)
            WITH u, count(DISTINCT r) as ratingsCount, count(DISTINCT v) as viewsCount,
                 avg(r.rating) as avgRating
            OPTIONAL MATCH (u)-[r2:RATED]->(m3:Movie)-[:IN_GENRE]->(g:Genre)
            WHERE r2.rating >= 4
            WITH u, ratingsCount, viewsCount, avgRating, g
            RETURN ratingsCount, viewsCount, avgRating,
                   collect(DISTINCT g.name)[0..5] as topGenres
        """), userId=user['userId']).single()
        
        user_stats = {
            'ratingsCount': stats['ratingsCount'] if stats else 0,
            'viewsCount': stats['viewsCount'] if stats else 0,
            'avgRating': round(stats['avgRating'], 1) if stats and stats['avgRating'] else 0,
            'topGenres': stats['topGenres'] if stats else []
        }
    
    return render_template('profile.html', user=user, stats=user_stats)


@app.route("/profile/edit", methods=["GET", "POST"])
@login_required
def edit_profile():
    """Edit user profile"""
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    
    if request.method == "POST":
        name = request.form.get('name')
        age = request.form.get('age', type=int)
        gender = request.form.get('gender')
        
        with driver.session(database=database) as db_session:
            db_session.run(query("""
                MATCH (u:User {userId: $userId})
                SET u.name = $name,
                    u.age = $age,
                    u.gender = $gender
            """), userId=user['userId'], name=name, age=age, gender=gender)
        
        session['user_name'] = name
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('edit_profile.html', user=user)


@app.route("/")
@track_interaction(InteractionType.VIEW)
def get_index():
    return render_template('index.html', user=get_current_user())


@app.route("/graph")
def get_graph():
    records, _, _ = driver.execute_query(
        query("""
            MATCH (m:Movie)<-[:ACTED_IN]-(a:Person)
            RETURN m.title AS movie, collect(a.name) AS cast
            LIMIT $limit
        """),
        database_=database,
        routing_="r",
        limit=request.args.get("limit", 100)
    )
    nodes = []
    rels = []
    i = 0
    for record in records:
        nodes.append({"title": record["movie"], "label": "movie"})
        target = i
        i += 1
        for name in record["cast"]:
            actor = {"title": name, "label": "actor"}
            try:
                source = nodes.index(actor)
            except ValueError:
                nodes.append(actor)
                source = i
                i += 1
            rels.append({"source": source, "target": target})
    return Response(dumps({"nodes": nodes, "links": rels}),
                    mimetype="application/json")


@app.route("/search")
@track_interaction(InteractionType.SEARCH)
def get_search():
    try:
        q = request.args["q"]
    except KeyError:
        return []
    else:
        records, _, _ = driver.execute_query(
            query("""
                MATCH (movie:Movie)
                WHERE toLower(movie.title) CONTAINS toLower($title)
                OPTIONAL MATCH (movie)-[:IN_GENRE]->(g:Genre)
                WITH movie, collect(g.name) as genres
                RETURN movie, genres
                ORDER BY movie.popularity DESC
            """),
            title=q,
            database_=database,
            routing_="r",
        )
        results = []
        for record in records:
            movie_data = dict(record["movie"])
            movie_data["genres"] = record["genres"]
            results.append(serialize_movie(movie_data))
        
        return Response(dumps(results), mimetype="application/json")


@app.route("/movie/<title>")
@track_interaction(InteractionType.MOVIE_DETAIL_VIEW)
def get_movie(title):
    result = driver.execute_query(
        query("""
            MATCH (movie:Movie {title:$title})
            OPTIONAL MATCH (movie)<-[r]-(person:Person)
            OPTIONAL MATCH (movie)-[:IN_GENRE]->(g:Genre)
            RETURN movie.title as title,
            COLLECT(
                [person.name, HEAD(SPLIT(TOLOWER(TYPE(r)), '_')), r.roles]
            ) AS cast,
            collect(DISTINCT g.name) as genres
            LIMIT 1
        """),
        title=title,
        database_=database,
        routing_="r",
        result_transformer_=neo4j.Result.single,
    )
    if not result:
        return Response(dumps({"error": "Movie not found"}), status=404,
                        mimetype="application/json")

    return Response(dumps({"title": result["title"],
                           "cast": [serialize_cast(member)
                                    for member in result["cast"]],
                           "genres": result["genres"]}),
                    mimetype="application/json")


@app.route("/movie/<title>/vote", methods=["POST"])
@track_interaction(InteractionType.CLICK)
def vote_in_movie(title):
    summary = driver.execute_query(
        query("""
            MATCH (m:Movie {title: $title})
            SET m.votes = coalesce(m.votes, 0) + 1;
        """),
        database_=database,
        title=title,
        result_transformer_=neo4j.Result.consume,
    )
    updates = summary.counters.properties_set
    return Response(dumps({"updates": updates}), mimetype="application/json")


@app.route("/recommend/content/<title>")
@track_interaction(InteractionType.RECOMMENDATION_VIEW)
def recommend_content(title):
    """Enhanced content-based recommendations"""
    recommendations = advanced_rec_engine.content_based_recommendations(title, limit=10)
    return render_template("recommendations.html", 
                         recommendations=recommendations, 
                         method="Advanced Content-Based", 
                         base_movie=title)


@app.route("/recommend/collaborative/<int:user_id>")
@track_interaction(InteractionType.RECOMMENDATION_VIEW)
def recommend_collaborative(user_id):
    """Enhanced collaborative filtering recommendations"""
    recommendations = advanced_rec_engine.collaborative_filtering_recommendations(user_id, limit=10)
    return render_template("recommendations.html", 
                         recommendations=recommendations, 
                         method="Advanced Collaborative Filtering", 
                         user_id=user_id)


@app.route("/recommend/hybrid")
@track_interaction(InteractionType.RECOMMENDATION_VIEW)
def recommend_hybrid():
    """Hybrid recommendations combining multiple approaches"""
    user_id = request.args.get('user_id', type=int)
    movie_title = request.args.get('movie_title')
    
    if not user_id and not movie_title:
        return render_template("recommend_form.html")
    
    recommendations = advanced_rec_engine.hybrid_recommendations(
        user_id=user_id, 
        movie_title=movie_title, 
        limit=15
    )
    
    return render_template("recommendations.html", 
                         recommendations=recommendations, 
                         method="Hybrid AI-Powered", 
                         user_id=user_id,
                         base_movie=movie_title)


@app.route("/recommend/trending")
@track_interaction(InteractionType.RECOMMENDATION_VIEW)
def recommend_trending():
    """Get trending movie recommendations"""
    recommendations = advanced_rec_engine.trending_recommendations(limit=10)
    return render_template("recommendations.html", 
                         recommendations=recommendations, 
                         method="Trending Now")


@app.route("/api/track", methods=["POST"])
def api_track_interaction():
    """API endpoint for tracking user interactions from frontend"""
    if 'user_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        interaction_type = InteractionType(data.get('type'))
        
        interaction_id = real_time_tracker.track_interaction(
            user_id=session['user_id'],
            interaction_type=interaction_type,
            movie_title=data.get('movie_title'),
            search_query=data.get('search_query'),
            page_url=data.get('page_url', request.referrer),
            user_agent=request.headers.get('User-Agent', ''),
            ip_address=request.remote_addr,
            duration=data.get('duration'),
            metadata=data.get('metadata', {})
        )
        
        return jsonify({
            'status': 'success',
            'interaction_id': interaction_id
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid interaction type: {e}'}), 400
    except Exception as e:
        logging.error(f"Error in API track: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route("/api/analytics/real-time")
@login_required
def api_real_time_analytics():
    """API endpoint for real-time analytics dashboard"""
    try:
        analytics = real_time_tracker.get_real_time_analytics()
        return jsonify(analytics)
    except Exception as e:
        logging.error(f"Error getting real-time analytics: {e}")
        return jsonify({'error': 'Failed to get analytics'}), 500


@app.route("/api/analytics/user/<user_id>")
@login_required
def api_user_behavior_analysis(user_id):
    """API endpoint for user behavior analysis"""
    # Check if user can access this data (admin or own data)
    current_user = get_current_user()
    if current_user['userId'] != user_id:
        # In production, add admin check here
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        analysis = real_time_tracker.get_user_behavior_analysis(user_id)
        return jsonify(analysis)
    except Exception as e:
        logging.error(f"Error getting user behavior analysis: {e}")
        return jsonify({'error': 'Failed to analyze user behavior'}), 500


@app.route("/analytics/dashboard")
@login_required
def analytics_dashboard():
    """Analytics dashboard page"""
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    
    return render_template('analytics_dashboard.html', user=user)


@app.route("/analytics/user-behavior")
@login_required
def user_behavior_page():
    """User behavior analysis page"""
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    
    return render_template('user_behavior.html', user=user)


@app.route("/track/view", methods=["POST"])
def track_view():
    """Track movie viewing for implicit feedback"""
    data = request.get_json()
    user_id = data.get('user_id')
    movie_title = data.get('movie_title')
    duration = data.get('duration', 0)
    completed = data.get('completed', False)
    
    if not user_id or not movie_title:
        return Response(dumps({"error": "Missing required parameters"}), 
                       status=400, mimetype="application/json")
    
    # Track with real-time tracker
    real_time_tracker.track_interaction(
        user_id=user_id,
        interaction_type=InteractionType.VIEW,
        movie_title=movie_title,
        duration=duration,
        metadata={'completed': completed}
    )
    
    with driver.session(database=database) as session:
        session.run(query("""
            MATCH (u:User {userId: $userId}), (m:Movie {title: $movieTitle})
            MERGE (u)-[v:VIEWED]->(m)
            SET v.timestamp = datetime(),
                v.duration = $duration,
                v.completed = $completed,
                v.device = 'web'
        """), userId=user_id, movieTitle=movie_title, 
             duration=duration, completed=completed)
    
    return Response(dumps({"status": "tracked"}), mimetype="application/json")


@app.route("/recommend/personalized")
@login_required
@track_interaction(InteractionType.RECOMMENDATION_VIEW)
def recommend_personalized():
    """Get personalized recommendations for logged-in user"""
    user = get_current_user()
    recommendations = advanced_rec_engine.personalized_recommendations(user['userId'], limit=15)
    return render_template("recommendations.html", 
                         recommendations=recommendations, 
                         method="Personalized for You", 
                         user=user)


@app.route("/recommend/advanced-hybrid")
@login_required
@track_interaction(InteractionType.RECOMMENDATION_VIEW)
def recommend_advanced_hybrid():
    """Advanced hybrid recommendations using ensemble methods"""
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    
    # Get context information
    context = {
        'time_of_day': datetime.now().hour,
        'day_of_week': datetime.now().weekday(),
        'user_agent': request.headers.get('User-Agent', ''),
    }
    
    try:
        recommendations = advanced_rec_engine.ensemble_recommendations(
            user['userId'], 
            context=context, 
            n_recommendations=15
        )
        
        # Add explanations for each recommendation
        for rec in recommendations:
            rec['explanation'] = advanced_rec_engine.get_explanation(
                user['userId'], 
                rec['title'], 
                rec
            )
        
        return render_template("advanced_recommendations.html", 
                             recommendations=recommendations, 
                             method="Advanced AI Hybrid", 
                             user=user)
    
    except Exception as e:
        logging.error(f"Error generating advanced recommendations: {e}")
        flash("Unable to generate recommendations at this time. Please try again later.", "error")
        return redirect(url_for('get_index'))


@app.route("/recommend/collaborative-advanced/<user_id>")
@login_required
@track_interaction(InteractionType.RECOMMENDATION_VIEW)
def recommend_collaborative_advanced(user_id):
    """Advanced collaborative filtering with matrix factorization"""
    try:
        recommendations = advanced_rec_engine.collaborative_filtering_advanced(
            user_id, 
            n_recommendations=12
        )
        
        return render_template("recommendations.html", 
                             recommendations=recommendations, 
                             method="Advanced Collaborative Filtering", 
                             user_id=user_id)
    
    except Exception as e:
        logging.error(f"Error in advanced collaborative filtering: {e}")
        return render_template("recommendations.html", 
                             recommendations=[], 
                             method="Advanced Collaborative Filtering", 
                             error="Unable to generate recommendations")


@app.route("/recommend/content-advanced/<user_id>")
@login_required
@track_interaction(InteractionType.RECOMMENDATION_VIEW)
def recommend_content_advanced(user_id):
    """Advanced content-based filtering with TF-IDF"""
    try:
        recommendations = advanced_rec_engine.content_based_advanced(
            user_id, 
            n_recommendations=12
        )
        
        return render_template("recommendations.html", 
                             recommendations=recommendations, 
                             method="Advanced Content-Based Filtering", 
                             user_id=user_id)
    
    except Exception as e:
        logging.error(f"Error in advanced content filtering: {e}")
        return render_template("recommendations.html", 
                             recommendations=[], 
                             method="Advanced Content-Based Filtering", 
                             error="Unable to generate recommendations")


@app.route("/recommend/context-aware")
@login_required
@track_interaction(InteractionType.RECOMMENDATION_VIEW)
def recommend_context_aware():
    """Context-aware recommendations based on time and user behavior"""
    user = get_current_user()
    if not user:
        return redirect(url_for('login'))
    
    context = {
        'time_of_day': datetime.now().hour,
        'day_of_week': datetime.now().weekday(),
    }
    
    try:
        recommendations = advanced_rec_engine.context_aware_recommendations(
            user['userId'], 
            context=context, 
            n_recommendations=12
        )
        
        return render_template("recommendations.html", 
                             recommendations=recommendations, 
                             method="Context-Aware Recommendations", 
                             user=user,
                             context_info=f"Optimized for {recommendations[0]['time_period'] if recommendations else 'current time'}")
    
    except Exception as e:
        logging.error(f"Error in context-aware recommendations: {e}")
        return render_template("recommendations.html", 
                             recommendations=[], 
                             method="Context-Aware Recommendations", 
                             error="Unable to generate recommendations")


@app.route("/api/retrain-models", methods=["POST"])
@login_required
def retrain_models():
    """API endpoint to retrain recommendation models (admin only)"""
    user = get_current_user()
    
    # In production, add admin check here
    # if not user.get('is_admin', False):
    #     return Response(dumps({"error": "Unauthorized"}), status=403, mimetype="application/json")
    
    try:
        advanced_rec_engine.initialize_models()
        return Response(dumps({"status": "Models retrained successfully"}), 
                       mimetype="application/json")
    except Exception as e:
        logging.error(f"Error retraining models: {e}")
        return Response(dumps({"error": "Failed to retrain models"}), 
                       status=500, mimetype="application/json")


@app.route("/rate/<title>", methods=["POST"])
@login_required
def rate_movie(title):
    """Enhanced movie rating with user tracking"""
    data = request.get_json() if request.is_json else request.form
    rating = data.get('rating', type=float)
    
    if not rating or rating < 1 or rating > 5:
        return Response(dumps({"error": "Rating must be between 1 and 5"}), 
                       status=400, mimetype="application/json")
    
    user = get_current_user()
    
    # Track rating interaction
    real_time_tracker.track_interaction(
        user_id=user['userId'],
        interaction_type=InteractionType.RATING,
        movie_title=title,
        rating_value=rating
    )
    
    with driver.session(database=database) as db_session:
        # Create or update rating
        db_session.run(query("""
            MATCH (u:User {userId: $userId}), (m:Movie {title: $movieTitle})
            MERGE (u)-[r:RATED]->(m)
            SET r.rating = $rating,
                r.timestamp = datetime(),
                r.device = 'web'
        """), userId=user['userId'], movieTitle=title, rating=rating)
        
        # Update movie statistics
        db_session.run(query("""
            MATCH (m:Movie {title: $movieTitle})<-[r:RATED]-()
            WITH m, avg(r.rating) as avgRating, count(r) as ratingCount
            SET m.avgRating = avgRating,
                m.ratingCount = ratingCount,
                m.popularity = ratingCount * 0.7 + avgRating * 20
        """), movieTitle=title)
        
        # Update user statistics
        db_session.run(query("""
            MATCH (u:User {userId: $userId})-[r:RATED]->()
            WITH u, avg(r.rating) as avgRating, count(r) as totalRatings
            SET u.avgRating = avgRating,
                u.totalRatings = totalRatings
        """), userId=user['userId'])
    
    flash(f'You rated "{title}" {rating} stars!', 'success')
    return Response(dumps({"status": "rated", "rating": rating}), mimetype="application/json")


def serialize_movie(movie):
    return {
        "id": movie["id"],
        "title": movie["title"],
        "summary": movie["summary"],
        "released": movie["released"],
        "duration": movie["duration"],
        "rated": movie["rated"],
        "tagline": movie["tagline"],
        "votes": movie.get("votes", 0),
        "avgRating": movie.get("avgRating", 3.0),
        "ratingCount": movie.get("ratingCount", 0),
        "popularity": movie.get("popularity", 0),
        "genres": movie.get("genres", [])
    }


def serialize_cast(cast):
    return {
        "name": cast[0],
        "job": cast[1],
        "role": cast[2]
    }

if __name__ == "__main__":
    logging.root.setLevel(logging.INFO)
    logging.info("Starting on port %d, database is at %s", port, url)
    try:
        app.run(port=port, debug=True)
    finally:
        real_time_tracker.stop()
        driver.close()
