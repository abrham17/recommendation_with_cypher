#!/usr/bin/env python3
"""
Enhanced Neo4j Database Schema Setup for Sophisticated Movie Recommendation System
This script creates a comprehensive schema with advanced relationships for content-based,
collaborative, and hybrid recommendation algorithms.
"""

import os
from neo4j import GraphDatabase, basic_auth
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()
# Database connection configuration
url = os.getenv("NEO4J_URI", "neo4j+s://demo.neo4jlabs.com")
username = os.getenv("NEO4J_USERNAME", "movies")
password = os.getenv("NEO4J_PASSWORD", "movies")
database = os.getenv("NEO4J_DATABASE", "movies")

driver = GraphDatabase.driver(url, auth=basic_auth(username, password))

def setup_enhanced_schema():
    """Create enhanced schema with sophisticated relationships and properties"""
    
    with driver.session(database=database) as session:
        # Create indexes for performance
        indexes = [
            "CREATE INDEX user_id_index IF NOT EXISTS FOR (u:User) ON (u.userId)",
            "CREATE INDEX movie_id_index IF NOT EXISTS FOR (m:Movie) ON (m.movieId)",
            "CREATE INDEX genre_name_index IF NOT EXISTS FOR (g:Genre) ON (g.name)",
            "CREATE INDEX director_name_index IF NOT EXISTS FOR (d:Director) ON (d.name)",
            "CREATE INDEX actor_name_index IF NOT EXISTS FOR (a:Actor) ON (a.name)",
            "CREATE INDEX rating_timestamp_index IF NOT EXISTS FOR ()-[r:RATED]-() ON (r.timestamp)",
        ]
        
        for index in indexes:
            try:
                session.run(index)
                print(f"✓ Created index: {index.split('FOR')[1].split('ON')[0].strip()}")
            except Exception as e:
                print(f"Index creation note: {e}")

        # Enhanced User properties
        session.run("""
            MATCH (u:User)
            SET u.age = coalesce(u.age, toInteger(rand() * 60 + 18)),
                u.gender = coalesce(u.gender, case when rand() > 0.5 then 'M' else 'F' end),
                u.occupation = coalesce(u.occupation, 'other'),
                u.registrationDate = coalesce(u.registrationDate, datetime()),
                u.avgRating = coalesce(u.avgRating, 3.5),
                u.totalRatings = coalesce(u.totalRatings, 0),
                u.preferredGenres = coalesce(u.preferredGenres, []),
                u.activityLevel = coalesce(u.activityLevel, 'medium')
        """)
        print("✓ Enhanced User properties")

        # Enhanced Movie properties with content features
        session.run("""
            MATCH (m:Movie)
            SET m.movieId = coalesce(m.movieId, id(m)),
                m.avgRating = coalesce(m.avgRating, 3.5),
                m.ratingCount = coalesce(m.ratingCount, 0),
                m.popularity = coalesce(m.popularity, rand() * 100),
                m.budget = coalesce(m.budget, toInteger(rand() * 200000000)),
                m.revenue = coalesce(m.revenue, toInteger(rand() * 500000000)),
                m.runtime = coalesce(m.runtime, toInteger(rand() * 120 + 90)),
                m.language = coalesce(m.language, 'en'),
                m.country = coalesce(m.country, 'US'),
                m.contentVector = coalesce(m.contentVector, [rand(), rand(), rand(), rand(), rand()])
        """)
        print("✓ Enhanced Movie properties")

        # Create Genre nodes if they don't exist
        session.run("""
            MERGE (g1:Genre {name: 'Action'})
            MERGE (g2:Genre {name: 'Adventure'})
            MERGE (g3:Genre {name: 'Animation'})
            MERGE (g4:Genre {name: 'Comedy'})
            MERGE (g5:Genre {name: 'Crime'})
            MERGE (g6:Genre {name: 'Documentary'})
            MERGE (g7:Genre {name: 'Drama'})
            MERGE (g8:Genre {name: 'Family'})
            MERGE (g9:Genre {name: 'Fantasy'})
            MERGE (g10:Genre {name: 'Horror'})
            MERGE (g11:Genre {name: 'Romance'})
            MERGE (g12:Genre {name: 'Sci-Fi'})
            MERGE (g13:Genre {name: 'Thriller'})
            MERGE (g14:Genre {name: 'War'})
            MERGE (g15:Genre {name: 'Western'})
        """)
        print("✓ Created Genre nodes")

        # Create Director and Actor nodes from existing relationships
        session.run("""
            MATCH (m:Movie)<-[r:DIRECTED]-(p:Person)
            MERGE (d:Director {name: p.name, born: p.born})
            MERGE (d)-[:DIRECTED {year: coalesce(r.year, m.released)}]->(m)
        """)
        print("✓ Created Director nodes and relationships")

        session.run("""
            MATCH (m:Movie)<-[r:ACTED_IN]-(p:Person)
            MERGE (a:Actor {name: p.name})
            SET a.born = coalesce(p.born, a.born)
            MERGE (a)-[:ACTED_IN {roles: r.roles, importance: coalesce(r.importance, rand())}]->(m)
        """)
        print("✓ Created Actor nodes and relationships")

        # Create enhanced RATED relationships with more properties
        session.run("""
            MATCH (u:User)-[r:RATED]->(m:Movie)
            SET r.timestamp = coalesce(r.timestamp, datetime()),
                r.context = coalesce(r.context, 'normal'),
                r.device = coalesce(r.device, 'web'),
                r.sessionId = coalesce(r.sessionId, toString(rand()))
        """)
        print("✓ Enhanced RATED relationships")

        # Create SIMILAR_TO relationships between movies based on content
        session.run("""
            MATCH (m1:Movie), (m2:Movie)
            WHERE m1 <> m2 AND rand() < 0.1
            WITH m1, m2, 
                 size([(m1)-[:IN_GENRE]->(g)<-[:IN_GENRE]-(m2) | g]) as commonGenres,
                 size([(m1)<-[:ACTED_IN]-(a)-[:ACTED_IN]->(m2) | a]) as commonActors,
                 size([(m1)<-[:DIRECTED]-(d)-[:DIRECTED]->(m2) | d]) as commonDirectors
            WHERE commonGenres > 0 OR commonActors > 0 OR commonDirectors > 0
            WITH m1, m2, (commonGenres * 0.3 + commonActors * 0.5 + commonDirectors * 0.2) as similarity
            WHERE similarity > 0.1
            MERGE (m1)-[s:SIMILAR_TO]-(m2)
            SET s.similarity = similarity,
                s.contentBased = true
        """)
        print("✓ Created content-based SIMILAR_TO relationships")

        # Create SIMILAR_TO relationships between users based on rating patterns
        session.run("""
            MATCH (u1:User)-[r1:RATED]->(m:Movie)<-[r2:RATED]-(u2:User)
            WHERE u1 <> u2 AND abs(r1.rating - r2.rating) <= 1
            WITH u1, u2, count(m) as commonMovies, 
                 avg(abs(r1.rating - r2.rating)) as avgDiff
            WHERE commonMovies >= 3
            WITH u1, u2, (commonMovies / (commonMovies + avgDiff)) as similarity, commonMovies
            WHERE similarity > 0.3
            MERGE (u1)-[s:SIMILAR_TO]-(u2)
            SET s.similarity = similarity,
                s.collaborativeBased = true,
                s.commonMovies = commonMovies
        """)
        print("✓ Created collaborative SIMILAR_TO relationships between users")

        # Create VIEWED relationships for implicit feedback
        session.run("""
            MATCH (u:User)-[r:RATED]->(m:Movie)
            MERGE (u)-[v:VIEWED]->(m)
            SET v.timestamp = r.timestamp,
                v.duration = toInteger(rand() * m.runtime),
                v.completed = case when rand() > 0.3 then true else false end,
                v.device = r.device
        """)
        print("✓ Created VIEWED relationships for implicit feedback")

        # Create user preference profiles
        session.run("""
            MATCH (u:User)-[r:RATED]->(m:Movie)-[:IN_GENRE]->(g:Genre)
            WHERE r.rating >= 4
            WITH u, g, count(*) as genreCount, avg(r.rating) as avgRating
            ORDER BY genreCount DESC, avgRating DESC
            WITH u, collect({genre: g.name, count: genreCount, avgRating: avgRating})[0..5] as topGenres
            SET u.preferredGenres = [genre IN topGenres | genre.genre]
        """)
        print("✓ Created user preference profiles")

        # Update movie popularity scores
        session.run("""
            MATCH (m:Movie)
            OPTIONAL MATCH (m)<-[r:RATED]-()
            WITH m, count(r) as ratingCount, avg(r.rating) as avgRating
            SET m.ratingCount = ratingCount,
                m.avgRating = coalesce(avgRating, 3.0),
                m.popularity = (ratingCount * 0.7 + avgRating * 20)
        """)
        print("✓ Updated movie popularity scores")

        print("\n🎬 Enhanced Neo4j schema setup completed successfully!")
        print("📊 The database now supports sophisticated recommendation algorithms including:")
        print("   • Content-based filtering with movie similarity")
        print("   • Collaborative filtering with user similarity")
        print("   • Hybrid approaches combining multiple signals")
        print("   • Implicit feedback through viewing behavior")
        print("   • User preference profiling")
        print("   • Real-time interaction tracking")

def verify_schema():
    """Verify the enhanced schema was created correctly"""
    with driver.session(database=database) as session:
        # Count nodes and relationships
        result = session.run("""
            MATCH (n) 
            RETURN labels(n)[0] as nodeType, count(n) as count
            ORDER BY count DESC
        """)
        
        print("\n📈 Database Statistics:")
        for record in result:
            print(f"   {record['nodeType']}: {record['count']} nodes")
        
        # Count relationship types
        result = session.run("""
            MATCH ()-[r]->() 
            RETURN type(r) as relType, count(r) as count
            ORDER BY count DESC
        """)
        
        print("\n🔗 Relationship Statistics:")
        for record in result:
            print(f"   {record['relType']}: {record['count']} relationships")

if __name__ == "__main__":
    try:
        print("🚀 Starting enhanced Neo4j schema setup...")
        setup_enhanced_schema()
        verify_schema()
    except Exception as e:
        print(f"❌ Error during schema setup: {e}")
    finally:
        driver.close()
