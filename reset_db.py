#!/usr/bin/env python3
"""
Quick database reset script to fix authentication issue
"""
import os
import sys
import bcrypt

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def reset_db():
    """Reset database with proper password hashing"""
    from app.db.models import Base, User, Document
    from app.db.session import db_config
    
    print("🔄 Resetting database...")
    
    # Drop and recreate tables
    Base.metadata.drop_all(bind=db_config.engine)
    Base.metadata.create_all(bind=db_config.engine)
    
    print("✅ Tables recreated")
    
    # Create session and add users
    session = db_config.SessionLocal()
    
    try:
        # Create users with properly hashed passwords using bcrypt directly
        def hash_password(password: str) -> str:
            return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        users = [
            User(
                username="admin",
                email="admin@mediassist.com", 
                hashed_password=hash_password("admin123"),
                role="admin"
            ),
            User(
                username="tech1",
                email="tech1@lab.com",
                hashed_password=hash_password("tech123"),
                role="technician"
            ),
            User(
                username="readonly", 
                email="readonly@lab.com",
                hashed_password=hash_password("readonly123"),
                role="readonly"
            )
        ]
        
        for user in users:
            session.add(user)
        
        session.commit()
        
        print("✅ Users created with proper password hashing:")
        print("   admin / admin123")
        print("   tech1 / tech123") 
        print("   readonly / readonly123")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error creating users: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    reset_db()