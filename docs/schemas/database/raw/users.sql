-- Core user demographic information
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,  -- Unique identifier for each user
    occupation_category TEXT,  -- User's occupation category
    age_range TEXT,           -- User's age range
    family_status TEXT,       -- User's family status
    region TEXT              -- User's regional information
);