-- Financial transaction records
CREATE TABLE transactions (
    transaction_id TEXT PRIMARY KEY,  -- Unique identifier for each transaction
    user_id TEXT,                    -- References USERS.user_id
    transaction_date TEXT,           -- Date of the transaction
    amount REAL,                     -- Transaction amount
    category TEXT,                   -- Transaction category
    subcategory TEXT,                -- Transaction subcategory
    vendor TEXT,                     -- Vendor information
    description TEXT,                -- Transaction details
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);