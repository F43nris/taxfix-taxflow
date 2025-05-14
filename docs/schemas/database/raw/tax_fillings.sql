-- Annual tax filing records
CREATE TABLE tax_filings (
    filing_id TEXT PRIMARY KEY,  -- Unique identifier for each tax filing
    user_id TEXT,               -- References USERS.user_id
    total_income REAL,          -- Total income for the filing
    total_deductions REAL,      -- Total deductions for the filing
    refund_amount REAL,         -- Refund amount
    tax_year INTEGER,          -- Tax year
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);