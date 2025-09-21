-- schema.sql
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT UNIQUE, name TEXT,
    grade TEXT, section TEXT, phone TEXT, email TEXT, parent_name TEXT, parent_phone TEXT,
    attendance_percentage REAL, average_grade REAL, fee_status TEXT,
    backlogs INTEGER, risk_level TEXT, risk_score REAL, last_updated TIMESTAMP
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, alert_type TEXT,
    message TEXT, priority TEXT, status TEXT, created_at TIMESTAMP, resolved_at TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS communications (
    id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, type TEXT, recipient TEXT,
    message TEXT, status TEXT, sent_at TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, activity_type TEXT NOT NULL,
    description TEXT, status TEXT, created_at TIMESTAMP
);
CREATE TABLE IF NOT EXISTS risk_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, risk_score REAL,
    analysis_date DATE,
    FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS interventions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, task TEXT NOT NULL,
    assigned_to TEXT, status TEXT DEFAULT 'To Do', notes TEXT,
    created_at TIMESTAMP, last_updated TIMESTAMP, due_date DATE,
    FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
);