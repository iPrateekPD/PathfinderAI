# risk_analysis_backend.py

from flask import Flask, request, jsonify, render_template, Response, g
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression
import sqlite3
import json
from datetime import datetime, date, timedelta
import os
from werkzeug.utils import secure_filename
import io
import joblib

from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import google.generativeai as genai

load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='.')
CORS(app)

# --- Configuration ---
DATABASE = 'student_risk.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Centralized file path constants
MODEL_PATH = 'risk_model.joblib'
SCALER_PATH = 'scaler.joblib'
MODEL_COLUMNS_PATH = 'model_columns.joblib'
PERFORMANCE_METRICS_PATH = 'model_performance.json'

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Database Management ---
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON;")
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

# --- Student Risk Analyzer Class ---
class StudentRiskAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_columns = None
        self.is_trained = False
        self.risk_weights = {'attendance': 0.35, 'academic_performance': 0.40, 'fee_status': 0.15, 'backlogs': 0.10}
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.model_columns = joblib.load(MODEL_COLUMNS_PATH)
            self.is_trained = True
            print("Trained model loaded successfully.")
        except FileNotFoundError:
            print("No trained model found. Falling back to rule-based system.")
    
    def calculate_risk_score_rule_based(self, student_data):
        # Handle potential None values from the database
        attendance = student_data.get('attendance_percentage') or 0
        grade = student_data.get('average_grade') or 0
        backlogs = student_data.get('backlogs') or 0
        
        attendance_score = max(0, (60 - attendance) / 60)
        academic_score = max(0, (40 - grade) / 40)
        fee_score = 0 if student_data.get('fee_status', 'paid') == 'paid' else 1
        backlog_score = min(1, backlogs / 5)

        risk_score = (
            attendance_score * self.risk_weights['attendance'] +
            academic_score * self.risk_weights['academic_performance'] +
            fee_score * self.risk_weights['fee_status'] +
            backlog_score * self.risk_weights['backlogs']
        )
        return min(1.0, risk_score)

    def predict_risk_level(self, student_data):
        if not self.is_trained:
            risk_score = self.calculate_risk_score_rule_based(student_data)
            if risk_score >= 0.7: return 'HIGH', risk_score
            elif risk_score >= 0.4: return 'MEDIUM', risk_score
            else: return 'LOW', risk_score
        
        try:
            df_input = pd.DataFrame([student_data])
            df_input = pd.get_dummies(df_input, columns=['fee_status'])
            
            df_processed = df_input.reindex(columns=self.model_columns, fill_value=0)
            
            scaled_data = self.scaler.transform(df_processed)
            prediction = self.model.predict(scaled_data)[0]
            probabilities = self.model.predict_proba(scaled_data)[0]
            risk_score = 0
            classes = list(self.model.classes_)
            if 'HIGH' in classes: risk_score += probabilities[classes.index('HIGH')]
            if 'MEDIUM' in classes: risk_score += probabilities[classes.index('MEDIUM')] * 0.5
            return prediction, min(1.0, risk_score)
        except Exception as e:
            print(f"Error during ML prediction, falling back to rules. Error: {e}")
            risk_score = self.calculate_risk_score_rule_based(student_data)
            if risk_score >= 0.7: return 'HIGH', risk_score
            elif risk_score >= 0.4: return 'MEDIUM', risk_score
            else: return 'LOW', risk_score

risk_analyzer = StudentRiskAnalyzer()

def log_activity(activity_type, description, status='INFO'):
    db = get_db()
    db.execute('''INSERT INTO activity_log (activity_type, description, status, created_at)
                  VALUES (?, ?, ?, ?)''', (activity_type, description, status, datetime.now()))
    db.commit()

@app.route('/api/reset_db', methods=['POST'])
def reset_db():
    try:
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for table in tables:
            table_name = table['name']
            if table_name != 'sqlite_sequence':
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        db.commit()
        init_db()
        for path in [PERFORMANCE_METRICS_PATH, MODEL_PATH, SCALER_PATH, MODEL_COLUMNS_PATH]:
            if os.path.exists(path): os.remove(path)
        
        log_activity('DATABASE_RESET', 'Database was reset successfully.', 'SUCCESS')
        return jsonify({"message": "Database reset successfully."}), 200
    except Exception as e:
        log_activity("DATABASE_RESET_FAILED", str(e), "ERROR")
        return jsonify({"error": f"Failed to reset database: {e}"}), 500

@app.cli.command('initdb')
def initdb_command():
    init_db()
    print('Initialized the database.')

@app.route('/api/train-model', methods=['POST'])
def train_model_endpoint():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    try:
        df = pd.read_csv(file)
        features = ['attendance_percentage', 'average_grade', 'fee_status', 'backlogs']
        target = 'outcome'
        if target not in df.columns: return jsonify({'error': f"Training file must contain an '{target}' column."}), 400
        for col in ['attendance_percentage', 'average_grade', 'backlogs']:
            if col in df.columns:
                df[col].fillna(df[col].mean(), inplace=True)

        X = pd.get_dummies(df[features], columns=['fee_status'], drop_first=True)
        y = df[target]
        model_columns = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        performance_metrics = {'accuracy': accuracy, 'precision': report['weighted avg']['precision'],
                               'recall': report['weighted avg']['recall'], 'f1_score': report['weighted avg']['f1-score']}
        with open(PERFORMANCE_METRICS_PATH, 'w') as f: json.dump(performance_metrics, f)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(model_columns, MODEL_COLUMNS_PATH)
        global risk_analyzer
        risk_analyzer = StudentRiskAnalyzer()
        log_activity('MODEL_TRAINED', f"New model trained with {accuracy*100:.2f}% accuracy.", 'SUCCESS')
        return jsonify({'message': 'Model trained successfully!', 'accuracy': accuracy, 'classification_report': report})
    except Exception as e:
        log_activity("MODEL_TRAINING_FAILED", str(e), "ERROR")
        return jsonify({'error': f'An error occurred during training: {str(e)}'}), 500

@app.route('/')
def index():
    return render_template('student_risk_ui.html')
    
@app.route('/api/students', methods=['GET'])
def get_students():
    db = get_db()
    base_query = 'SELECT student_id, name, grade, section, risk_level, risk_score FROM students'
    params = []
    conditions = []
    search_term = request.args.get('search', '')
    if search_term:
        conditions.append('(name LIKE ? OR student_id LIKE ?)')
        params.extend([f'%{search_term}%', f'%{search_term}%'])
    risk_level = request.args.get('risk', '')
    if risk_level and risk_level != 'all':
        conditions.append('risk_level = ?')
        params.append(risk_level.upper())
    if conditions:
        base_query += ' WHERE ' + ' AND '.join(conditions)
    base_query += ' ORDER BY risk_score DESC'
    students = db.execute(base_query, params).fetchall()
    return jsonify([dict(row) for row in students])

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(io.StringIO(file.stream.read().decode('utf-8', errors='ignore')))
            cleanup_old_students(df)
            processed_records, _ = process_csv_data(df)
            run_risk_analysis_internal()
            log_activity('DATA_UPLOAD', f"Processed {len(processed_records)} records from {secure_filename(file.filename)}.", 'SUCCESS')
            return jsonify({'message': f'Successfully processed {len(processed_records)} records. Analysis complete.'})
        except Exception as e:
            log_activity("DATA_UPLOAD_FAILED", str(e), "ERROR")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file format. Please upload CSV files only.'}), 400

def cleanup_old_students(df):
    db = get_db()
    new_student_ids = set(df['student_id'].astype(str).tolist())
    db_student_ids = set([row['student_id'] for row in db.execute("SELECT student_id FROM students").fetchall()])
    ids_to_remove = db_student_ids - new_student_ids
    if ids_to_remove:
        db.executemany("DELETE FROM students WHERE student_id = ?", [(id,) for id in ids_to_remove])
        db.commit()

# ## MODIFIED FUNCTION ##
def process_csv_data(df):
    column_mapping = {
        'student_id': ['student_id', 'Student_ID'], 'name': ['name', 'Name'],
        'grade': ['grade', 'Grade'], 'section': ['section', 'Section'], 'phone': ['phone', 'Phone'], 
        'email': ['email', 'Email'], 'parent_name': ['parent_name', 'Parent_Name'], 
        'parent_phone': ['parent_phone', 'Parent_Phone'], 'attendance_percentage': ['attendance_percentage', 'Attendance_Percentage'],
        'average_grade': ['average_grade', 'Average_Grade'], 'fee_status': ['fee_status', 'Fee_Status'], 
        'backlogs': ['backlogs', 'Backlogs']
    }
    csv_to_db_map = {csv_opt: db_col for db_col, csv_options in column_mapping.items() for csv_opt in csv_options if csv_opt in df.columns}
    if 'student_id' not in csv_to_db_map.values(): return [], ["CSV must contain a 'student_id' column."]
    df.rename(columns=csv_to_db_map, inplace=True)
    df_columns = [col for col in csv_to_db_map.values() if col in df.columns]
    rows_to_insert = [tuple(row) for row in df[df_columns].to_numpy()]

    if rows_to_insert:
        db = get_db()
        db_cols_str = ', '.join(df_columns)
        placeholders = ', '.join(['?'] * len(df_columns))
        
        # Using a non-destructive UPSERT command instead of INSERT OR REPLACE
        # This updates existing students without deleting them, which prevents the ON DELETE CASCADE on alerts
        update_clause = ', '.join([f"{col} = excluded.{col}" for col in df_columns if col != 'student_id'])
        
        sql_query = f"""
            INSERT INTO students ({db_cols_str}) VALUES ({placeholders})
            ON CONFLICT(student_id) DO UPDATE SET {update_clause}
        """
        
        db.executemany(sql_query, rows_to_insert)
        db.commit()
    return rows_to_insert, []

def run_risk_analysis_internal():
    db = get_db()
    students = db.execute('SELECT * FROM students').fetchall()
    updates, history_logs, alerts = [], [], []
    analysis_date = datetime.now().date()
    for student in students:
        student_data = dict(student)
        risk_level, risk_score = risk_analyzer.predict_risk_level(student_data)
        updates.append((risk_level, risk_score, datetime.now(), student['student_id']))
        history_logs.append((student['student_id'], risk_score, analysis_date))
        if risk_level == 'HIGH' and risk_score > 0.8:
            existing = db.execute("SELECT id FROM alerts WHERE student_id = ? AND alert_type = ? AND status = 'ACTIVE'", 
                                  (student['student_id'], 'CRITICAL_RISK')).fetchone()
            if not existing:
                alerts.append((student['student_id'], 'CRITICAL_RISK', f'Critical risk detected for {student["name"]}.',
                               'HIGH', 'ACTIVE', datetime.now()))
    if updates: db.executemany('UPDATE students SET risk_level=?, risk_score=?, last_updated=? WHERE student_id=?', updates)
    if history_logs: db.executemany('INSERT INTO risk_history (student_id, risk_score, analysis_date) VALUES (?,?,?)', history_logs)
    if alerts: db.executemany('INSERT INTO alerts (student_id, alert_type, message, priority, status, created_at) VALUES (?,?,?,?,?,?)', alerts)
    db.commit()
    if len(students) > 0: log_activity('ANALYSIS_RUN', f"Analyzed {len(students)} students.", 'SUCCESS')
    return {'message': 'Risk analysis completed', 'students_analyzed': len(students), 'alerts_created': len(alerts)}

@app.route('/api/analyze', methods=['POST'])
def run_risk_analysis_api(): return jsonify(run_risk_analysis_internal())

# (The rest of the file is unchanged)
# ...
@app.route('/api/student/<student_id>', methods=['GET'])
def get_student_profile(student_id):
    db = get_db()
    student = db.execute('SELECT * FROM students WHERE student_id = ?', (student_id,)).fetchone()
    if not student: return jsonify({'error': 'Student not found'}), 404
    student_data = dict(student)
    student_data['alerts'] = [dict(r) for r in db.execute('SELECT * FROM alerts WHERE student_id=? ORDER BY created_at DESC LIMIT 5', (student_id,)).fetchall()]
    student_data['communications'] = [dict(r) for r in db.execute('SELECT * FROM communications WHERE student_id=? ORDER BY sent_at DESC LIMIT 5', (student_id,)).fetchall()]
    student_data['risk_history'] = [dict(r) for r in db.execute('SELECT * FROM risk_history WHERE student_id=? ORDER BY analysis_date DESC LIMIT 15', (student_id,)).fetchall()]
    student_data['interventions'] = [dict(r) for r in db.execute('SELECT * FROM interventions WHERE student_id=? ORDER BY created_at DESC', (student_id,)).fetchall()]
    return jsonify(student_data)

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    db = get_db()
    alerts = db.execute('''SELECT a.id, s.name as student_name, a.* FROM alerts a JOIN students s ON a.student_id = s.student_id
                           WHERE a.status = 'ACTIVE' ORDER BY a.created_at DESC''').fetchall()
    return jsonify([dict(row) for row in alerts])
    
@app.route('/api/alert/resolve/<int:alert_id>', methods=['POST'])
def resolve_alert(alert_id):
    db = get_db()
    db.execute('UPDATE alerts SET status = ?, resolved_at = ? WHERE id = ?', ('RESOLVED', datetime.now(), alert_id))
    db.commit()
    return jsonify({'message': f'Alert {alert_id} has been resolved.'})

def send_real_communication(comm_type, recipient, message):
    if comm_type.lower()!='email' or not SENDGRID_API_KEY or not SENDER_EMAIL: return False
    mail_object = Mail(from_email=SENDER_EMAIL, to_emails=recipient, subject='Important Student Update', html_content=f'<p>{message}</p>')
    try:
        SendGridAPIClient(SENDGRID_API_KEY).send(mail_object)
        return True
    except Exception as e:
        print(f"Error sending email via SendGrid: {e}")
        return False

@app.route('/api/communicate', methods=['POST'])
def send_communication():
    data = request.json
    success = send_real_communication(data['type'], data['recipient'], data['message'])
    if success:
        db = get_db()
        db.execute('INSERT INTO communications (student_id, type, recipient, message, status, sent_at) VALUES (?,?,?,?,?,?)',
                   (data['student_id'], data['type'], data['recipient'], data['message'], 'SENT', datetime.now()))
        db.commit()
        return jsonify({'message': f'{data["type"].upper()} sent successfully'})
    return jsonify({'error': 'Failed to send communication.'}), 500

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_stats():
    try:
        with open(PERFORMANCE_METRICS_PATH, 'r') as f: model_performance = json.load(f)
    except FileNotFoundError: model_performance = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0}
    db = get_db()
    total_students = db.execute('SELECT COUNT(*) FROM students').fetchone()[0]
    risk_counts = dict(db.execute('SELECT risk_level, COUNT(*) FROM students GROUP BY risk_level').fetchall())
    active_alerts = db.execute('SELECT COUNT(*) FROM alerts WHERE status = "ACTIVE"').fetchone()[0]
    return jsonify({'total_students': total_students, 'risk_distribution': {'high': risk_counts.get('HIGH', 0), 'medium': risk_counts.get('MEDIUM', 0), 'low': risk_counts.get('LOW', 0)},
        'active_alerts': active_alerts, 'model_performance': model_performance})

@app.route('/api/recent-activity', methods=['GET'])
def get_recent_activity():
    activities = get_db().execute('SELECT * FROM activity_log ORDER BY created_at DESC LIMIT 5').fetchall()
    return jsonify([dict(row) for row in activities])

@app.route('/api/report/comprehensive', methods=['GET'])
def generate_comprehensive_report():
    try:
        db = get_db()
        students = db.execute('SELECT * FROM students ORDER BY risk_score DESC').fetchall()
        if not students:
            return jsonify({"error": "No student data available to generate a report."}), 404
        
        df = pd.DataFrame([dict(row) for row in students])
        csv_data = df.to_csv(index=False)
        
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition":
                     "attachment; filename=student_risk_report.csv"})
    except Exception as e:
        log_activity("REPORT_GENERATION_FAILED", str(e), "ERROR")
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500

@app.route('/api/student/<student_id>/report/txt', methods=['GET'])
def generate_student_txt_report(student_id):
    try:
        db = get_db()
        student = db.execute('SELECT * FROM students WHERE student_id = ?', (student_id,)).fetchone()
        if not student:
            return jsonify({"error": "Student not found."}), 404

        report_lines = []
        report_lines.append(f"STUDENT RISK REPORT - ID: {student['student_id']}")
        report_lines.append("="*40)
        
        report_lines.append("\n[ STUDENT PROFILE ]")
        for key, value in dict(student).items():
            if key != 'id':
                report_lines.append(f"{key.replace('_', ' ').title():<25}: {value}")

        history = db.execute('SELECT * FROM risk_history WHERE student_id=? ORDER BY analysis_date DESC LIMIT 10', (student_id,)).fetchall()
        report_lines.append("\n[ RECENT RISK HISTORY ]")
        if history:
            for item in history:
                report_lines.append(f"- {item['analysis_date'].strftime('%Y-%m-%d')}: Score {item['risk_score']:.2f}")
        else:
            report_lines.append("No risk history found.")

        interventions = db.execute('SELECT * FROM interventions WHERE student_id=? ORDER BY created_at DESC', (student_id,)).fetchall()
        report_lines.append("\n[ LOGGED INTERVENTIONS ]")
        if interventions:
            for item in interventions:
                report_lines.append(f"- Task: {item['task']} (Status: {item['status']})")
                report_lines.append(f"  Assigned to: {item['assigned_to'] or 'N/A'}, Due: {item['due_date'] or 'N/A'}")
        else:
            report_lines.append("No interventions logged.")

        report_content = "\n".join(report_lines)
        
        return Response(
            report_content,
            mimetype="text/plain",
            headers={"Content-disposition":
                     f"attachment; filename=report-{student_id}.txt"})
    except Exception as e:
        log_activity("TXT_REPORT_FAILED", str(e), "ERROR")
        return jsonify({"error": f"Failed to generate text report: {str(e)}"}), 500

@app.route('/api/weights', methods=['POST'])
def update_weights():
    data = request.json
    try:
        weights = {k: float(data.get(k, 0)) / 100 for k in ['attendance', 'academic', 'financial', 'backlogs']}
        if not np.isclose(sum(weights.values()), 1.0): return jsonify({'error': 'Weights must sum to 100'}), 400
        risk_analyzer.risk_weights.update({'attendance': weights['attendance'], 'academic_performance': weights['academic'], 
                                           'fee_status': weights['financial'], 'backlogs': weights['backlogs']})
        return jsonify({'message': 'Risk model weights updated successfully.'})
    except (ValueError, TypeError): return jsonify({'error': 'Invalid weight values provided.'}), 400

@app.route('/api/student/<student_id>/interventions', methods=['POST'])
def add_intervention(student_id):
    data = request.json
    due_date = datetime.strptime(data['due_date'], '%Y-%m-%d').date() if data.get('due_date') else None
    db = get_db()
    cursor = db.execute('INSERT INTO interventions (student_id, task, assigned_to, due_date, created_at, last_updated) VALUES (?,?,?,?,?,?)', 
                      (student_id, data['task'], data.get('assigned_to'), due_date, datetime.now(), datetime.now()))
    db.commit()
    log_activity('INTERVENTION_ADDED', f"New intervention for {student_id}: {data['task']}", 'SUCCESS')
    return jsonify({'message': 'Intervention added', 'id': cursor.lastrowid}), 201

@app.route('/api/interventions/<int:intervention_id>', methods=['PUT'])
def update_intervention(intervention_id):
    data = request.json
    db = get_db()
    db.execute("UPDATE interventions SET status=?, notes=?, last_updated=? WHERE id=?", (data['status'], data.get('notes'), datetime.now(), intervention_id))
    db.commit()
    log_activity('INTERVENTION_UPDATED', f"Intervention {intervention_id} status set to {data['status']}", 'SUCCESS')
    return jsonify({'message': 'Intervention updated'})

@app.route('/api/interventions/<int:intervention_id>', methods=['DELETE'])
def remove_intervention(intervention_id):
    db = get_db()
    cursor = db.execute("DELETE FROM interventions WHERE id = ?", (intervention_id,))
    db.commit()
    if cursor.rowcount == 0: return jsonify({'error': 'Intervention not found'}), 404
    log_activity('INTERVENTION_REMOVED', f"Intervention {intervention_id} was removed.", 'SUCCESS')
    return jsonify({'message': 'Intervention removed successfully'})

@app.route('/api/student/<student_id>/what-if', methods=['POST'])
def what_if_scenario(student_id):
    data = request.json
    db = get_db()
    student = db.execute('SELECT * FROM students WHERE student_id = ?', (student_id,)).fetchone()
    if not student: return jsonify({'error': 'Student not found'}), 404
    hypothetical_data = dict(student)
    for key, value in data.items():
        if value is not None and value != '':
            try:
                if key in ['attendance_percentage', 'average_grade']: hypothetical_data[key] = float(value)
                elif key == 'backlogs': hypothetical_data[key] = int(value)
                else: hypothetical_data[key] = value
            except (ValueError, TypeError): return jsonify({'error': f'Invalid value for {key}'}), 400
    risk_level, risk_score = risk_analyzer.predict_risk_level(hypothetical_data)
    return jsonify({'hypothetical_risk_level': risk_level, 'hypothetical_risk_score': risk_score})

@app.route('/api/student/<student_id>/forecast', methods=['GET'])
def forecast_risk(student_id):
    db = get_db()
    history = db.execute('SELECT risk_score, analysis_date FROM risk_history WHERE student_id=? ORDER BY analysis_date ASC', (student_id,)).fetchall()
    if len(history) < 3: return jsonify({'forecast': []})
    scores = [h['risk_score'] for h in history]
    X = np.array(range(len(scores))).reshape(-1, 1)
    y = np.array(scores)
    model = LinearRegression().fit(X, y)
    future_steps = np.array(range(len(scores), len(scores) + 4)).reshape(-1, 1)
    predictions = model.predict(future_steps)
    last_date = history[-1]['analysis_date']
    forecast_dates = [(last_date + timedelta(days=(i+1)*7)) for i in range(4)]
    forecast_data = [{'date': d.isoformat(), 'score': max(0, min(1.0, p))} for d, p in zip(forecast_dates, predictions)]
    return jsonify({'forecast': forecast_data})

@app.route('/api/student/<student_id>/ai-recommendation', methods=['POST'])
def ai_recommendation(student_id):
    db = get_db()
    student = db.execute('SELECT * FROM students WHERE student_id = ?', (student_id,)).fetchone()
    if not student: return jsonify({'error': 'Student not found'}), 404
    student_data, recommendations = dict(student), []
    if student_data.get('attendance_percentage', 100) < 75: recommendations.append({'task': 'Schedule Parent Meeting on attendance', 'priority': 'High'})
    if student_data.get('average_grade', 100) < 50: recommendations.append({'task': 'Arrange peer tutoring for key subjects', 'priority': 'High'})
    if student_data.get('backlogs', 0) > 2: recommendations.append({'task': 'Create backlog clearance plan with advisor', 'priority': 'Medium'})
    if not recommendations: recommendations.append({'task': 'Schedule general progress check-in', 'priority': 'Low'})
    return jsonify({'recommendations': recommendations})

@app.route('/api/student/<student_id>/ai-draft', methods=['POST'])
def ai_draft_communication(student_id):
    if not GEMINI_API_KEY: return jsonify({'error': 'Generative AI is not configured.'}), 500
    data = request.json
    context = data.get('context')
    db = get_db()
    student = db.execute('SELECT * FROM students WHERE student_id = ?', (student_id,)).fetchone()
    if not student: return jsonify({'error': 'Student not found'}), 404
    prompt = f"""You are a compassionate Indian school counselor. Student: {student['name']}, Grade: {student['grade']}, Parent: {student['parent_name']}. Key issue: {context}. Attendance: {student['attendance_percentage']}%, Grade: {student['average_grade']}. Draft a short, supportive SMS (under 160 characters) to {student['parent_name']} to schedule a brief meeting. Be professional, not alarming. Start with "Dear {student['parent_name']}"."""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return jsonify({'draft': response.text.strip()})
    except Exception as e:
        return jsonify({'error': f'Failed to generate AI draft: {str(e)}'}), 500

if __name__ == '__main__':
    if not os.path.exists(DATABASE):
        print("Database not found. Creating a new one...")
        init_db()
        print("Database created successfully.")
    print("Student Risk Analysis System starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)