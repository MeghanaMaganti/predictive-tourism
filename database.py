import sqlite3

def init_db():
    conn = sqlite3.connect('tourism.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT, email TEXT UNIQUE, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS uploads (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER, filename TEXT, upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def add_user(name, email, password):
    conn = sqlite3.connect('tourism.db')
    c = conn.cursor()
    c.execute('INSERT INTO users (name,email,password) VALUES (?,?,?)',(name,email,password))
    conn.commit()
    conn.close()

def check_user(email, password):
    conn = sqlite3.connect('tourism.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email=? AND password=?',(email,password))
    data = c.fetchone()
    conn.close()
    return data
